import random

import gym
import pandas as pd
from stable_baselines3 import PPO
import cv2
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from gym import spaces
import os
import imageio

img_dir = "Experiment2_images"
gif_dir = "Experiment2_gifs"

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)

# Load the environment
class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImagePreprocessingWrapper, self).__init__(env)
        # Update the observation space after image preprocessing
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        # Convert image to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Add a new axis to create a single-channel image
        obs = np.expand_dims(obs, axis=-1)

        # Crop the dashboard from the image (bottom 12 pixels) and remove 6 pixels from both side for size 84x84
        obs = obs[0:84, 6:90, :]

        return obs


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        # Define the discrete action space
        self.action_space = spaces.Discrete(7)

    def action(self, action):
        # Map the discrete actions to continuous actions
        if action == 0:
            return [1, 0, 0]  # Turn right
        elif action == 1:
            return [-1, 0, 0]  # Turn left
        elif action == 2:
            return [0, 1, 0]  # Accelerate
        elif action == 3:
            return [0, 0, 0.8]  # Brake
        elif action == 4:
            return [0, 0, 0]  # No action
        elif action == 5:
            return [0.5, 0, 0]  # Turn right slightly
        elif action == 6:
            return [-0.5, 0, 0]  # Turn left slightly


class RainWrapper(gym.ObservationWrapper):
    def __init__(self, env, drop_probability, drop_intensity):
        super().__init__(env)
        self.drop_probability = drop_probability
        self.drop_intensity = drop_intensity

    def observation(self, obs):
        # add noise to the observation
        h, w, _ = obs.shape

        mask = np.random.random((h, w)) < self.drop_probability
        obs[mask] = np.clip(obs[mask] * self.drop_intensity, 0, 255).astype(np.uint8)
        return obs


class NightWrapper(gym.ObservationWrapper):
    def __init__(self, env, brightness_factor):
        super().__init__(env)
        self.brightness_factor = brightness_factor

    def observation(self, obs):
        # # Decrease brightness of the rgb image by brightness_factor
        obs = np.clip(obs * self.brightness_factor, 0, 255).astype(np.uint8)
        return obs


# Function to create a new environment instance with modifications
def create_modified_env(base_env, experiment):
    env = base_env

    if experiment == "midnight":
        # Decrease brightness to 50% of the original value
        env = NightWrapper(env, brightness_factor=0.5)
    elif experiment == "night":
        # Decrease brightness to 80% of the original value
        env = NightWrapper(env, brightness_factor=0.8)
    elif experiment == "heavyrain":
        # Add rain with drop probability of 0.1 and intensity of 0.5
        env = RainWrapper(env, drop_probability=0.1, drop_intensity=0.5)
    elif experiment == "lightrain":
        # Add rain with drop probability of 0.01 and intensity of 0.7
        env = RainWrapper(env, drop_probability=0.05, drop_intensity=0.7)

    return env


# Set random seed for track generation consistency
seed = 20216554
random.seed(seed)
np.random.seed(seed)

# Create list of models
model_names = ["definitive-model", "benchmark-model"]

# Create list of experiment names
experiments = ["None", "midnight", "night", "heavyrain", "lightrain"]

n_episodes = 20  # Number of test episodes to run

# experiment results list
experiment_results = []

# Iterate through all experiments
for experiment in experiments:
    # Load a new racing car environment
    env = gym.make("CarRacing-v0")

    # Create a copy of the environment with the experiment modifications
    env = create_modified_env(env, experiment)

    # Set random seed for track generation consistency
    env.seed(seed)

    # Iterate through all models
    for model_name in model_names:
        # Apply the preprocessing wrappers for definitive model
        if model_name == "definitive-model":
            # Apply the custom image preprocessing wrapper
            env_copy = ImagePreprocessingWrapper(env)
            # Apply the custom discrete action wrapper
            env_copy = DiscreteActionWrapper(env_copy)
            env_copy = DummyVecEnv([lambda: env_copy])
            env_copy = VecFrameStack(env_copy, n_stack=4)
            # Load the pretrained definitive model
            model = PPO.load(f"models_20216554/PPO_cnn_all_1m/830000.zip")

        else:
            # Create a copy of the environment with no modifications
            env_copy = create_modified_env(env, experiment)
            model = PPO.load(f"models_20216554/PPO_cnn_1m/380000.zip")

        scores = []  # List of scores for each episode
        print(f"Starting {model_name}_{experiment}")

        # Set random seed for track generation consistency
        env_copy.seed(seed)

        # Save image and gif flag to save only once per experiment+model
        save_image = True
        save_gif = False
        frames = []
        counter = 0

        for episode in range(n_episodes):
            obs = env_copy.reset()  # Reset the environment for a new episode
            done = False
            score = 0

            while not done:
                counter += 1
                action, _ = model.predict(obs.copy())  # Use the pretrained model to select actions
                obs, reward, done, _ = env_copy.step(action)  # Interact with the environment

                if save_image and counter == 300:
                    # Save the image
                    cv2.imwrite(f"{img_dir}/{model_name}_{experiment}.jpg", obs)
                    save_image = False

                score += reward  # Accumulate rewards

                # Save the frames for the gif
                if save_gif:
                    frame = env_copy.render(mode='rgb_array')
                    frames.append(frame)

            scores.append(score)

        print(scores)
        # Calculate the mean and std deviation of the scores
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        std_deviation = np.std(scores)

        # Store results for each experiment
        experiment_results.append([f"{model_name}_{experiment}", mean_score, max_score, std_deviation])
        print(f"Done {model_name}_{experiment}")

        # Save GIF
        if save_gif:
            imageio.mimsave(f"{gif_dir}/{model_name}_{experiment}.gif", frames, fps=20)

# Save the results to a csv file
df = pd.DataFrame(experiment_results, columns=["Experiment Type", "Mean Score", "Max Score", "Std. Deviation"])
df.to_csv("experiment_results.csv", index=False)
