import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import os
import imageio
import cv2
import numpy as np

experiment_name = "PPO_cnn"

models_dir = f"models/{experiment_name}"
logdir = "logs"
img_dir = f"images/{experiment_name}"
vid_dir = f"videos/{experiment_name}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(vid_dir):
    os.makedirs(vid_dir)

# Define custom callback to save best model
class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.save_path = models_dir
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        mean_reward = self.model.logger.get_stats('ep_rew_mean')[0][1]
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.save_path)
        return True

class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImagePreprocessingWrapper, self).__init__(env)
        # Update the observation space after image preprocessing
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(72, 96, 3), dtype=np.uint8)

    def observation(self, obs):
        # Convert image to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Crop the image to retain only the top part
        obs = obs[0:65, :]

        # Convert grayscale image to RGB image
        obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2RGB)

        return obs

# Custom wrapper to convert CarRacing environment into a vectorized environment
class CarRacingVecEnv(gym.Wrapper):
    def __init__(self, env):
        super(CarRacingVecEnv, self).__init__(env)
        self.num_envs = 1

    def reset(self):
        return [self.env.reset()]

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions[0])
        return [obs], [reward], [done], [info]

# Load racing car environment
env = gym.make("CarRacing-v0")
# Apply the custom image preprocessing wrapper
env = ImagePreprocessingWrapper(env)
# Convert CarRacing environment into a vectorized environment
env = CarRacingVecEnv(env)
# Wrap the vectorized environment with DummyVecEnv
env = DummyVecEnv([lambda: env])
# Reset the environment to get the initial observation
obs = env.reset()

# Load model
model = PPO("CnnPolicy", env, verbose=1, device="cpu", tensorboard_log=logdir)

# Create callback to save the best model
save_best_model_callback = SaveBestModelCallback(f"{models_dir}/best_model", verbose=1)

# Create callback to stop training on reward threshold
stop_training_callback = StopTrainingOnRewardThreshold(reward_threshold=950)

time_steps = 10000
# Create video recorder
video_recorder = VecVideoRecorder(env, vid_dir, record_video_trigger=lambda x: x == (time_steps - 1), video_length=300)

for i in range(1,30):
    # Train the model with the callbacks
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="PPO_cnn", callback=[save_best_model_callback, stop_training_callback, video_recorder])
    # Save model every time_steps*i
    model.save(f"{models_dir}/{time_steps*i}")

    # Check if the reward threshold has been reached
    if stop_training_callback.reward_threshold_reached:
        print("Training stopped: Reward threshold reached!")
        break

env.close()

'''episodes = 10
# 
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)'''

