import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import imageio
import cv2
import numpy as np

experiment_name = "PPO_cnn_gray_1m"
models_dir = f"models/{experiment_name}"
logdir = "logs"
img_dir = f"images/{experiment_name}"
vid_dir = f"videos/{experiment_name}"
start_timesteps = 290000
total_timesteps = 1000000

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(vid_dir):
    os.makedirs(vid_dir)

class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImagePreprocessingWrapper, self).__init__(env)
        # Update the observation space after image preprocessing
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 1), dtype=np.uint8)

    def observation(self, obs):
        # Convert image to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # # Crop the image to retain only the top part
        # obs = obs[:65, :]

        # Add a new axis to create a single-channel image
        obs = np.expand_dims(obs, axis=-1)

        return obs

# Load racing car environment
env = gym.make("CarRacing-v0")
# Apply the custom image preprocessing wrapper
env = ImagePreprocessingWrapper(env)
# Reset the environment to get the initial observation
obs = env.reset()

# Load model from the saved checkpoint
model = PPO.load(f"{models_dir}/{start_timesteps}")
time_steps = 10000
# Create evaluation callback to monitor episode rewards during evaluation
# eval_callback = EvalCallback(env, best_model_save_path=models_dir, log_path=logdir, eval_freq=1, n_eval_episodes=10)
remaining_timesteps = int((total_timesteps - start_timesteps)/10000)
for i in range(1, remaining_timesteps):
    # Train the model with the callbacks
    # model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="PPO_cnn", callback=[save_best_model_callback, stop_training_callback])
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=experiment_name)
    # Save model every total_timesteps*i
    model.save(f"{models_dir}/{time_steps*i}")

env.close()
