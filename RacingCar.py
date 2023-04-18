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

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(vid_dir):
    os.makedirs(vid_dir)

# Define custom callback to save best model
# class SaveBestModelCallback(BaseCallback):
#     def __init__(self, save_path, verbose=0):
#         super(SaveBestModelCallback, self).__init__(verbose)
#         self.save_path = save_path
#         self.best_mean_reward = -float('inf')
#
#     def _on_step(self) -> bool:
#         if isinstance(self.training_env.env.env, Monitor):
#             if self.training_env.env.env.get_episode_rewards():  # Check if episode is done
#                 episode_rewards = self.training_env.env.env.get_episode_rewards()
#                 mean_reward = np.mean(episode_rewards)
#                 if mean_reward > self.best_mean_reward:
#                     self.best_mean_reward = mean_reward
#                     self.model.save(self.save_path)
#         else:
#             mean_reward = np.mean(self.training_env.get_attr('episode_rewards')[-100:])
#             if mean_reward > self.best_mean_reward:
#                 self.best_mean_reward = mean_reward
#                 self.model.save(self.save_path)
#         return True

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

# Load model
model = PPO("CnnPolicy", env, verbose=1, device="cpu", tensorboard_log=logdir)

# # Create callback to save the best model
# save_best_model_callback = SaveBestModelCallback(f"{models_dir}/best_model", verbose=1)

# Create callback to stop training on reward threshold
# stop_training_callback = StopTrainingOnRewardThreshold(reward_threshold=950)

#Create evaluation callback to monitor episode rewards during evaluation
# eval_callback = EvalCallback(env, best_model_save_path=models_dir, log_path=logdir, eval_freq=1, n_eval_episodes=10)

time_steps = 10000

for i in range(1,100):
    # Train the model with the callbacks
    # model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="PPO_cnn", callback=[save_best_model_callback, stop_training_callback])
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=experiment_name)
    # Save model every time_steps*i
    model.save(f"{models_dir}/{time_steps*i}")

    # # Check if the reward threshold has been reached
    # if stop_training_callback.reward_threshold_reached:
    #     print("Training stopped: Reward threshold reached!")
    #     break

env.close()

'''episodes = 10
# 
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)'''

