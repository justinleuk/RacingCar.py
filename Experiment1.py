import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from gym import spaces
import os
import cv2
import numpy as np

experiment_name = "PPO_cnn_1m_2"

models_dir = f"models_20216554/{experiment_name}"
logdir = "logs_20216554"
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


class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImagePreprocessingWrapper, self).__init__(env)
        # Update the observation space after image preprocessing
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        # Convert image to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Add a new axis to create a single-channel
        obs = np.expand_dims(obs, axis=-1)

        # # # Crop the dashboard from the image (bottom 12 pixels) and remove 6 pixels from both side for size 84x84
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


# Load racing car environment
env = gym.make("CarRacing-v0")
# Apply the custom image preprocessing wrapper
env = ImagePreprocessingWrapper(env)
# Apply the custom discrete action wrapper
env = DiscreteActionWrapper(env)

# Monitor the environment statistics
env = Monitor(env, logdir)

# Wrap the environment in a DummyVecEnv wrapper
env = DummyVecEnv([lambda: env])

# Set random seed for reproducibility
env.seed(20216554)
set_random_seed(seed=20216554)

obs = env.reset()

# Stack 4 consecutive frames
env = VecFrameStack(env, n_stack=4)

model = PPO(policy="CnnPolicy", env=env, verbose=1, device="cpu", tensorboard_log=logdir)

time_steps = 10000

for i in range(1, 100): 
    # Train the model with the callbacks
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=experiment_name)
    # Save model every time_steps*i
    model.save(f"{models_dir}/{time_steps * i}")

env.close()
