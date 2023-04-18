import gym
from stable_baselines3 import PPO
import cv2
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import imageio

experiment_name = "PPO_cnn_gray"
model_timestep = 290000
vid_dir = f"videos/{experiment_name}"
if not os.path.exists(vid_dir):
    os.makedirs(vid_dir)


# Load the environment
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

# Load the saved model
model = PPO.load(f"models/{experiment_name}/{model_timestep}.zip")

# Reset the environment to get the initial observation
obs = env.reset()
seed = env.seed()[0]

images = []
img = env.render(mode='rgb_array')

while True:
    images.append(img)
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    img = env.render(mode='rgb_array')
    if done:
        obs = env.reset()
        break

imageio.mimsave(f'{vid_dir}/{experiment_name}/{model_timestep}_{seed}.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
