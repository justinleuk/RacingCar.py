import gym
from stable_baselines3 import PPO, A2C
import os
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

# Define your custom observation wrapper
class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImagePreprocessingWrapper, self).__init__(env)
        # Update the observation space after image preprocessing
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(65, 96), dtype=np.uint8)

    def observation(self, obs):
        # Convert image to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Crop the image to retain only the top part
        obs = obs[:65, :]

        # # # Resize the image back to original
        # obs = cv2.resize(obs, (96, 72))

        # # Convert grayscale image to RGB image
        # obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2RGB)
        # obs = np.expand_dims(obs, axis=-1)

        return obs

# Create the original Gym environment
env = gym.make("CarRacing-v0")

# Apply the custom image preprocessing wrapper
env = ImagePreprocessingWrapper(env)
# Reset the environment to get the initial observation
obs = env.reset()

# # Set a counter for frames
frame_counter = 0
#
# Loop for a few steps to collect observations
for i in range(1000):  # Replace 10 with the number of timesteps you want to collect observations for
    action = env.action_space.sample()  # Replace with your desired action selection strategy
    obs, reward, done, info = env.step(action)
    frame_counter += 1

    # Display the image
    cv2.imshow('Processed Observation', obs)
    cv2.waitKey(0)
    print(obs.shape)
    # Save the image every 20 frames
    if frame_counter % 20 == 0:
        # Save the image to a folder
        image_filename = os.path.join(img_dir, f'image_{frame_counter}.png')  # Concatenate folder path with image filename
        cv2.imwrite(image_filename, obs)
        print(f'Saved image: {image_filename}')

cv2.destroyAllWindows()  # Close the window after loop finishes
#
# # #load model
# # model = A2C("CnnPolicy", env, verbose=1, device="cpu", tensorboard_log=logdir)
# #
# # time_steps = 10000
# # for i in range(1,30):
# #     model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="A2C_cnn")
# #     model.save(f"{models_dir}/{time_steps*i}")
#
#
