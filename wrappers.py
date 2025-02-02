import gym
import torch
import numpy as np
from torchvision import transforms as T
from gym.spaces import Box


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat the action for a number of frames and sum the rewards"""
        total_reward = 0.0
        for i in range(self._skip):
            # Note: Gym 0.26+ returns (obs, reward, done, truncated, info)
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update the observation space to reflect grayscale (single channel)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # Convert [H, W, C] to [C, H, W] and make it a torch tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Compose transforms: resize then normalize.
        transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            # Using Normalize here with mean=0 and std=255 as in the tutorial.
            # (Alternatively, you might prefer to simply scale the image by 1/255.)
            T.Normalize((0,), (255,))
        ])
        observation = transforms(observation).squeeze(0)
        return observation
