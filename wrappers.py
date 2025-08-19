import gym
import torch
import numpy as np
from torchvision import transforms as T
from gym.spaces import Box
from collections import deque


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
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update the observation space to reflect grayscale (single channel)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
        # Pre-compile transform for efficiency
        self.transform = T.Grayscale()

    def permute_orientation(self, observation):
        # Convert [H, W, C] to [C, H, W] and make it a torch tensor
        # Ensure positive strides/contiguous memory to avoid torch.from_numpy errors
        observation = np.transpose(observation, (2, 0, 1)).copy()
        # Use from_numpy for better performance and avoid copy when possible
        observation = torch.from_numpy(observation).float()
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        observation = self.transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        # After resizing we normalize to 0..1 floats; update the space accordingly
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        
        # Pre-compile transforms for efficiency
        self.transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            # Using Normalize here with mean=0 and std=255 as in the tutorial.
            # (Alternatively, you might prefer to simply scale the image by 1/255.)
            T.Normalize((0,), (255,))
        ])

    def observation(self, observation):
        observation = self.transforms(observation).squeeze(0)
        # Return a NumPy array for compatibility with Gym's FrameStack
        # Use detach() to avoid keeping computation graph and improve memory efficiency
        return observation.detach().numpy()


class OptimizedFrameBuffer(gym.Wrapper):
    """
    Optimized frame buffer that reduces CPU-GPU transfers by batching operations.
    """
    def __init__(self, env, buffer_size=4):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Clear buffer and fill with initial observation
        self.frame_buffer.clear()
        for _ in range(self.buffer_size):
            self.frame_buffer.append(obs)
        return obs
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frame_buffer.append(obs)
        return obs, reward, done, truncated, info
    
    def get_stacked_frames(self):
        """Return stacked frames as a tensor on GPU for efficient processing."""
        if len(self.frame_buffer) < self.buffer_size:
            return None
        
        # Stack frames efficiently
        frames = list(self.frame_buffer)
        stacked = np.stack(frames, axis=0)
        return torch.from_numpy(stacked).to(self.device, non_blocking=True)
