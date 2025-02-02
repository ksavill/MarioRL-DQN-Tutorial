import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from metrics import MetricLogger
from agent import Mario

# Initialize Super Mario environment with API compatibility for Gym 0.26+
if gym.__version__ < "0.26":
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", new_step_api=True)
else:
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v3",
        render_mode="rgb",
        apply_api_compatibility=True,
    )

# Limit the action space to:
#   0: walk right
#   1: jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])
env.reset()  # initialize the environment

# Apply wrappers to preprocess the environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < "0.26":
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

# Create a unique directory for saving checkpoints and logs
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True, exist_ok=True)

checkpoint = None  # Set this to a checkpoint path if you wish to load one
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
logger = MetricLogger(save_dir)

episodes = 40000  # For demonstration. Increase (e.g., to 40000) for real training.
"""
An episode is a complete run through the environment from a reset until a terminal condition is met.
- for example, when Mario dies or reaches the flag).

Note: in agent.py, there is a save_every parameter that determines how often to save the model.
It is currently set to 5e5 (500,000 steps). If the number of episodes is too low, the model may not be saved due to not reaching enough steps.
"""

for e in range(episodes):
    state = env.reset()  # In Gym 0.26, reset() returns (obs, info); see agent.act below
    while True:
        # Get action from Mario (the agent)
        action = mario.act(state)
        # Step through the environment; env.step returns (next_state, reward, done, truncated, info)
        next_state, reward, done, trunc, info = env.step(action)
        # Store experience
        mario.cache(state, next_state, action, reward, done)
        # Learn from experience
        q, loss = mario.learn()
        # Log metrics
        logger.log_step(reward, loss, q)
        state = next_state
        # End episode if game over or Mario reached the flag
        if done or info.get("flag_get", False):
            break

    logger.log_episode()

    # Record progress every 20 episodes (or on the last episode)
    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
