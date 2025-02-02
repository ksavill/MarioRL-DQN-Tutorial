import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

# 1. Initialize the environment with proper API compatibility for Gym 0.26+
if gym.__version__ < "0.26":
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3', new_step_api=True)
else:
    env = gym_super_mario_bros.make(
        'SuperMarioBros-1-1-v0',
        render_mode='rgb',
        apply_api_compatibility=True,
    )

# 2. Limit the action space to:
#      0: walk right
#      1: jump right
env = JoypadSpace(
    env,
    [['right'],
     ['right', 'A']]
)

# 3. Apply wrappers to preprocess the observations
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)           # No "keep_dim" parameter here
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# Reset to get the initial observation (using new Gym API: (obs, info))
state, info = env.reset()

# 4. Create a directory for saving checkpoints & logs
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True, exist_ok=True)

# 5. Load an existing checkpoint if available
checkpoint = Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint
)

# Set exploration rate to minimum for greedy (replay) actions
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):
    # Reset the environment (Gym 0.26 API: returns (state, info))
    state, info = env.reset()

    while True:
        # Render the game window (optional; may slow replay)
        env.render()

        # Agent chooses an action
        action = mario.act(state)

        # Step the environment; Gym 0.26 returns (next_state, reward, done, truncated, info)
        next_state, reward, done, truncated, info = env.step(action)

        # Cache the experience (for logging/replay purposes; no training is done here)
        mario.cache(state, next_state, action, reward, done or truncated)

        # Log the reward (loss and Q-value are not computed during replay)
        logger.log_step(reward, None, None)

        # Update the state
        state = next_state

        # End episode if done, truncated, or if Mario got the flag
        if done or truncated or info.get('flag_get', False):
            break

    # Log episode-level stats
    logger.log_episode()

    # Record stats every 20 episodes
    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
