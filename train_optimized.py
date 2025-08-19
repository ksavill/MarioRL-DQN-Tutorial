import datetime
from pathlib import Path
import torch
import numpy as np

import gym
import warnings
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, OptimizedFrameBuffer
from metrics import MetricLogger
from agent import Mario
from config_utils import load_config

# Silence benign DeprecationWarnings emitted by Gym with NumPy>=1.24
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.*")

class OptimizedTrainer:
    """
    Optimized trainer that maximizes GPU utilization and minimizes CPU bottlenecks.
    """
    def __init__(self, episodes=40000, batch_size=32, update_frequency=4):
        cfg = load_config()
        opt_cfg = cfg.get("train_optimized", {})
        # Resolve constructor args with config defaults
        self.episodes = int(opt_cfg.get("episodes", episodes))
        self.batch_size = int(opt_cfg.get("batch_size", batch_size))
        self.update_frequency = int(opt_cfg.get("update_frequency", update_frequency))
        self.log_every = int(opt_cfg.get("log_every_episodes", 20))
        self._agent_overrides = opt_cfg.get("agent_overrides", {})
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.setup_environment()
        self.setup_agent()
        self.setup_logging()
        
        # Batching variables for improved GPU utilization
        self.experience_batch = []
        self.step_count = 0
        
    def setup_environment(self):
        """Setup the Mario environment with optimized wrappers."""
        # Initialize Super Mario environment with API compatibility for Gym 0.26+
        if gym.__version__ < "0.26":
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", new_step_api=True)
        else:
            env = gym_super_mario_bros.make(
                "SuperMarioBros-1-1-v3",
                apply_api_compatibility=True,
            )

        # Limit the action space to:
        #   0: walk right
        #   1: jump right
        env = JoypadSpace(env, [["right"], ["right", "A"]])
        
        # Apply optimized wrappers
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        
        # Use standard FrameStack for compatibility
        if gym.__version__ < "0.26":
            env = FrameStack(env, num_stack=4, new_step_api=True)
        else:
            env = FrameStack(env, num_stack=4)
            
        env.reset()  # initialize the environment
        self.env = env
        
    def setup_agent(self):
        """Setup the Mario agent with optimized parameters."""
        # Create a unique directory for saving checkpoints and logs
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = None  # Set this to a checkpoint path if you wish to load one
        self.mario = Mario(
            state_dim=(4, 84, 84), 
            action_dim=self.env.action_space.n, 
            save_dir=save_dir, 
            checkpoint=checkpoint
        )
        
        # Optimize learning parameters for better GPU utilization (configurable)
        self.mario.batch_size = int(self._agent_overrides.get("batch_size", 256))
        self.mario.updates_per_step = int(self._agent_overrides.get("updates_per_step", 1))
        self.mario.learn_every = int(self._agent_overrides.get("learn_every", 4))
        
    def setup_logging(self):
        """Setup logging with GPU memory tracking."""
        self.logger = MetricLogger(self.mario.save_dir)
        
    def collect_experience_batch(self, state, action, reward, next_state, done_or_trunc):
        """Collect experiences in batches for more efficient processing."""
        self.experience_batch.append({
            'state': state,
            'action': action, 
            'reward': reward,
            'next_state': next_state,
            'done': done_or_trunc
        })
        
        # Process batch when it reaches target size
        if len(self.experience_batch) >= self.batch_size:
            self.process_experience_batch()
            
    def process_experience_batch(self):
        """Process a batch of experiences efficiently."""
        if not self.experience_batch:
            return
            
        # Cache all experiences in the batch
        for exp in self.experience_batch:
            self.mario.cache(exp['state'], exp['next_state'], exp['action'], exp['reward'], exp['done'])
            
        # Clear the batch
        self.experience_batch.clear()
        
        # Perform learning with GPU-optimized batching
        if self.step_count % self.update_frequency == 0:
            # Perform multiple learning steps for better GPU utilization
            total_q, total_loss = 0.0, 0.0
            learning_steps = min(4, len(self.mario.memory) // self.mario.batch_size)
            
            for _ in range(learning_steps):
                q, loss = self.mario.learn()
                if q is not None and loss is not None:
                    total_q += q
                    total_loss += loss
                    
            # Average the metrics
            if learning_steps > 0:
                avg_q = total_q / learning_steps
                avg_loss = total_loss / learning_steps
                return avg_q, avg_loss
                
        return None, None
        
    def train(self):
        """Main training loop with optimized GPU utilization."""
        print(f"Starting optimized training for {self.episodes} total episodes")
        print(f"Batch size: {self.batch_size}, Update frequency: {self.update_frequency}")
        print(f"Mario batch size: {self.mario.batch_size}")
        
        with open(self.logger.save_log, "a") as f:
            f.write(
                f"# Starting optimized training for {self.episodes} episodes at {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n"
            )

        for e in range(self.episodes):
            state = self.env.reset()  # In Gym 0.26, reset() returns (obs, info); see agent.act below
            episode_reward = 0
            episode_steps = 0
            
            while True:
                # Get action from Mario (the agent)
                action = self.mario.act(state)
                
                # Step through the environment
                next_state, reward, done, trunc, info = self.env.step(action)
                done_or_trunc = done or trunc
                
                # Collect experience in batch
                self.collect_experience_batch(state, action, reward, next_state, done_or_trunc)
                
                episode_reward += reward
                episode_steps += 1
                self.step_count += 1
                
                # Learn from experience batch
                q, loss = self.process_experience_batch()
                
                # Log metrics (only when we have valid loss/q values)
                if loss is not None:
                    self.logger.log_step(reward, loss, q)
                else:
                    self.logger.log_step(reward, None, None)
                
                state = next_state
                
                # End episode if game over or Mario reached the flag
                if done_or_trunc or info.get("flag_get", False):
                    break

            # Process any remaining experiences in the batch
            if self.experience_batch:
                self.process_experience_batch()
                
            self.logger.log_episode()

            # Record progress every N episodes (or on the last episode)
            if (e % self.log_every == 0) or (e == self.episodes - 1):
                self.logger.record(episode=e, epsilon=self.mario.exploration_rate, step=self.mario.curr_step)
                
                # Print GPU utilization info
                if torch.cuda.is_available():
                    print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")
                    torch.cuda.reset_peak_memory_stats()

        # Always write a final checkpoint at the end of training
        self.mario.save()
        print("Training completed!")


if __name__ == "__main__":
    # Create and run the optimized trainer
    trainer = OptimizedTrainer(episodes=40000, batch_size=32, update_frequency=4)
    trainer.train()
