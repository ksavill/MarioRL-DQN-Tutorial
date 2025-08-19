import time
import torch
import psutil
import threading
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class PerformanceMonitor:
    """
    Monitor CPU, GPU, and memory usage during training.
    """
    def __init__(self, log_interval=1.0):
        self.log_interval = log_interval
        self.monitoring = False
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.start_time = None
        
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        return {
            'cpu_usage': self.cpu_usage,
            'gpu_usage': self.gpu_usage, 
            'gpu_memory': self.gpu_memory,
            'timestamps': self.timestamps
        }
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            current_time = time.time() - self.start_time
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu_percent)
            
            # GPU usage and memory (if available)
            if torch.cuda.is_available():
                try:
                    # GPU memory usage
                    gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                    self.gpu_memory.append(gpu_memory_used)
                    
                    # GPU utilization (approximation based on memory allocation changes)
                    gpu_util = min(100, (gpu_memory_used / (torch.cuda.get_device_properties(0).total_memory / (1024**3))) * 100)
                    self.gpu_usage.append(gpu_util)
                except:
                    self.gpu_usage.append(0)
                    self.gpu_memory.append(0)
            else:
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
                
            self.timestamps.append(current_time)
            time.sleep(self.log_interval)
            
    def save_plots(self, save_dir, prefix="performance"):
        """Save performance plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU Usage
        ax1.plot(self.timestamps, self.cpu_usage, 'b-', linewidth=2)
        ax1.set_title('CPU Usage Over Time')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # GPU Usage
        ax2.plot(self.timestamps, self.gpu_usage, 'r-', linewidth=2)
        ax2.set_title('GPU Usage Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('GPU Usage (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # GPU Memory
        ax3.plot(self.timestamps, self.gpu_memory, 'g-', linewidth=2)
        ax3.set_title('GPU Memory Usage Over Time')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('GPU Memory (GB)')
        ax3.grid(True, alpha=0.3)
        
        # Combined CPU/GPU
        ax4.plot(self.timestamps, self.cpu_usage, 'b-', label='CPU Usage (%)', linewidth=2)
        ax4.plot(self.timestamps, self.gpu_usage, 'r-', label='GPU Usage (%)', linewidth=2)
        ax4.set_title('CPU vs GPU Usage')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Usage (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_monitor.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\n=== Performance Summary ===")
        print(f"Average CPU Usage: {np.mean(self.cpu_usage):.1f}%")
        print(f"Max CPU Usage: {np.max(self.cpu_usage):.1f}%")
        print(f"Average GPU Usage: {np.mean(self.gpu_usage):.1f}%")
        print(f"Max GPU Usage: {np.max(self.gpu_usage):.1f}%")
        print(f"Average GPU Memory: {np.mean(self.gpu_memory):.2f} GB")
        print(f"Max GPU Memory: {np.max(self.gpu_memory):.2f} GB")
        print(f"Total monitoring time: {self.timestamps[-1]:.1f} seconds")


def benchmark_training(episodes=100, use_optimized=True):
    """
    Benchmark training performance with monitoring.
    """
    print(f"Benchmarking {'optimized' if use_optimized else 'original'} training...")
    
    # Setup monitoring
    monitor = PerformanceMonitor(log_interval=0.5)
    monitor.start_monitoring()
    
    start_time = time.time()
    
    try:
        if use_optimized:
            # Import and run optimized trainer
            from train_optimized import OptimizedTrainer
            trainer = OptimizedTrainer(episodes=episodes, batch_size=32, update_frequency=4)
            trainer.train()
        else:
            # Run original training (simplified version for benchmarking)
            import datetime
            from pathlib import Path
            import gym
            import warnings
            import gym_super_mario_bros
            from gym.wrappers import FrameStack
            from nes_py.wrappers import JoypadSpace
            from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
            from metrics import MetricLogger
            from agent import Mario
            
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.*")
            
            # Setup environment (original way)
            if gym.__version__ < "0.26":
                env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", new_step_api=True)
            else:
                env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", apply_api_compatibility=True)
            
            env = JoypadSpace(env, [["right"], ["right", "A"]])
            env = SkipFrame(env, skip=4)
            env = GrayScaleObservation(env)
            env = ResizeObservation(env, shape=84)
            if gym.__version__ < "0.26":
                env = FrameStack(env, num_stack=4, new_step_api=True)
            else:
                env = FrameStack(env, num_stack=4)
            env.reset()
            
            save_dir = Path("checkpoints") / f"benchmark_original_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=None)
            logger = MetricLogger(save_dir)
            
            # Original training loop
            for e in range(episodes):
                state = env.reset()
                while True:
                    action = mario.act(state)
                    next_state, reward, done, trunc, info = env.step(action)
                    done_or_trunc = done or trunc
                    mario.cache(state, next_state, action, reward, done_or_trunc)
                    q, loss = mario.learn()
                    logger.log_step(reward, loss, q)
                    state = next_state
                    if done_or_trunc or info.get("flag_get", False):
                        break
                logger.log_episode()
                if e % 20 == 0:
                    logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
            mario.save()
            
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        end_time = time.time()
        training_time = end_time - start_time
        
        # Stop monitoring and save results
        performance_data = monitor.stop_monitoring()
        
        # Save performance plots
        save_dir = Path("benchmarks") / f"{'optimized' if use_optimized else 'original'}_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
        monitor.save_plots(save_dir, prefix=f"{'optimized' if use_optimized else 'original'}")
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Performance data saved to: {save_dir}")
        
        return {
            'training_time': training_time,
            'performance_data': performance_data,
            'save_dir': save_dir
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark RL Mario training performance')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--mode', choices=['original', 'optimized', 'both'], default='optimized', 
                        help='Which training mode to benchmark')
    
    args = parser.parse_args()
    
    if args.mode in ['optimized', 'both']:
        print("=== Benchmarking Optimized Training ===")
        optimized_results = benchmark_training(episodes=args.episodes, use_optimized=True)
        
    if args.mode in ['original', 'both']:
        print("\n=== Benchmarking Original Training ===")
        original_results = benchmark_training(episodes=args.episodes, use_optimized=False)
        
    if args.mode == 'both':
        print("\n=== Comparison ===")
        speedup = original_results['training_time'] / optimized_results['training_time']
        print(f"Speedup: {speedup:.2f}x faster with optimized training")
