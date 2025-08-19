import numpy as np
from agent import MarioNet

def calculate_model_memory():
    """Calculate memory requirements for the MarioNet model."""
    
    # Model parameters
    state_dim = (4, 84, 84)  # 4 stacked frames, 84x84 pixels
    action_dim = 2  # right, jump-right
    
    # Create model to calculate parameters
    model = MarioNet(state_dim, action_dim)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory per parameter (float32 = 4 bytes)
    bytes_per_param = 4
    
    # Model memory (both online and target networks)
    model_memory_mb = (total_params * bytes_per_param * 2) / (1024**2)  # 2 networks
    
    # Gradients memory (only for online network)
    gradient_memory_mb = (trainable_params * bytes_per_param) / (1024**2)
    
    # Optimizer state (Adam: 2 states per parameter)
    optimizer_memory_mb = (trainable_params * bytes_per_param * 2) / (1024**2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_memory_mb': model_memory_mb,
        'gradient_memory_mb': gradient_memory_mb,
        'optimizer_memory_mb': optimizer_memory_mb
    }

def calculate_data_memory(batch_size=256, replay_buffer_size=10000):
    """Calculate memory requirements for training data."""
    
    # State dimensions
    state_shape = (4, 84, 84)  # 4 channels, 84x84 pixels
    bytes_per_float = 4
    
    # Single experience memory
    state_memory = int(np.prod(state_shape)) * bytes_per_float  # state
    next_state_memory = int(np.prod(state_shape)) * bytes_per_float  # next_state
    action_memory = 8  # int64
    reward_memory = 4  # float32
    done_memory = 1   # bool
    
    experience_memory_bytes = int(state_memory + next_state_memory + action_memory + reward_memory + done_memory)
    
    # Replay buffer memory
    replay_buffer_memory_mb = (experience_memory_bytes * replay_buffer_size) / (1024**2)
    
    # Training batch memory
    batch_memory_mb = (experience_memory_bytes * batch_size) / (1024**2)
    
    # Forward pass activations (approximate)
    # Conv layers + FC layers activations
    conv1_output = batch_size * 32 * 20 * 20 * bytes_per_float  # After conv1
    conv2_output = batch_size * 64 * 9 * 9 * bytes_per_float   # After conv2  
    conv3_output = batch_size * 64 * 7 * 7 * bytes_per_float   # After conv3
    fc1_output = batch_size * 512 * bytes_per_float            # After fc1
    fc2_output = batch_size * 2 * bytes_per_float              # After fc2
    
    activation_memory_mb = (conv1_output + conv2_output + conv3_output + fc1_output + fc2_output) / (1024**2)
    
    return {
        'experience_memory_bytes': experience_memory_bytes,
        'replay_buffer_memory_mb': replay_buffer_memory_mb,
        'batch_memory_mb': batch_memory_mb,
        'activation_memory_mb': activation_memory_mb
    }

def calculate_mixed_precision_savings():
    """Calculate memory savings from mixed precision training."""
    # AMP typically saves ~40-50% memory for activations
    # Model weights stay in FP32, but forward pass uses FP16
    return 0.45  # 45% savings on activation memory

def print_memory_requirements():
    """Print comprehensive memory requirements."""
    
    print("=" * 60)
    print("RL MARIO TRAINING MEMORY REQUIREMENTS")
    print("=" * 60)
    
    # Model memory
    model_info = calculate_model_memory()
    print(f"\n🧠 MODEL MEMORY:")
    print(f"   Total Parameters: {model_info['total_params']:,}")
    print(f"   Trainable Parameters: {model_info['trainable_params']:,}")
    print(f"   Model Weights (Online + Target): {model_info['model_memory_mb']:.1f} MB")
    print(f"   Gradients: {model_info['gradient_memory_mb']:.1f} MB")
    print(f"   Optimizer State (Adam): {model_info['optimizer_memory_mb']:.1f} MB")
    
    model_total_mb = (model_info['model_memory_mb'] + 
                     model_info['gradient_memory_mb'] + 
                     model_info['optimizer_memory_mb'])
    print(f"   📊 Model Total: {model_total_mb:.1f} MB")
    
    # Data memory for different configurations
    configs = [
        {"name": "Original", "batch_size": 128, "buffer_size": 10000},
        {"name": "Optimized", "batch_size": 256, "buffer_size": 10000},
        {"name": "High Memory", "batch_size": 512, "buffer_size": 20000}
    ]
    
    print(f"\n💾 DATA MEMORY (by configuration):")
    
    for config in configs:
        data_info = calculate_data_memory(config["batch_size"], config["buffer_size"])
        amp_savings = calculate_mixed_precision_savings()
        
        print(f"\n   {config['name']} Configuration:")
        print(f"   - Batch Size: {config['batch_size']}")
        print(f"   - Replay Buffer Size: {config['buffer_size']:,}")
        print(f"   - Replay Buffer: {data_info['replay_buffer_memory_mb']:.1f} MB")
        print(f"   - Training Batch: {data_info['batch_memory_mb']:.1f} MB")
        print(f"   - Activations (FP32): {data_info['activation_memory_mb']:.1f} MB")
        print(f"   - Activations (Mixed Precision): {data_info['activation_memory_mb'] * (1-amp_savings):.1f} MB")
        
        # Total VRAM calculation
        data_total_fp32 = (data_info['replay_buffer_memory_mb'] + 
                          data_info['batch_memory_mb'] + 
                          data_info['activation_memory_mb'])
        
        data_total_amp = (data_info['replay_buffer_memory_mb'] + 
                         data_info['batch_memory_mb'] + 
                         data_info['activation_memory_mb'] * (1-amp_savings))
        
        vram_fp32 = model_total_mb + data_total_fp32
        vram_amp = model_total_mb + data_total_amp
        
        print(f"   📊 Data Total (FP32): {data_total_fp32:.1f} MB")
        print(f"   📊 Data Total (Mixed Precision): {data_total_amp:.1f} MB")
        print(f"   🎯 TOTAL VRAM (FP32): {vram_fp32:.1f} MB ({vram_fp32/1024:.2f} GB)")
        print(f"   🎯 TOTAL VRAM (Mixed Precision): {vram_amp:.1f} MB ({vram_amp/1024:.2f} GB)")
    
    # System memory requirements
    print(f"\n🖥️  SYSTEM MEMORY REQUIREMENTS:")
    print(f"   Environment Processing: ~500-1000 MB")
    print(f"   Python + Libraries: ~200-500 MB") 
    print(f"   Mario Game Environment: ~100-200 MB")
    print(f"   Logging & Metrics: ~50-100 MB")
    print(f"   OS + Other Processes: ~2-4 GB")
    print(f"   📊 Recommended System RAM: 8-16 GB")
    
    # GPU recommendations
    print(f"\n🎮 GPU RECOMMENDATIONS:")
    print(f"   Minimum VRAM: 2 GB (Original config, mixed precision)")
    print(f"   Recommended VRAM: 4-6 GB (Optimized config)")
    print(f"   Optimal VRAM: 8+ GB (High memory config)")
    print(f"   ")
    print(f"   Suitable GPUs:")
    print(f"   - RTX 3060 (12GB) - Excellent")
    print(f"   - RTX 3070 (8GB) - Very Good") 
    print(f"   - RTX 3080 (10GB) - Excellent")
    print(f"   - RTX 4060 (8GB) - Very Good")
    print(f"   - RTX 4070 (12GB) - Excellent")
    print(f"   - GTX 1660 Ti (6GB) - Good (original config only)")
    print(f"   - RTX 2060 (6GB) - Good")
    
    print(f"\n⚡ PERFORMANCE TIPS:")
    print(f"   - Use mixed precision training (enabled by default)")
    print(f"   - Start with batch_size=128 if VRAM limited")
    print(f"   - Increase batch_size=256+ for better GPU utilization")
    print(f"   - Monitor GPU memory with benchmark_performance.py")
    
    print("=" * 60)

if __name__ == "__main__":
    print_memory_requirements()
