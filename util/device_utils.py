"""
Device utilities for M1 Mac compatibility
"""
import torch

def get_device(gpu_id=0):
    """
    Get the best available device for the current system
    
    Args:
        gpu_id: GPU ID (ignored on M1 Mac, kept for compatibility)
    
    Returns:
        torch.device: The device to use
    """
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA device: {gpu_id}")
        torch.cuda.set_device(device=gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    else:
        print("Using CPU device")
        return torch.device("cpu")

def to_device(data, device):
    """
    Move data to device, handling different data types
    
    Args:
        data: Data to move (tensor, list, dict, etc.)
        device: Target device
    
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    else:
        return data

def setup_device_and_threads(config):
    """
    Setup device and thread configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        torch.device: The device to use
    """
    device = get_device(config.get('gpu_id', 0))
    
    # Set number of threads (good for M1 CPU performance)
    if device.type == 'cpu' or device.type == 'mps':
        torch.set_num_threads(4)  # M1 has 4-8 performance cores
    else:
        torch.set_num_threads(1)  # Original CUDA setting
    
    return device