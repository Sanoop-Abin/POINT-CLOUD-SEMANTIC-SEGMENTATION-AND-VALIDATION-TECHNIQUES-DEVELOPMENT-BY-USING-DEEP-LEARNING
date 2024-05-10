import torch
import torchvision
if torch.cuda.is_available():
    # Get the GPU device count
    gpu_count = torch.cuda.device_count()
    
    if gpu_count > 0:
        print(f"Found {gpu_count} GPU(s) available.")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPU available.")
else:
    print("CUDA is not available. Using CPU.")

# Your code for dataset visualization here

# After running your code, check the GPU usage again
gpu_memory_used = torch.cuda.memory_allocated()
gpu_memory_cached = torch.cuda.memory_cached()
print(f"GPU Memory Used: {gpu_memory_used / 1e6} MB")
print(f"GPU Memory Cached: {gpu_memory_cached / 1e6} MB")