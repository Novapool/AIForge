import torch
import sys

def test_gpu_availability():
    """Test GPU availability and configuration"""
    
    print("\n=== PyTorch GPU Test ===\n")
    
    # Basic PyTorch version and CUDA information
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Get number of CUDA devices
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # Get current device properties
        current_device = torch.cuda.current_device()
        print(f"\nCurrent CUDA device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Device name: {device_name}")
        
        # Get device properties
        device_properties = torch.cuda.get_device_properties(current_device)
        print("\nDevice Properties:")
        print(f"  Total memory: {device_properties.total_memory / 1024**3:.2f} GB")
        print(f"  Multi Processor Count: {device_properties.multi_processor_count}")
        print(f"  Compute Capability: {device_properties.major}.{device_properties.minor}")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        try:
            # Create tensors on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Perform matrix multiplication
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            z = torch.matmul(x, y)
            end_time.record()
            
            # Synchronize CUDA events
            torch.cuda.synchronize()
            
            # Calculate elapsed time
            elapsed_time = start_time.elapsed_time(end_time)
            
            print(f"Matrix multiplication test successful!")
            print(f"Elapsed time: {elapsed_time:.2f} ms")
            
            # Test memory allocation
            print("\nTesting GPU memory allocation...")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"\nError during GPU computation test: {str(e)}")
    else:
        print("\nNo CUDA device available. Possible reasons:")
        print("1. NVIDIA GPU is not present in the system")
        print("2. NVIDIA drivers are not installed or are outdated")
        print("3. CUDA toolkit is not installed")
        print("4. PyTorch was installed without CUDA support")
        print("\nTroubleshooting steps:")
        print("1. Verify NVIDIA GPU is present: Check Device Manager on Windows")
        print("2. Update NVIDIA drivers: Visit https://www.nvidia.com/Download/index.aspx")
        print("3. Install CUDA toolkit: Visit https://developer.nvidia.com/cuda-downloads")
        print("4. Reinstall PyTorch with CUDA support:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    test_gpu_availability()
