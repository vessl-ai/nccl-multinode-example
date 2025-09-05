import os
import time
import torch
import torch.distributed as dist

def print_environment():
    """Print all relevant environment variables for debugging."""
    print("\n=== Environment Information ===")
    env_vars = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT",
        "NCCL_DEBUG", "NCCL_IB_HCA", "UCX_NET_DEVICES",
        "NCCL_SOCKET_IFNAME"
    ]
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Current Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name()}")
    print("==============================\n")

def test_nccl_bandwidth():
    """Test NCCL bandwidth using multiple tensor sizes."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            print(f"Process with LOCAL_RANK {local_rank} assigned to GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        else:
            print(f"Warning: LOCAL_RANK {local_rank} exceeds available GPUs {torch.cuda.device_count()}")

    print_environment()

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\n=== NCCL Bandwidth Test ===")
        print(f"World Size: {world_size}")
        print(f"Running on device: {torch.cuda.get_device_name()}")

    # Test different tensor sizes
    sizes_mb = [1, 8, 64, 256, 1024]  # Size in MB
    iterations = 10
    warmup = 5

    results = []

    for size_mb in sizes_mb:
        # Create tensor of specified size
        size_bytes = size_mb * 1024 * 1024
        tensor = torch.ones(size_bytes // 4, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(warmup):
            dist.all_reduce(tensor)

        # Synchronize before timing
        torch.cuda.synchronize()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iterations):
            dist.all_reduce(tensor)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / iterations

        # Calculate bandwidth: tensor_size * 2 (for all-reduce) / time
        # Factor of 2 because all-reduce both sends and receives data
        bandwidth_gb_s = size_bytes * 2 / (elapsed_time / 1000) / (10**9)

        if rank == 0:
            print(f"Tensor Size: {size_mb} MB, Bandwidth: {bandwidth_gb_s:.2f} GB/s, Time: {elapsed_time:.2f} ms")
            results.append((size_mb, bandwidth_gb_s, elapsed_time))

    # Print summary if rank 0
    if rank == 0:
        print("\n=== Bandwidth Summary ===")
        print("Size (MB) | Bandwidth (GB/s) | Time (ms)")
        print("---------|-----------------|----------")
        for size_mb, bandwidth, time in results:
            print(f"{size_mb:9} | {bandwidth:15.2f} | {time:9.2f}")
        print("\n")

    # Check if bandwidth seems reasonable for InfiniBand
    if rank == 0 and results:
        max_bandwidth = max([b for _, b, _ in results])
        if max_bandwidth > 50:  # Good InfiniBand should exceed 50 GB/s
            print("✅ EXCELLENT: Peak bandwidth exceeds 50 GB/s, indicating good InfiniBand performance")
        elif max_bandwidth > 20:  # Decent bandwidth
            print("✓ GOOD: Bandwidth indicates InfiniBand is working, but may not be optimal")
        else:  # Questionable performance
            print("⚠️ WARNING: Bandwidth is lower than expected for InfiniBand. Check configuration.")

    # Test point-to-point communication if world size > 1
    if world_size > 1:
        if rank == 0:
            print("\n=== Testing Point-to-Point Communication ===")

        # Select a moderate tensor size
        size_mb = 64
        size_bytes = size_mb * 1024 * 1024
        tensor = torch.ones(size_bytes // 4, dtype=torch.float32, device='cuda') * (rank + 1)

        # Every rank sends to every other rank
        for src in range(world_size):
            for dst in range(world_size):
                if src == dst:
                    continue

                if rank == 0:
                    print(f"Testing P2P: {src} → {dst}", flush=True)

                if rank == src:
                    dist.send(tensor, dst)
                elif rank == dst:
                    received = torch.zeros_like(tensor)
                    dist.recv(received, src)
                    # Verify data
                    expected = torch.ones_like(tensor) * (src + 1)
                    if torch.allclose(received, expected):
                        if rank == 0 or rank == dst:
                            print(f"✅ Verified data from rank {src} to {dst}")
                    else:
                        print(f"❌ Data verification failed from rank {src} to {dst}")

        if rank == 0:
            print("=== P2P Communication Test Complete ===\n")

    # Barrier to ensure all processes are done
    dist.barrier()

def main():
    """Main function to run the InfiniBand test."""
    # Wait a bit for other processes to start
    time.sleep(5)

    # Run the bandwidth test
    test_nccl_bandwidth()

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"Rank {os.environ.get('RANK', '?')}: Test completed successfully")

if __name__ == "__main__":
    main()
