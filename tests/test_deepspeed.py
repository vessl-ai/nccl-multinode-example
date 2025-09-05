# deepspeed_test.py
import os
import deepspeed
import torch
import torch.distributed as dist
from deepspeed.runtime.config import DeepSpeedConfig

def main():
    # DeepSpeed initialization
    deepspeed.init_distributed()

    # Get rank info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    # Report configuration
    if rank == 0:
        print(f"DeepSpeed Test: World Size = {world_size}")

    # Create a simple model for testing
    model = torch.nn.Linear(1024, 1024).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_batch_size": 16 * world_size,

            # FP16 (half precision) related settings
            "fp16": {
                "enabled": True,
                "auto_cast": True,         # Enable automatic FP32 -> FP16 conversion
                "loss_scale": 0,           # Use dynamic loss scaling

                "initial_scale_power": 16, # Set initial loss scale to 2^16
                "loss_scale_window": 1000, # Number of steps to wait before adjusting loss scale
                "hysteresis": 2,           # Number of overflows to tolerate before adjusting loss scale
                "min_loss_scale": 1        # Minimum loss scale value
            },

            # ZeRO (Zero Redundancy Optimizer) Config
            "zero_optimization": {
                "stage": 2,                    # ZeRO stage 2: optimizer states and gradients are distributed
                "contiguous_gradients": True   # Store gradients in contiguous memory for better communication efficiency
            },
            "gradient_clipping": 1.0,          # gradient clipping value (prevent gradient explosion)
            "communication_data_type": "fp16"  # Data type for communication
        }
    )

    # Test bandwidth with parameter all-reduce
    if rank == 0:
        print("\n=== DeepSpeed Bandwidth Test ===")

    # Run a few training steps to test communication
    for i in range(10):
        # Generate random input
        input_data = torch.randn(16, 1024, dtype=torch.float16).cuda()
        labels = torch.randn(16, 1024, dtype=torch.float16).cuda()

        # Forward pass
        outputs = model_engine(input_data)

        # Calculate loss
        loss = torch.nn.functional.mse_loss(outputs, labels)

        # Backward pass
        model_engine.backward(loss)

        # Weight update
        model_engine.step()

        # Report progress
        if rank == 0 and i % 2 == 0:
            print(f"Step {i}, Loss: {loss.item()}")

    # All finished
    if rank == 0:
        print("DeepSpeed test completed successfully!")

if __name__ == "__main__":
    main()
