#!/bin/bash

apt-get update && apt-get install -y libmlx5-1 libibverbs1 ibverbs-utils netcat
pip install -U deepspeed peft transformers vessl datasets

export MASTER_PORT=29500
export NCCL_IB_HCA="mlx5"
export PYTORCH_TCP_SOCKET_IFNAME="eth0"
export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL  # More detailed debugging
export NCCL_IB_DISABLE=0                 # Make sure IB is not disabled
export NCCL_IB_CUDA_SUPPORT=1            # Enable GPU Direct RDMA if available
export NCCL_IB_GID_INDEX=3               # Try different GID indices (0,1,2,3)
export NCCL_IB_TIMEOUT=22                # Extend timeout for IB operations
export NCCL_IB_RETRY_CNT=7               # Increase retry count
export NCCL_IB_SL=0                      # Service level
export NCCL_NET_GDR_LEVEL=5              # Enable GPUDirect RDMA

# Set MASTER_ADDR based on ROLE
if [ "$ROLE" = "launcher" ]; then
  export MASTER_ADDR=$(hostname -i)
  # Create a temporary yaml file for each worker
  for NODE_RANK in $(seq 1 $((NUM_NODES-1))); do
    echo "Creating worker node with NODE_RANK=$NODE_RANK"

    # Create a temporary file with placeholders replaced
    TMP_YAML=$(mktemp)

    # Replace placeholders in the YAML file
    sed "s|<<MASTER_ADDR>>|$MASTER_ADDR|g; s|<<NODE_RANK>>|$NODE_RANK|g; s|<<RDZV_ID>>|$RDZV_ID|g; s|<<NUM_NODES>>|$NUM_NODES|g; s|<<GPUS_PER_NODE>>|$GPUS_PER_NODE|g" /repo/specs/worker.yaml.tpl > "$TMP_YAML"

    # Run vessl run create with the temporary file
    echo "Running: vessl run create -f $TMP_YAML"
    vessl run create -f "$TMP_YAML"

    # Clean up temporary file
    rm "$TMP_YAML"

    echo "Worker node $NODE_RANK created"
  done

  export NODE_RANK=0
else
  # Keep existing MASTER_ADDR value from environment
  # If not set, this will remain empty
  :
fi

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# Update /etc/hosts
ROLE=$ROLE MASTER_HOST=$MASTER_ADDR NUM_NODES=$NUM_NODES /repo/scripts/gather_hosts.sh

torchrun --nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT /repo/finetune/finetune.py \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --data_path "/dataset/train-00000-of-00001-d9b93805488c263e.parquet" \
  --bf16 True \
  --output_dir "/output" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --deepspeed "/repo/finetune/ds_config_zero2.json"
