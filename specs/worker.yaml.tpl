name: nccl-worker-<<NODE_RANK>>
import:
  /dataset/: volume://vessl-storage/120k-alpaca
  /repo:
    git:
      url: github.com/vessl-ai/test-nccl.git
      ref: main
      credential_name: vessl-ai
export:
  /output/: volume://vessl-storage
resources:
  cluster: vessl-eu-h100-80g-sxm
  preset: gpu-h100-80g-xlarge
image: quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5
run: /repo/finetune/run.sh
env:
  MASTER_ADDR: <<MASTER_ADDR>>
  NODE_RANK: <<NODE_RANK>>
  NUM_NODES: <<NUM_NODES>>
  GPUS_PER_NODE: <<GPUS_PER_NODE>>
  RDZV_ID: <<RDZV_ID>>
  ROLE: worker
