nccl-multinode-example
=======================

VESSL Run을 활용하여 Infiniband가 연결된 클러스터에서 multi-node 학습을 진행하는 예제입니다.

## Usage

```
vessl run create -f specs/launcher.yaml
```

## Detailed Information

예제는 다음과 같이 작동합니다.

1. 먼저 launcher 역할 (다른 worker들 관리 + 자기 자신도 rank=0의 worker로 학습) 을 할 Run을 실행합니다. ([specs/launcher.yaml](specs/launcher.yaml))
1. 해당 Run은 Infiniband communication이 가능한 자기 주소를 확인 후, 해당 주소를 MASTER_ADDR로 삼아 n-1개의 worker들을 Run으로 실행합니다. ([finetune/run.sh#L37](finetune/run.sh#L37)
    1. 실행할 worker의 spec은 [specs/worker.yaml.tpl](specs/worker.yaml.tpl) 처럼 정의하시면 됩니다.
1. 이후 해당 worker들이 모두 연결될때까지 기다립니다. ([scripts/gather_hosts.sh](scripts/gather_hosts.sh))
1. Worker들의 연결 과정은 아래와 같은 순서로 작동합니다.
    1. worker들은 자기 자신의 endpoint(hostname과 IP)를 launcher job에 보고합니다. ([scripts/gather_hosts.sh#L208](scripts/gather_hosts.sh#L208))
    1. Launcher는 모든 worker들의 hostname/IP를 수집 후 worker에 endpoint를 전파합니다. ([scripts/gather_hosts.sh#L179](scripts/gather_hosts.sh#L179))
    1. worker는 해당 endpoint를 /etc/hosts에 등록해서 worker 간 통신이 가능하게 합니다. ([scripts/gather_hosts.sh#235](scripts/gather_hosts.sh#L235))
1. worker pool 구성 완료 후 모든 launcher와 worker들이 distributed job을 수행합니다. ([finetune/finetune.py](finetune/finetune.py))
    1. 예제에서는 torchrun으로 job을 실행하였으며 실행하는 파이썬 스크립트에서는 deepspeed 라이브러리를 사용하였습니다.
1. 학습 도중 로그는 transformers 라이브러리의 TrainerCallback을 사용합니다. 간단한 구현을 준비해두었습니다. ([finetune/finetune.py#L24-49](finetune/finetune.py#L24-49))
1. 학습 작업이 끝나면 정상적으로 모든 worker와 launcher가 종료됩니다.

예제를 기존에 사용하시던 학습 코드와 맞게 커스텀하실 경우 아래 사항을 참고하여 커스텀하셔서 사용하시면 됩니다.
- 컨테이너 이미지에 `libmlx5-1`, `libibverbs1`, `ibverbs-utils`이 설치되어있어야 합니다. (e.g. 우분투 기반 이미지의 경우 `apt-get install -y libmlx5-1 libibverbs1 ibverbs-utils`로 설치)
- 모든 launcher와 worker는 네개의 파일을 공유합니다. (코드 마운트 등으로 같은 코드를 공유하시면 됩니다)
    - Deepspeed configuration file ([finetune/ds_config_zero2.json](finetune/ds_config_zero2.json))
    - 학습에 사용할 python script ([finetune/finetune.py](finetune/finetune.py))
    - Worker launch 및 학습 스크립트 실행에 사용하는 bash script ([finetune/run.sh](finetune/run.sh))
    - Peer discovery에 사용하는 bash script ([scripts/gather_hosts.sh](scripts/gather_hosts.sh))
- Launcher 역할을 할 잡에 아래 환경 변수가 정의되어 있어야 합니다.
    - `NCCL_IB_HCA="mlx5"`
    - `UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"`
- launcher를 실행할 때 `NUM_NODES`, `GPUS_PER_NODE` 등의 distribution 설정을 환경 변수로 잡아줘야 하며, 디스커버리에 사용하기 위한 임의의 키를 `RDZV_ID`라는 환경 변수로 launcher를 실행할 때 주입해주셔야 합니다. ([specs/launcher.yaml](specs/launcher.yaml))
