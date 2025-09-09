from argparse import ArgumentParser 
import yaml
from pathlib import Path

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["OMP_NUM_THREADS"] = "12"


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(device, flush=True)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    dist.destroy_process_group()
    
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
    
    
    
# OMP_NUM_THREADS=12 uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 train.py --config config.yaml