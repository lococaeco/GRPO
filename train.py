from argparse import ArgumentParser 
import yaml
from pathlib import Path
from datetime import datetime
import time
import html

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


from countdown_task import CountdownTasksDataset, reward_function
from qwen2_model import Transformer
from tokenizer import Tokenizer
from optimizer import MemoryEfficientAdamW
from grpo import rollout, update_policy

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def evaluate_rank0_only(model, tokenizer, device, dtype, config):
    model_w = model.module if hasattr(model, "module") else model
    model_w.eval()
    try:
        test_dataset = CountdownTasksDataset(
            data_path=config["data"]["path"],
            tokenizer=tokenizer,
            split="test",
            test_size=config["data"]["test_size"],
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            collate_fn=CountdownTasksDataset.collate_fn,
            batch_size=max(1, config["training"]["batch_size"] // 2),
            pin_memory=True,
            num_workers=2,
        )
        success = []
        with torch.no_grad():
            for batch in test_loader:
                episodes = rollout(
                    model=model_w,
                    tokenizer=tokenizer,
                    batch=batch,
                    max_gen_len=config["training"]["max_gen_len"] * 2,
                    num_answer_per_question=1,
                    reward_function=reward_function,
                    device=device,
                    dtype=dtype,
                )
                success.extend([ep.reward_info["answer_reward"] for ep in episodes])
        return float(np.mean(success)) if success else 0.0
    finally:
        model_w.train()


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    
    train_dataset = CountdownTasksDataset(
    data_path=config["data"]["path"],
    tokenizer=tokenizer,
    split="train",
    test_size=config["data"]["test_size"],
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    # DataLoader에서 shuffle=False로 두고 sampler만 넘기기
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=CountdownTasksDataset.collate_fn,
        batch_size=NUM_QUESTIONS_PER_BATCH,
        pin_memory=True,
        num_workers=config["training"].get("num_workers", 4),
        persistent_workers=config["training"].get("num_workers", 4) > 0,
    )
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    
    model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,         # 필요 시 True (비활성 branch가 있으면)
    gradient_as_bucket_view=True,
    broadcast_buffers=False,              # 통상 Transformer엔 불필요 버퍼 브로드캐스트 방지
)

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )
    
    base_seed = int(config["training"]["random_seed"])
    seed = base_seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()

    for step, batch in enumerate(train_dataloader, start=1):
        # (에폭 개념이 있다면 밖에서 set_epoch 호출)
        # train_sampler.set_epoch(step)  # step 기반 셔플 갱신을 원할 때

        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [ep for ep in episodes if ep.is_finished]

        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )

        torch.cuda.synchronize(device)  # 로컬 디바이스 동기화
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # ------ 메트릭 집계 ------
        reward = np.array([ep.reward for ep in episodes], dtype=np.float32)
        formatted_reward = np.array([ep.reward_info["format_reward"] for ep in episodes], dtype=np.float32)
        answer_reward = np.array([ep.reward_info["answer_reward"] for ep in episodes], dtype=np.float32)
        num_finished_episodes = sum(ep.is_finished for ep in episodes)

        # (선택) 모든 rank에서 평균 내고 싶다면 dist.all_reduce 사용
        t = torch.tensor(
            [reward.mean(), reward.std(), formatted_reward.mean(), answer_reward.mean(), num_finished_episodes, duration],
            device=device, dtype=torch.float32,
        )
        dist.all_reduce(t, op=dist.ReduceOp.AVG)  # 평균
        mean_reward, std_reward, format_reward, success_rate, num_finished_episodes_avg, duration_avg = t.tolist()

        # 로깅/프린트는 rank0만
        if is_main_process():
            grad_norm = float(results["grad_norm"])
            entropy = float(results["entropy"])
            lr = float(optimizer.param_groups[0]["lr"])
            loss = float(results["loss"])
            mean_response_len = float(np.mean([len(ep.generated_token_ids) for ep in episodes]))

            print(
                f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
                f"train success_rate: {success_rate:.2f}, "
                f"grad_norm: {grad_norm:.2f}, duration: {duration_avg:.2f}, "
                f"num_finished_episodes: {int(num_finished_episodes_avg)}, "
                f"mean_response_len: {mean_response_len:.2f}, "
                f"entropy: {entropy:.2f}",
                end="",
                flush=True,
            )

            if tb_writer is not None:
                tb_writer.add_scalar("loss", loss, step)
                tb_writer.add_scalar("mean_reward", mean_reward, step)
                tb_writer.add_scalar("std_reward", std_reward, step)
                tb_writer.add_scalar("success_rate/train", success_rate, step)
                tb_writer.add_scalar("format_reward", format_reward, step)
                tb_writer.add_scalar("grad_norm", grad_norm, step)
                tb_writer.add_scalar("duration", duration_avg, step)
                tb_writer.add_scalar("num_finished_episodes", num_finished_episodes_avg, step)
                tb_writer.add_scalar("learning_rate", lr, step)
                tb_writer.add_scalar("entropy", entropy, step)
                for i, ep in enumerate(episodes[:4]):  # 텍스트 로그는 일부만
                    text = html.escape(ep.text)
                    tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # (선택) 주기적 barrier로 I/O 타이밍 정렬
        if step % 50 == 0:
            dist.barrier()

        # ------ Eval & Checkpoint (rank0만) ------
        if is_main_process() and step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate_rank0_only(model, tokenizer, device, dtype, config)  # 아래 8) 참고
            print(f"\nEval success rate: {eval_success_rate:.2f}")
            if tb_writer is not None:
                tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        if is_main_process() and step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            # DDP일 때는 model.module 이 실제 nn.Module
            torch.save((model.module if hasattr(model, "module") else model).state_dict(), output_file)
            print(f"\nSaved checkpoint to {output_file}")

    
    dist.barrier()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    dist.destroy_process_group()
    
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
    
    
    
# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 train.py --config config.yaml