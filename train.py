# train.py
import os
import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# file modules (동일한 인터페이스 가정)
from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


# ---------------------------
# DDP / Utility
# ---------------------------
def ddp_setup() -> Dict[str, int]:
    """torchrun이 설정한 ENV를 사용해 DDP 초기화."""
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "This script must be launched with torchrun. "
            "Example: torchrun --nproc_per_node=4 train.py --config config.yaml"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(base_seed: int, rank: int):
    # 분산 환경에서 rank마다 다른 seed로 설정
    seed = base_seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------
# Evaluation (rank0 전용)
# ---------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             tokenizer: Tokenizer,
             device: torch.device,
             dtype: torch.dtype,
             config: Dict[str, Any]) -> float:
    """간단 평가: rank0에서만 호출 (DDP밖에서는 그냥 단일 GPU 평가)."""
    model.eval()

    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )

    # 평가에선 DistributedSampler를 쓰지 않고, 단일 프로세스(rank0) 전용으로 순차 샘플링
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        batch_size=max(1, config["training"]["batch_size"] // 2),
        drop_last=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=True,
    )

    success = []
    for batch in test_loader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=int(config["training"]["max_gen_len"]) * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([e.reward_info["answer_reward"] for e in episodes])

    model.train()
    return float(np.mean(success)) if len(success) > 0 else 0.0


# ---------------------------
# Main
# ---------------------------
def main(config_path: str):
    # ---------------------------
    # DDP 초기화
    # ---------------------------
    ddp_info = ddp_setup()
    rank, local_rank, world_size = ddp_info["rank"], ddp_info["local_rank"], ddp_info["world_size"]

    # ---------------------------
    # Config/Device/Seed
    # ---------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # device/dtype
    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(str(config["model"]["dtype"]), torch.bfloat16)

    # seed
    set_seed(int(config["training"]["random_seed"]), rank)

    # (옵션) cudnn 설정
    torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = bool(config["training"].get("cudnn_deterministic", False))

    # ---------------------------
    # Paths / Tokenizer
    # ---------------------------
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    # ---------------------------
    # Dataset / Sampler / Loader
    # ---------------------------
    BATCH_SIZE = int(config["training"]["batch_size"])
    NUM_QUESTIONS_PER_BATCH = int(config["training"]["num_questions_per_batch"])
    assert BATCH_SIZE % NUM_QUESTIONS_PER_BATCH == 0, \
        f"batch_size({BATCH_SIZE}) % num_questions_per_batch({NUM_QUESTIONS_PER_BATCH}) != 0"
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=int(config["data"]["test_size"]),
    )
    assert len(train_dataset) > 0, "train_dataset 길이가 0입니다. 데이터 경로/전처리를 확인하세요."

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    # ❗ generator는 CPU 전용 또는 제거 (권장: 제거)
    train_loader = DataLoader(
        train_dataset,
        batch_size=NUM_QUESTIONS_PER_BATCH,
        sampler=train_sampler,       # shuffle은 sampler가 담당
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(config["training"].get("persistent_workers", False)) and int(config["training"].get("num_workers", 2)) > 0,
    )

    # ---------------------------
    # Model / Optimizer / DDP
    # ---------------------------
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    # DDP 래핑
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=tuple(config["training"]["betas"]),
        enabled=bool(config["training"]["memory_efficient_adamw"]),
    )

    # ---------------------------
    # Logging / Checkpoint (rank0만)
    # ---------------------------
    if is_main_process():
        current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
        log_dir = Path(config["training"]["log_dir"]) / current_time
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(log_dir))
        ckpt_dir = Path(config["training"]["ckpt_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        tb_writer = None
        ckpt_dir = None

    start_time = time.time()
    max_epochs = int(config["training"].get("epochs", 1))
    eval_interval = int(config["training"]["eval_interval"])
    ckpt_save_interval = int(config["training"]["ckpt_save_interval"])
    skip_unfinished = bool(config["training"]["skip_unfinished_episodes"])
    max_gen_len = int(config["training"]["max_gen_len"])
    micro_bs = int(config["training"]["micro_batch_size"])
    pad_token_id = tokenizer.pad_token_id
    max_grad_norm = float(config["training"]["max_grad_norm"])

    # ---------------------------
    # Train Loop
    # ---------------------------
    try:
        global_step = 0
        for epoch in range(max_epochs):
            # epoch마다 셔플 seed 갱신 (중요)
            train_sampler.set_epoch(epoch)

            for step, batch in enumerate(train_loader, start=1):
                global_step += 1

                episodes = rollout(
                    model=model.module,   # DDP 사용 시 module로 실제 모델 접근
                    tokenizer=tokenizer,
                    batch=batch,
                    max_gen_len=max_gen_len,
                    num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                    reward_function=reward_function,
                    device=device,
                    dtype=dtype,
                )
                if skip_unfinished:
                    episodes = [e for e in episodes if e.is_finished]

                results = update_policy(
                    model=model.module,
                    optimizer=optimizer,
                    episodes=episodes,
                    micro_batch_size=micro_bs,
                    pad_token_id=pad_token_id,
                    max_grad_norm=max_grad_norm,
                    device=device,
                    dtype=dtype,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                # ---------------------------
                # Rank0 Logging
                # ---------------------------
                if is_main_process():
                    end_time = time.time()
                    duration = end_time - start_time
                    start_time = end_time

                    reward = [e.reward for e in episodes]
                    formatted_reward = [e.reward_info["format_reward"] for e in episodes]
                    answer_reward = [e.reward_info["answer_reward"] for e in episodes]
                    num_finished = sum(e.is_finished for e in episodes)
                    mean_reward = float(np.mean(reward)) if reward else 0.0
                    std_reward = float(np.std(reward)) if reward else 0.0
                    success_rate = float(np.mean(answer_reward)) if answer_reward else 0.0
                    format_reward = float(np.mean(formatted_reward)) if formatted_reward else 0.0
                    grad_norm = float(results["grad_norm"])
                    entropy = float(results["entropy"])
                    lr = float(optimizer.param_groups[0]["lr"])
                    loss = float(results["loss"])
                    mean_response_len = float(np.mean([len(e.generated_token_ids) for e in episodes])) if episodes else 0.0

                    print(
                        f"\rEpoch {epoch} Step {step}, "
                        f"mean_reward: {mean_reward:.2f}, "
                        f"train success_rate: {success_rate:.2f}, "
                        f"grad_norm: {grad_norm:.2f}, "
                        f"duration: {duration:.2f}, "
                        f"num_finished_episodes: {num_finished}, "
                        f"mean_response_len: {mean_response_len:.2f}, "
                        f"entropy: {entropy:.2f}",
                        end="",
                        flush=True,
                    )

                    # TensorBoard
                    tb_writer.add_scalar("loss", loss, global_step)
                    tb_writer.add_scalar("mean_reward", mean_reward, global_step)
                    tb_writer.add_scalar("std_reward", std_reward, global_step)
                    tb_writer.add_scalar("success_rate/train", success_rate, global_step)
                    tb_writer.add_scalar("format_reward", format_reward, global_step)
                    tb_writer.add_scalar("grad_norm", grad_norm, global_step)
                    tb_writer.add_scalar("duration", duration, global_step)
                    tb_writer.add_scalar("learning_rate", lr, global_step)
                    tb_writer.add_scalar("mean_response_len", mean_response_len, global_step)
                    tb_writer.add_scalar("entropy", entropy, global_step)

                    # text 로그 (markdown으로 렌더)
                    for i, e in enumerate(episodes):
                        tb_writer.add_text(f"text_{i}", f"<pre>{html.escape(e.text)}</pre>", global_step)

                    # 평가
                    if (global_step % eval_interval) == 0:
                        eval_sr = evaluate(model.module, tokenizer, device, dtype, config)
                        print(f"\nEval success rate: {eval_sr:.2f}")
                        tb_writer.add_scalar("success_rate/eval", eval_sr, global_step)

                    # 체크포인트
                    if (global_step % ckpt_save_interval) == 0:
                        out = Path(ckpt_dir) / f"ckpt_{epoch:02d}_{global_step:06d}.pt"
                        torch.save(model.module.state_dict(), out)
                        print(f"\nSaved checkpoint to {out}")

        # 에폭 종료 동기화
        dist.barrier()

    finally:
        # rank0 writer 정리
        if is_main_process() and tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        cleanup_ddp()


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)


#OMP_NUM_THREADS=4 uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 train.py --config config.yaml