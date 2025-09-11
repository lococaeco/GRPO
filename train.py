
import os
import time
import yaml
import math
import re

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from argparse import ArgumentParser
from datetime import datetime
import pathlib
import numpy as np
import torch
from torch.optim import AdamW
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from qwen2_model import Transformer

# file module
# from countdown_task import CountdownTasksDataset, reward_function
# from grpo import rollout, update_policy
# from optimizer import MemoryEfficientAdamW
# from tokenizer import Tokenizer

def ddp_setup():
    dist.init_process_group("nccl")   # GPU 분산 학습 backend
    local_rank = int(os.environ["LOCAL_RANK"])  # torchrun이 환경변수로 줌
    torch.cuda.set_device(local_rank)           # 현재 프로세스가 특정 GPU만 바라보도록
    device = torch.device(f"cuda:{local_rank}") # 이 프로세스가 쓸 device
    return device

####################################################################
@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]


@dataclass
class MiniBatch:
    """Batch of data for each training step."""  

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]] # Batchsize x ids Length
    numbers: List[List[int]]
    target: List[int]
####################################################################

import json
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    def __init__(self, tokenizer_path: str):
        super().__init__()
        tokenizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokenizer_config_path))
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)
        self.chat_template = Environment().from_string(
            self.tokenizer_config["chat_template"]
        )
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

####################################################################


import pandas as pd

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

class CountdownTasksDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, split: str, test_size: int):
        data = pd.read_parquet(pathlib.Path(data_path) / "data")
        self.data = (data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): # 인덱스의 정보를 읽고 dict로 변경하여 반환
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["nums"], item["target"])) #기초직인 사칙연산에 관한 데이터셋인가 보오
        return item
    
    # Token으로 인코딩 , prefix는 모델에게 넣어주는 입력
    def encode_prefix(self, numbers: List[int], target: int):
        """Prefix is the *actual* input to the model."""
        user_message = (
                        f"Using the numbers {numbers}, create an equation that equals {target}. "
                        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                        "Show your work in <think> </think> tags. "
                        "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
                       )
        prefix = self.tokenizer.encode_chat_with_response_prompt( #Tokenizer에 템플릿이 정해져있나보네
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens, #["Using", " the", " numbers", " [", "3", ",", " 5", ",", " 7", "]", ...]
            "prefix_token_ids": tokens.ids, #[3512, 117, 9821, 191, 123, 16, 456, 16, 789, 195, ...]
        }

    # 실제로 미니 배치로 만들어주는 함수
    @staticmethod 
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch: 
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )
    

def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


####################################################################


class MemoryEfficientAdamW(AdamW):
    # 속도, 가속도와 같은 Parameter의 gradient 정보들을 CPU를 통해 복사하고 RAM에 저장하여 GPU 공간을 아끼겠다.
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        pin_memory=True,
        enabled=True,
    ):
        super(MemoryEfficientAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.pin_memory = pin_memory
        self.enabled = enabled

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        if not self.enabled:
            # Use the parent AdamW implementation when disabled
            return super(MemoryEfficientAdamW, self).step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Store optimizer states on CPU with pinned memory
                    device = "cpu"
                    pin_memory = self.pin_memory
                    dtype = torch.float32

                    state["exp_avg"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p.data, device=device, pin_memory=pin_memory, dtype=dtype
                        )

                # Get state values
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            # Process all parameters in the group
            self._memory_efficient_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def _memory_efficient_update(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
    ):
        """
        Performs the AdamW parameter update on GPU with CPU-stored optimizer states.
        Uses pinned memory for efficient CPU-to-GPU transfer of optimizer states.
        """
        for i, param in enumerate(params):
            grad = grads[i]
            param_device = param.device

            # Access optimizer states - they'll transfer efficiently due to pin_memory
            exp_avg = exp_avgs[i].to(param_device, non_blocking=True)
            exp_avg_sq = exp_avg_sqs[i].to(param_device, non_blocking=True)

            step = state_steps[i]

            # Decay the first and second moment running averages
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                # Access max_exp_avg_sq - transfers efficiently with pin_memory
                max_exp_avg_sq = max_exp_avg_sqs[i].to(param_device, non_blocking=True)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max for normalizing running avg of gradient
                denom = max_exp_avg_sq.sqrt().add_(eps)
                # Store back to CPU
                max_exp_avg_sqs[i].copy_(max_exp_avg_sq, non_blocking=True)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            # Apply weight decay directly to the parameter (AdamW)
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            # Update parameters (directly on GPU)
            param.addcdiv_(exp_avg, denom, value=-step_size)

            # Store optimizer states back to CPU
            exp_avgs[i].copy_(exp_avg, non_blocking=True)
            exp_avg_sqs[i].copy_(exp_avg_sq, non_blocking=True)

####################################################################
import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update the policy using the GRPO algorithm."""
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }

####################################################################


def main():

    # pathlib.Path()를 통해 여러 메서드를 사용할 수 있음. ex) exists(), is_file()
    pretrained_model_path = pathlib.Path("Qwen2.5-3B-Instruct")
    print(pretrained_model_path)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    #.get("key, default")
    dtype = dtype_map.get("bfloat16", torch.bfloat16)

    torch.manual_seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
     
    # BATCH_SIZE = config["training"]["batch_size"]
    # NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    # NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    BATCH_SIZE = 256
    NUM_QUESTIONS_PER_BATCH = 32
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH #8 

    tokenizer_path = str(pretrained_model_path / "tokenizer.json")
    print(tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)


    device = ddp_setup()
    print(f"Using device: {device}")

    train_dataset = CountdownTasksDataset(
        data_path="Countdown-Tasks-3to4",
        tokenizer=tokenizer,
        split="train",
        test_size=100,
    )

    train_data_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=dist.get_world_size(), # 전체 GPU 수
        rank = dist.get_rank(),             # 현재 GPU Rank
        shuffle=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=NUM_QUESTIONS_PER_BATCH,
        sampler=train_data_sampler,                     # sampler 연결
        collate_fn=CountdownTasksDataset.collate_fn,    # batch 만들기
        num_workers=4,                                  # (선택) CPU 병렬 로딩
        pin_memory=True,                                # (선택) GPU 학습 속도 향상
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],   # 현재 프로세스의 GPU
        output_device=device.index,
    )

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=1.0e-5,
        weight_decay=0.0,
        betas=[0.9, 0.999],
        enabled=True,
    )

    ckpt_dir = pathlib.Path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epoch_num = 1
    for epoch in range(epoch_num):
        train_data_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader, start=1):
            episodes = rollout(
                model=model.module,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=1024,
                num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
            )
            skip_unfinished_episodes = False
            if skip_unfinished_episodes:
                episodes = [episode for episode in episodes if episode.is_finished]
            
            results = update_policy(
                # model=model.module,
                model=model,
                optimizer=optimizer,
                episodes=episodes,
                micro_batch_size=2,
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=1.0,
                device=device,
                dtype=dtype,
            )

            # compute and log important metrics
            reward = [episode.reward for episode in episodes]
            formatted_reward = [
                episode.reward_info["format_reward"] for episode in episodes
            ]
            answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
            num_finished_episodes = sum(episode.is_finished for episode in episodes)
            mean_reward = np.mean(reward)
            std_reward = np.std(reward)
            success_rate = np.mean(answer_reward)
            format_reward = np.mean(formatted_reward)
            grad_norm = results["grad_norm"]
            entropy = results["entropy"]
            lr = optimizer.param_groups[0]["lr"]
            loss = results["loss"]
            mean_response_len = np.mean(
                [len(episode.generated_token_ids) for episode in episodes]
            )

            print(
                f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
                f"train success_rate: {success_rate:.2f}, "
                f"num_finished_episodes: {num_finished_episodes}, "
                f"mean_response_len: {mean_response_len:.2f}, "
                f"entropy: {entropy:.2f}")
            
        #     if step % eval_interval == 0:
        #         eval_success_rate = evaluate()
        #         print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)

            if dist.get_rank() == 0 and step % 100 == 0:
                output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
                torch.save(model.module.state_dict(), output_file)
                print(f"Saved checkpoint to {output_file}")


    dist.destroy_process_group()

if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--config", type=str, default="config.yaml")
    # args = parser.parse_args()
    # main(args.config)
    main()

#OMP_NUM_THREADS=12 uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 train.py --config config.yaml