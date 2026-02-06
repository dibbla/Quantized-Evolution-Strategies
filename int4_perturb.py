# In this script, we implement the QuZO perturbation, but update in the standard ES eay

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import os
import random
import shutil
import time
from typing import List, Dict, Any

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.utils import get_ip, get_open_port
import tempfile

from countdown.countdown_task import reward_function
from countdown.countdown_prompts import SYSTEM_MESSAGE, USER_TEMPLATE, RESPONSE_PROMPT

# Default Hyperparameters
SIGMA = 0.0005
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 800
EXPERIMENT_DIR = "es-ft-experiment"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Countdown Task with static scheduling"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--alpha_warmup_start",
        type=float,
        default=None,
        help=(
            "Starting alpha for linear warmup. If set and --alpha_warmup_steps>0, "
            "alpha is linearly ramped from this value to --alpha. "
            "If unset, warmup is disabled unless --alpha_warmup_steps is 0."
        ),
    )
    parser.add_argument(
        "--alpha_warmup_steps",
        type=int,
        default=0,
        help=(
            "Number of warmup iterations for alpha (linear). "
            "Warmup uses alpha_warmup_start at iteration 0 and reaches --alpha at iteration (steps-1). "
            "0 disables warmup."
        ),
    )
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")
    parser.add_argument("--global_seed", type=int, help="Global random seed")

    # Evaluation args
    parser.add_argument("--eval_data_path", type=str,
                        default="data/countdown.json",
                        help="Path to evaluation JSON (same schema as train)")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="Evaluate every N iterations (0 disables)")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Batch size for evaluation generation")
    
    # Quantization args
    parser.add_argument("--save_residual", action="store_true",
                        help="Whether to save residuals for discrete ES updates")
    parser.add_argument("--momentum", type=float, default=0.3,
                        help="Momentum for residual updates if --save_residual is set")
    parser.add_argument("--p_zero", type=float, default=0.9,
                        help="Probability of sampling zero in weight perturbation")
    parser.add_argument("--stochastic", action="store_true",
                        help="Whether to use stochastic discretization in ES updates")
    parser.add_argument("--discrete_step_gain", type=float, default=1.0,
                        help="Gain for discrete ES vote in code space")
    parser.add_argument("--disc_threshold", type=float, default=0.5,
                        help="Deterministic discretization threshold (>= abs(acc) -> step)")
    parser.add_argument(
        "--update_scales",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable continuous scale updates during ES (default: enabled; use --no-update-scales to disable)",
    )
    parser.add_argument("--coeff_clip", type=float, default=3.0,
                        help="Clip normalized rewards to [-coeff_clip, coeff_clip]")
    parser.add_argument("--mirror_sampling", action="store_true",
                        help="Whether to use mirror sampling for ES")

    # Reward shaping / ES weighting
    parser.add_argument(
        "--reward_weighting",
        type=str,
        default="zscore",
        choices=["zscore", "rank"],
        help="How to convert per-seed rewards into ES coefficients. "
             "'zscore' reproduces prior behavior; 'rank' uses centered rank weights (more robust for int4).",
    )
    parser.add_argument(
        "--elite_frac",
        type=float,
        default=1.0,
        help="For --reward_weighting rank: keep only the top elite_frac seeds (0<elite_frac<=1) and zero out others. "
             "Use 1.0 to use all seeds.",
    )

    # For different experiments
    parser.add_argument("--worker_extn_cls", type=str,
                        default="utils_int4.worker_extn_seed_replay.SeedReplayESExtension",
                        help="Worker extension class for VLLM LLM")
    parser.add_argument("--window_size", type=int,
                        default=50,
                        help="Window size for seed replay history")
    parser.add_argument("--decay", type=float,
                        default=0.9,
                        help="Decay for residuals in seed replay")
    

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)
        os.environ["PYTHONHASHSEED"] = str(args.global_seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True) # Connot use with GPTQ
        except Exception:
            pass

    return args


def _alpha_for_iteration(iter_idx: int, args) -> float:
    """Compute per-iteration learning rate (alpha) with optional linear warmup.

    - If warmup is disabled, returns args.alpha.
    - If enabled, uses args.alpha_warmup_start at iter 0 and reaches args.alpha at iter (warmup_steps-1).
    """
    target = float(args.alpha)
    warm_steps = int(getattr(args, "alpha_warmup_steps", 0) or 0)
    start = getattr(args, "alpha_warmup_start", None)

    if warm_steps <= 0 or start is None:
        return target

    start_f = float(start)
    if warm_steps == 1:
        return target

    if iter_idx >= warm_steps:
        return target

    t = float(iter_idx) / float(warm_steps - 1)
    return start_f + t * (target - start_f)


def _assign_rank_weights(seeds: List[int], seeds_perf: Dict[int, Dict[str, Any]], elite_frac: float) -> None:
    """Populate seeds_perf[seed]['norm_reward'] using centered rank weights.

    - Best seed gets weight +1, worst gets -1 (approximately), ties broken by order.
    - If elite_frac < 1, non-elites get weight 0.
    """
    n = len(seeds)
    if n == 0:
        return

    rewards = np.array([float(seeds_perf[int(s)]["avg_reward"]) for s in seeds], dtype=np.float64)
    order = np.argsort(-rewards, kind="stable")  # descending: best first

    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)  # 0 is best

    denom = max(1, n - 1)
    weights = 1.0 - 2.0 * (ranks.astype(np.float64) / float(denom))  # best=+1, worst=-1

    ef = float(elite_frac)
    if not (0.0 < ef <= 1.0):
        raise ValueError(f"elite_frac must be in (0, 1], got {elite_frac}")
    elite_n = int(np.ceil(ef * n))
    elite_n = max(1, min(n, elite_n))
    elite_mask = ranks < elite_n
    weights = np.where(elite_mask, weights, 0.0)

    for idx, seed in enumerate(seeds):
        seeds_perf[int(seed)]["norm_reward"] = float(weights[idx])

# Align with GRPOZero
def _process_context(task_data, tokenizer):
    numbers = task_data["numbers"]
    target = task_data["target"]
    user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]

    messages = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    ) # + RESPONSE_PROMPT

    prompts = tokenizer(
        messages,
    )

    return TokensPrompt(prompt_token_ids=prompts['input_ids'])


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Avoid adding non-serializable objects; rely on worker-side ops
        super().__init__(*args, **kwargs)

def launch_int4_engines(num_engines, model_name, worker_extn_cls="utils_int4.worker_extn_seed_replay.SeedReplayESExtension", **kwargs):
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls=worker_extn_cls,
            enable_prefix_caching=False,
            enforce_eager=False,
            gpu_memory_utilization=0.9,
            # quantization="gptq_marlin"
        )
        for strategy in strategies
    ]

    print(f"Launched {len(engines)} int4 QuZO engines with worker extension {worker_extn_cls}")
    
    return engines, pgs

def evaluate_countdown(llm, prompts, seed: int):
    # NOTE: We simply hardcoded the seed
    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)
    return llm.generate.remote(
        prompts, 
        sampling_params, 
        use_tqdm=False,
    )

def _postprocess_outputs(outputs, task_datas):
    rewards = []
    avg_rewards = []
    format_rewards = []
    answer_rewards = []
    for output, data in zip(outputs, task_datas):
        response = RESPONSE_PROMPT + output.outputs[0].text
        r = reward_function(response, data["numbers"], data["target"])
        rewards.append(r)
        avg_rewards.append(r["reward"])
        if "reward_info" in r:
            format_rewards.append(r["reward_info"].get("format_reward", 0.0))
            answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
        "avg_format": avg_format,
        "avg_answer": avg_answer,
        "accuracy": accuracy,
    }

def evaluate_model(llm, eval_task_datas: List[Dict[str, Any]], writer, step: int, args):
    if not eval_task_datas:
        return
    batch_size = max(1, args.eval_batch_size)
    eval_seed = args.global_seed if args.global_seed is not None else 999
    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=1024) # NOTE: hardcoded seed

    all_rewards = []
    format_rewards = []
    answer_rewards = []
    start = time.time()

    for b in range(0, len(eval_task_datas), batch_size):
        batch = eval_task_datas[b:b+batch_size]
        prompts = [d["context"] for d in batch]

        outputs = ray.get(
            llm.generate.remote(
                prompts, 
                sampling_params, 
                use_tqdm=False
            )
        )
        
        for idx,(out, data) in enumerate(zip(outputs, batch)):
            # for inspection
            if idx == 0:
                print(f"Eval Sample Response:\n{RESPONSE_PROMPT + out.outputs[0].text}\n---")
                print(f"Target Answer: {data['target']}\n===")
                print(f"The rewards: {reward_function(RESPONSE_PROMPT + out.outputs[0].text, data['numbers'], data['target'])}\n===")
            response = RESPONSE_PROMPT + out.outputs[0].text
            r = reward_function(response, data["numbers"], data["target"])
            all_rewards.append(r["reward"])
            if "reward_info" in r:
                format_rewards.append(r["reward_info"].get("format_reward", 0.0))
                answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
    elapsed = time.time() - start

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    min_reward = float(np.min(all_rewards)) if all_rewards else 0.0
    max_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0

    print(f"[Eval @ step {step}] avg_reward={avg_reward:.4f} ± {std_reward:.4f} "
          f"range=[{min_reward:.4f},{max_reward:.4f}] format={avg_format:.4f} "
          f"answer={avg_answer:.4f} acc={accuracy:.1f}% time={elapsed:.2f}s")

    writer.add_scalar("eval/avg_reward", avg_reward, step)
    writer.add_scalar("eval/std_reward", std_reward, step)
    writer.add_scalar("eval/min_reward", min_reward, step)
    writer.add_scalar("eval/max_reward", max_reward, step)
    writer.add_scalar("eval/format_reward", avg_format, step)
    writer.add_scalar("eval/answer_reward", avg_answer, step)
    writer.add_scalar("eval/accuracy", accuracy, step)
    writer.add_scalar("eval/time", elapsed, step)


@dataclass
class _TimingAccumulator:
    perturb: float = 0.0
    infer: float = 0.0
    restore: float = 0.0
    update_s: float = 0.0
    broadcast: float = 0.0
    total: float = 0.0
    iters: int = 0

    def update(self, *, per_seed: Dict[str, float]) -> None:
        self.iters += 1
        self.perturb += float(per_seed["perturb"])
        self.infer += float(per_seed["infer"])
        self.restore += float(per_seed["restore"])
        self.update_s += float(per_seed["update"])
        self.broadcast += float(per_seed["broadcast"])
        self.total += float(per_seed["total"])

    def avg_total(self) -> float:
        return float(self.total / max(1, self.iters))


def _log_iteration_timings(
    *,
    writer: SummaryWriter,
    step: int,
    pop: int,
    iter_weighted_wall: Dict[str, float],
    update_elapsed: float,
    bcast_elapsed: float,
    iter_elapsed: float,
    timing_acc: _TimingAccumulator,
) -> None:
    pop_i = max(1, int(pop))

    # Interpret the measured timings consistently:
    # - iter_weighted_wall holds sum(batch_wall_time * batch_size)
    # - per-seed wall time is weighted_sum / population
    wall_per_seed = {
        "perturb": float(iter_weighted_wall["perturb"] / pop_i),
        "infer": float(iter_weighted_wall["infer"] / pop_i),
        "restore": float(iter_weighted_wall["restore"] / pop_i),
        "update": float(update_elapsed / pop_i),
        "broadcast": float(bcast_elapsed / pop_i),
    }
    wall_per_seed["total"] = float(
        wall_per_seed["perturb"]
        + wall_per_seed["infer"]
        + wall_per_seed["restore"]
        + wall_per_seed["update"]
        + wall_per_seed["broadcast"]
    )

    # Wall times for the iteration
    writer.add_scalar("time/wall/iter_total_s", float(iter_elapsed), step)
    writer.add_scalar("time/wall/perturb_s", wall_per_seed["perturb"], step)
    writer.add_scalar("time/wall/infer_s", wall_per_seed["infer"], step)
    writer.add_scalar("time/wall/restore_s", wall_per_seed["restore"], step)
    writer.add_scalar("time/wall/update_s", float(update_elapsed), step)
    writer.add_scalar("time/wall/broadcast_s", float(bcast_elapsed), step)

    # Per-seed times (helps compare different population sizes)
    writer.add_scalar("time/per_seed/perturb_s", wall_per_seed["perturb"], step)
    writer.add_scalar("time/per_seed/infer_s", wall_per_seed["infer"], step)
    writer.add_scalar("time/per_seed/restore_s", wall_per_seed["restore"], step)
    writer.add_scalar("time/per_seed/update_s", wall_per_seed["update"], step)
    writer.add_scalar("time/per_seed/broadcast_s", wall_per_seed["broadcast"], step)
    writer.add_scalar("time/per_seed/total_s", wall_per_seed["total"], step)

    # Simple throughput
    writer.add_scalar(
        "time/throughput/seeds_per_s",
        float(pop_i / max(1e-9, float(iter_elapsed))),
        step,
    )

    # Cumulative running average (per-seed)
    timing_acc.update(per_seed=wall_per_seed)
    writer.add_scalar("time/cum_avg_per_seed/total_s", timing_acc.avg_total(), step)


def main(args):
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

    unique_dir = tempfile.mkdtemp(prefix=f"ray_temp_session_{int(time.time())}_")

    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=unique_dir,
        dashboard_port=None
    )

    logging_dir = f"{args.experiment_dir}/countdown_nccl_static_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # dump the args into a json file
    with open(f"{logging_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_path = "data/countdown.json"
    with open(data_path, "r") as f:
        task_datas = json.load(f)
    task_datas = task_datas[:200]

    eval_task_datas = []
    if args.eval_interval > 0 and os.path.exists(args.eval_data_path):
        with open(args.eval_data_path, "r") as f:
            eval_task_datas = json.load(f)[200:]
        print(f"Loaded {len(eval_task_datas)} eval samples from {args.eval_data_path}")

        # pre-process eval contexts
        for d in eval_task_datas:
            d["context"] = _process_context(d, tokenizer)

    engines, pgs = launch_int4_engines(args.num_engines, args.model_name, worker_extn_cls=args.worker_extn_cls)

    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])


    def cleanup():
        for llm in engines:
            try: ray.kill(llm)
            except: pass
        for pg in pgs:
            try: remove_placement_group(pg)
            except: pass
        ray.shutdown()

    prompts = [_process_context(d, tokenizer) for d in task_datas]

    timing_acc = _TimingAccumulator()

    # Rolling moving-average tracker for reward (helps visualize noisy ES progress)
    reward_ma_window = 10
    reward_ma_buf = deque(maxlen=reward_ma_window)

    for i in range(args.num_iterations):
        print(f"\n\n=== Generation {i} (static scheduling) ===")
        total_iter_start = time.time()

        alpha_t = _alpha_for_iteration(i, args)
        writer.add_scalar("opt/alpha", float(alpha_t), i)

        # Per-iteration weighted wall times (sum of batch_wall_time * batch_size)
        iter_weighted_wall = {
            "perturb": 0.0,
            "infer": 0.0,
            "restore": 0.0,
        }

        # Deterministic per-iteration seed list
        loop_rng = np.random.default_rng(seed=(args.global_seed or 42) + i)

        if args.mirror_sampling:
            if args.population_size % 2 != 0:
                raise ValueError(
                    f"--mirror_sampling requires even --population_size, got {args.population_size}"
                )
            num_pairs = args.population_size // 2
            base_seeds = loop_rng.integers(0, 2**30, size=num_pairs, dtype=np.int64).tolist()
            # Interleave so batching distributes +/- across engines reasonably
            seeds = []
            for s in base_seeds:
                seeds.append(int(s))
                seeds.append(-int(s))
        else:
            seeds = loop_rng.integers(0, 2**30, size=args.population_size, dtype=np.int64).tolist()

        seeds_perf: Dict[int, Dict[str, Any]] = {}

        # Static batching: issue exactly one seed per engine per batch in fixed order, then wait
        for b in range(0, len(seeds), args.num_engines):
            batch = seeds[b:b+args.num_engines]
            # 1) Perturb
            _t0 = time.time()
            ray.get([
                engines[eng_idx].collective_rpc.remote(
                    "perturb_self_weights",
                    kwargs={
                        "seed": int(seed),
                        "sigma": args.sigma,
                    }
                )
                for eng_idx, seed in enumerate(batch)
            ])
            iter_weighted_wall["perturb"] += (time.time() - _t0) * len(batch)

            # 2) Generate with fixed generation seed tied to iteration
            gen_seed = (args.global_seed or 42) + i
            handles = [
                evaluate_countdown(engines[eng_idx], prompts, seed=gen_seed)
                for eng_idx, _ in enumerate(batch)
            ]

            # 3) Collect outputs in the same order
            _t1 = time.time()
            outputs_per_engine = ray.get(handles)
            iter_weighted_wall["infer"] += (time.time() - _t1) * len(batch)

            # 4) Restore weights
            _t2 = time.time()
            ray.get([
                engines[eng_idx].collective_rpc.remote(
                    "restore_self_weights",
                    kwargs={
                        "seed": int(seed),
                        "sigma": args.sigma,
                    }
                )
                for eng_idx, seed in enumerate(batch)
            ])
            iter_weighted_wall["restore"] += (time.time() - _t2) * len(batch)

            # 5) Score and record
            for eng_idx, seed in enumerate(batch):
                metrics = _postprocess_outputs(outputs_per_engine[eng_idx], task_datas)
                seeds_perf[int(seed)] = metrics

        # Aggregate (over *all* evaluated seeds, including +/- if mirror is on)
        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        reward_ma_buf.append(mean_reward)
        mean_reward_ma = float(np.mean(reward_ma_buf)) if reward_ma_buf else mean_reward

        # Also aggregate format and answer rewards and accuracy over seeds
        all_avg_formats = [v.get("avg_format", 0.0) for v in seeds_perf.values()]
        all_avg_answers = [v.get("avg_answer", 0.0) for v in seeds_perf.values()]
        all_accuracies = [v.get("accuracy", 0.0) for v in seeds_perf.values()]
        mean_format = float(np.mean(all_avg_formats)) if all_avg_formats else 0.0
        mean_answer = float(np.mean(all_avg_answers)) if all_avg_answers else 0.0
        mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0.0

        print(
            f"Mean reward: {mean_reward:.4f}, std: {std_reward:.4f}, "
            f"format: {mean_format:.4f}, answer: {mean_answer:.4f}, acc: {mean_accuracy:.1f}%"
        )

        # === ES coefficient shaping ===
        if args.mirror_sampling:
            # Build per-pair diffs: diff = r(+s) - r(-s)
            base_seeds = [abs(int(s)) for s in seeds[::2]]  # because we interleaved [+s,-s,+t,-t,...]
            # (If someone later changes ordering, still safe-ish: we just use unique abs seeds in insertion order)
            seen = set()
            base_seeds_ordered = []
            for s in base_seeds:
                if s not in seen:
                    seen.add(s)
                    base_seeds_ordered.append(s)

            pair_perf: Dict[int, Dict[str, Any]] = {}
            diffs = []
            for s in base_seeds_ordered:
                sp = int(s)
                sn = -int(s)
                if sp not in seeds_perf or sn not in seeds_perf:
                    raise RuntimeError(f"Mirror sampling missing pair evaluations for seed {s}: have +? {sp in seeds_perf}, -? {sn in seeds_perf}")
                diff = float(seeds_perf[sp]["avg_reward"]) - float(seeds_perf[sn]["avg_reward"])
                pair_perf[sp] = {"avg_reward": diff}
                diffs.append(diff)

            diff_mean = float(np.mean(diffs)) if diffs else 0.0
            diff_std = float(np.std(diffs)) if diffs else 0.0
            writer.add_scalar("mirror/diff_mean", diff_mean, i)
            writer.add_scalar("mirror/diff_std", diff_std, i)

            if args.reward_weighting == "zscore":
                for s in base_seeds_ordered:
                    d = float(pair_perf[int(s)]["avg_reward"])
                    nr = (d - diff_mean) / (diff_std + 1e-8)
                    c = float(np.clip(nr, -args.coeff_clip, args.coeff_clip))
                    # Symmetric coefficients across the mirrored pair
                    seeds_perf[int(s)]["norm_reward"] = c
                    seeds_perf[-int(s)]["norm_reward"] = -c
            elif args.reward_weighting == "rank":
                _assign_rank_weights([int(s) for s in base_seeds_ordered], pair_perf, args.elite_frac)
                for s in base_seeds_ordered:
                    c = float(pair_perf[int(s)]["norm_reward"])
                    seeds_perf[int(s)]["norm_reward"] = c
                    seeds_perf[-int(s)]["norm_reward"] = -c
            else:
                raise ValueError(f"Unknown reward_weighting: {args.reward_weighting}")

        else:
            if args.reward_weighting == "zscore":
                for k in seeds_perf:
                    nr = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
                    seeds_perf[k]["norm_reward"] = float(np.clip(nr, -args.coeff_clip, args.coeff_clip))
            elif args.reward_weighting == "rank":
                _assign_rank_weights([int(s) for s in seeds], seeds_perf, args.elite_frac)
            else:
                raise ValueError(f"Unknown reward_weighting: {args.reward_weighting}")

        writer.add_scalar("reward/mean", mean_reward, i)
        writer.add_scalar(f"reward/mean_ma_w{reward_ma_window}", mean_reward_ma, i)
        writer.add_scalar("reward/std", std_reward, i)
        writer.add_scalar("reward/min", min_reward, i)
        writer.add_scalar("reward/max", max_reward, i)
        writer.add_scalar("train/format_reward", mean_format, i)
        writer.add_scalar("train/answer_reward", mean_answer, i)
        writer.add_scalar("train/accuracy", mean_accuracy, i)

        # Update weights on engine 0 using batched method
        per_seed_coeffs = [
            (int(seed), float(seeds_perf[int(seed)]["norm_reward"]) * float(args.discrete_step_gain))
            for seed in seeds
        ]
        update_start = time.time()
        upd_res = ray.get(engines[0].collective_rpc.remote(
            "apply_quzo_perturb_update",
            args=(per_seed_coeffs, args.sigma, float(alpha_t), args.decay, args.window_size)
        ))[0] # ray is going to return a list of results from all engines, but we have 1 engine
        update_elapsed = time.time() - update_start
        if args.verbose:
            print(f"Applied updates in {update_elapsed}s; res={upd_res}")

        if isinstance(upd_res, dict):
            if "boundary_hits" in upd_res:
                writer.add_scalar("update/boundary_hits", float(upd_res["boundary_hits"]), i)
            if "boundary_hit_ratio" in upd_res:
                writer.add_scalar(
                    "update/boundary_hit_ratio",
                    float(upd_res["boundary_hit_ratio"]),
                    i,
                )

        # Broadcast from engine 0
        bcast_start = time.time()
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        bcast_elapsed = time.time() - bcast_start
        torch.cuda.synchronize()
        iter_elapsed = time.time() - total_iter_start
        print(f"Iteration {i} finished in {iter_elapsed:.2f}s")

        # Compute per-seed averages for this iteration
        pop = max(1, args.population_size)
        _log_iteration_timings(
            writer=writer,
            step=i,
            pop=pop,
            iter_weighted_wall=iter_weighted_wall,
            update_elapsed=update_elapsed,
            bcast_elapsed=bcast_elapsed,
            iter_elapsed=iter_elapsed,
            timing_acc=timing_acc,
        )

        if i % 10 == 0:
            writer.flush()

        if (args.eval_interval > 0 and (i % args.eval_interval == 0)) or (i == args.num_iterations - 1):
            evaluate_model(engines[0], eval_task_datas, writer, i, args)

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    main(args)