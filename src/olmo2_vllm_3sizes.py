
"""
Evaluate OLMo 2 pretraining checkpoints on abstractive ICL tasks.

Example usage:
    python src/eval_olmo2.py
    python src/eval_olmo2.py --config config/eval_olmo2_config.json
    python src/eval_olmo2.py --model_size 7b
    python src/eval_olmo2.py --model_size 1b --revisions stage1-step1000-tokens2B stage1-step20000-tokens42B
    python src/eval_olmo2.py --tasks antonym country-capital
    python src/eval_olmo2.py --resume

vLLM examples:
    python src/eval_olmo2_vllm.py --model_size 7b --resume
    python src/eval_olmo2_vllm.py --model_size 13b --tensor_parallel_size 2 --resume
    python src/eval_olmo2_vllm.py --model_size 32b --tensor_parallel_size 4 --gpu_memory_utilization 0.92

Notes:
- This script auto-discovers all HF-accessible *pretraining* revisions for each OLMo 2 size.
- For 1B, it includes both:
    - allenai/OLMo-2-0425-1B
    - allenai/OLMo-2-0425-1B-early-training
- Results are keyed by revision string, not integer checkpoint step, because OLMo 2
  revisions are irregular and named like:
    - stage1-step140000-tokens294B
    - stage2-ingredient3-step23852-tokens51B
"""

from __future__ import annotations

import sys
import json
import argparse
import time
import re
import gc
from tqdm import tqdm
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, scan_cache_dir
from vllm import LLM, SamplingParams

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
PROJECT_ROOT = Path("/projectnb/cs505am/students/amao/icl-without-copying")
TASK_DIR = PROJECT_ROOT / "icl_tasks" / "abstractive"

sys.path.insert(0, str(PROJECT_ROOT))
from lib.fv import fv_icl_tasks_benchmark  # noqa: E402

ALL_TASKS = [f.stem for f in sorted(TASK_DIR.glob("*.json"))]

# -----------------------------------------------------------------------------
# Model repos
# -----------------------------------------------------------------------------

MODEL_REPOS = {
    "1b": [
        "allenai/OLMo-2-0425-1B-early-training",
        "allenai/OLMo-2-0425-1B",
    ],
    "7b": [
        "allenai/OLMo-2-1124-7B",
    ],
    "13b": [
        "allenai/OLMo-2-1124-13B",
    ],
    "32b": [
        "allenai/OLMo-2-0325-32B",
    ],
}

# Keep only pretraining revisions.
PRETRAIN_PATTERNS = [
    re.compile(r"^stage1-step(\d+)-tokens([0-9]+(?:\.[0-9]+)?)B$"),
    re.compile(r"^stage2-ingredient(\d+)-step(\d+)-tokens([0-9]+(?:\.[0-9]+)?)B$"),
    re.compile(r"^step(\d+)-tokens([0-9]+(?:\.[0-9]+)?)B$"),
]

EXCLUDE_KEYWORDS = {
    "sft",
    "dpo",
    "instruct",
    "rl",
    "rlvr",
    "reward",
    "rm",
    "chat",
    "preview",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def is_excluded_ref(name: str) -> bool:
    lowered = name.lower()
    return any(k in lowered for k in EXCLUDE_KEYWORDS)


def parse_sort_key(name: str) -> tuple:
    """
    Sort revisions in a sensible training order:
      stage1 before stage2, then by ingredient, then step, then tokens.
    """
    m = PRETRAIN_PATTERNS[0].match(name)
    if m:
        return (0, -1, int(m.group(1)), float(m.group(2)), name)

    m = PRETRAIN_PATTERNS[1].match(name)
    if m:
        return (1, int(m.group(1)), int(m.group(2)), float(m.group(3)), name)

    m = PRETRAIN_PATTERNS[2].match(name)
    if m:
        return (2, -1, int(m.group(1)), float(m.group(2)), name)

    return (99, 99, 99_999_999, 99_999_999.0, name)


def is_pretraining_ref(name: str) -> bool:
    if is_excluded_ref(name):
        return False
    return any(p.match(name) for p in PRETRAIN_PATTERNS)


def list_pretrain_revisions_for_repo(repo_id: str) -> list[str]:
    api = HfApi()
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")

    names = []
    for branch in getattr(refs, "branches", []) or []:
        names.append(branch.name)
    for tag in getattr(refs, "tags", []) or []:
        names.append(tag.name)

    # Deduplicate, preserve order.
    seen = set()
    deduped = []
    for n in names:
        if n not in seen:
            seen.add(n)
            deduped.append(n)

    filtered = [n for n in deduped if is_pretraining_ref(n)]
    filtered.sort(key=parse_sort_key)
    return filtered


def list_all_pretrain_revisions(model_size: str) -> list[dict]:
    """
    Returns a list of:
      [{"repo_id": ..., "revision": ...}, ...]
    in training order.
    """
    items = []
    for repo_id in MODEL_REPOS[model_size]:
        revisions = list_pretrain_revisions_for_repo(repo_id)
        for rev in revisions:
            items.append({"repo_id": repo_id, "revision": rev})

    # Global sort across repos.
    items.sort(key=lambda x: parse_sort_key(x["revision"]))
    return items


def sample_revisions_evenly(
    all_revisions: list[dict],
    keep_first_n: int = 5,
    total_n: int = 25,
) -> list[dict]:
    """
    Keep the first `keep_first_n` revisions, then sample the remaining revisions
    so the final total count is `total_n`.

    If there are <= total_n revisions total, return all of them.
    """
    n = len(all_revisions)
    if n <= total_n:
        return all_revisions

    keep_first_n = max(0, keep_first_n)
    total_n = max(0, total_n)

    if total_n == 0:
        return []

    head = all_revisions[:keep_first_n]
    if len(head) >= total_n:
        return head[:total_n]

    tail = all_revisions[keep_first_n:]
    remaining_to_pick = total_n - len(head)

    if remaining_to_pick <= 0:
        return head

    if len(tail) <= remaining_to_pick:
        sampled_tail = tail
    elif remaining_to_pick == 1:
        sampled_tail = [tail[-1]]
    else:
        indices = [
            round(i * (len(tail) - 1) / (remaining_to_pick - 1))
            for i in range(remaining_to_pick)
        ]

        # Deduplicate in case rounding collides.
        seen = set()
        deduped_indices = []
        for idx in indices:
            if idx not in seen:
                deduped_indices.append(idx)
                seen.add(idx)

        # Fill any missing slots from left to right.
        if len(deduped_indices) < remaining_to_pick:
            for idx in range(len(tail)):
                if idx not in seen:
                    deduped_indices.append(idx)
                    seen.add(idx)
                if len(deduped_indices) == remaining_to_pick:
                    break

        sampled_tail = [tail[idx] for idx in deduped_indices[:remaining_to_pick]]

    return head + sampled_tail


def delete_checkpoint_cache(model_name: str, revision: str):
    try:
        cache_info = scan_cache_dir()
        commits_to_delete = [
            rev.commit_hash
            for repo in cache_info.repos
            if repo.repo_id == model_name
            for rev in repo.revisions
            if revision in rev.refs
        ]
        if commits_to_delete:
            strategy = cache_info.delete_revisions(*commits_to_delete)
            strategy.execute()
            tqdm.write(
                f"  Deleted cache for {model_name} @ {revision} "
                f"(freed {strategy.expected_freed_size_str})"
            )
    except Exception as e:
        tqdm.write(
            f"  Warning: could not delete cache for {model_name} @ {revision}: {e}"
        )


def choose_tensor_parallel_size(model_size: str, user_tp: Optional[int]) -> int:
    if user_tp is not None:
        return user_tp
    defaults = {
        "1b": 1,
        "7b": 1,
        "13b": 2,
        "32b": 4,
    }
    return defaults[model_size]


def choose_gpu_memory_utilization(model_size: str, user_val: Optional[float]) -> float:
    if user_val is not None:
        return user_val
    defaults = {
        "1b": 0.90,
        "7b": 0.90,
        "13b": 0.92,
        "32b": 0.92,
    }
    return defaults[model_size]


class VLLMGenerateAdapter:
    """
    Minimal wrapper to mimic Hugging Face model.generate(...) with vLLM underneath.
    This assumes the benchmark only needs generation, not forward/logits calls.
    """

    def __init__(self, llm: LLM, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.config = type("Config", (), {})()
        self.config.pad_token_id = tokenizer.pad_token_id
        self.config.eos_token_id = tokenizer.eos_token_id

    def eval(self):
        return self

    def _decode_prompts_from_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[str]:
        prompts = []
        input_ids = input_ids.detach().cpu()
        if attention_mask is not None:
            attention_mask = attention_mask.detach().cpu()

        for i in range(input_ids.shape[0]):
            ids = input_ids[i]
            if attention_mask is not None:
                ids = ids[attention_mask[i].bool()]
            text = self.tokenizer.decode(
                ids.tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts.append(text)
        return prompts

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int | list[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        prompts = self._decode_prompts_from_input_ids(input_ids, attention_mask)

        if not do_sample:
            temperature = 0.0
            top_p = 1.0

        stop_token_ids = None
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                stop_token_ids = [eos_token_id]
            else:
                stop_token_ids = list(eos_token_id)

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
        )

        outputs = self.llm.generate(prompts, sampling_params)


        # full_sequences = []
        # generated_ids = []

        # for out in outputs:
        #     text = out.outputs[0].text.strip()
        #     ids = self.tokenizer.encode(text, add_special_tokens=False)
        #     generated_ids.append(ids)

        # pad_id = (
        #     pad_token_id
        #     if pad_token_id is not None
        #     else self.tokenizer.pad_token_id
        # )

        # max_len = max(1, max(len(x) for x in generated_ids))
        # padded = [seq + [pad_id] * (max_len - len(seq)) for seq in generated_ids]

        # return torch.tensor(padded, dtype=torch.long)

        full_sequences = []
        prompt_ids_cpu = input_ids.detach().cpu()

        for i, output in enumerate(outputs):
            prompt_ids = prompt_ids_cpu[i]  # keep full padded input row
            generated_text = output.outputs[0].text
            gen_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
            full_sequences.append(prompt_ids.tolist() + gen_ids)

        effective_pad_id = (
            pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        )
        max_len = max(len(seq) for seq in full_sequences)
        padded = [
            seq + [effective_pad_id] * (max_len - len(seq))
            for seq in full_sequences
        ]
        return torch.tensor(padded, dtype=torch.long)

def load_model(
    model_size: str,
    repo_id: str,
    revision: str,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    trust_remote_code: bool = True,
):
    tqdm.write(f"  Loading tokenizer {repo_id} @ {revision} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )

    # OLMo 2 may not define a pad token.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 2048

    tp = choose_tensor_parallel_size(model_size, tensor_parallel_size)
    gmu = choose_gpu_memory_utilization(model_size, gpu_memory_utilization)

    tqdm.write(
        f"  Loading vLLM engine for {repo_id} @ {revision} "
        f"(tp={tp}, gpu_memory_utilization={gmu}) ..."
    )

    llm = LLM(
        model=repo_id,
        tokenizer=repo_id,
        revision=revision,
        tokenizer_revision=revision,
        trust_remote_code=trust_remote_code,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=gmu,
        max_model_len=2048,
        enable_prefix_caching=True,
    )

    model = VLLMGenerateAdapter(llm, tokenizer)
    tqdm.write("  vLLM engine ready")
    return model, tokenizer


def cleanup_model(model):
    try:
        if hasattr(model, "llm"):
            del model.llm
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(
    model_size: str,
    revisions: list[dict],
    tasks: list[str],
    num_shots: int = 5,
    batch_size: int = 32,
    max_new_tokens: int = 3,
    output_path: Optional[Path] = None,
    resume: bool = False,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
):
    """
    results[revision_key][task] = accuracy
    outputs[revision_key][task] = ...
    timing[revision_key][task] = seconds
    timing[revision_key]["_total"] = seconds

    revision_key format:
        "{repo_id}::{revision}"
    """

    results = {}
    outputs = {}
    timing = {}
    revisions_to_run = revisions

    outputs_path = (
        output_path.parent / output_path.name.replace("results_", "outputs_")
        if output_path else None
    )

    if resume and output_path and output_path.exists():
        with open(output_path) as f:
            saved = json.load(f)
        results = saved.get("results", {})
        timing = saved.get("timing", {})

        if outputs_path and outputs_path.exists():
            with open(outputs_path) as f:
                outputs = json.load(f)

        completed = set(results.keys())
        skipped = []
        remaining = []
        for item in revisions:
            key = f"{item['repo_id']}::{item['revision']}"
            if key in completed:
                skipped.append(key)
            else:
                remaining.append(item)

        revisions_to_run = remaining
        tqdm.write(
            f"Resuming: skipping {len(skipped)} completed revisions, "
            f"{len(revisions_to_run)} remaining."
        )
        if not revisions_to_run:
            tqdm.write("All revisions already completed.")
            return results, timing

    total_start = time.time()

    for item in tqdm(revisions_to_run, desc="Checkpoints", position=0):
        repo_id = item["repo_id"]
        revision = item["revision"]
        revision_key = f"{repo_id}::{revision}"

        tqdm.write(f"\n=== {revision_key} ===")
        ckpt_start = time.time()

        model, tokenizer = load_model(
            model_size=model_size,
            repo_id=repo_id,
            revision=revision,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        results[revision_key] = {}
        outputs[revision_key] = {}
        timing[revision_key] = {}

        for task in tqdm(tasks, desc="Tasks", leave=False, position=1):
            task_start = time.time()
            acc, task_outputs = fv_icl_tasks_benchmark(
                model=model,
                tokenizer=tokenizer,
                task_name=task,
                task_dir=str(TASK_DIR),
                num_of_shots=num_shots,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
            task_elapsed = time.time() - task_start

            results[revision_key][task] = acc
            outputs[revision_key][task] = task_outputs
            timing[revision_key][task] = task_elapsed

            tqdm.write(f"  {task:30s}  acc={acc:.4f}  ({task_elapsed:.1f}s)")

        ckpt_elapsed = time.time() - ckpt_start
        timing[revision_key]["_total"] = ckpt_elapsed
        tqdm.write(f"  revision total: {ckpt_elapsed:.1f}s")
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"results": results, "timing": timing}, f, indent=2)
            with open(outputs_path, "w") as f:
                json.dump(outputs, f, indent=2)
            tqdm.write(f"  Saved to {output_path} and {outputs_path}")

        cleanup_model(model)
        delete_checkpoint_cache(repo_id, revision)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"{'Revision':<60} {'Total(s)':>10}")
    print("-" * 80)
    for item in revisions_to_run:
        revision_key = f"{item['repo_id']}::{item['revision']}"
        if revision_key not in timing:
            continue
        ckpt_total = timing[revision_key]["_total"]
        print(f"{revision_key:<60} {ckpt_total:>10.1f}s")
    print("-" * 80)
    print(f"{'TOTAL':<60} {total_elapsed:>10.1f}s")
    print("=" * 80)

    return results, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval_olmo2_config.json",
        help="Path to JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--model_size",
        default=None,
        choices=["1b", "7b", "13b", "32b"],
        help="Which OLMo 2 model size to evaluate.",
    )
    parser.add_argument(
        "--revisions",
        nargs="+",
        default=None,
        help="Specific revision names to evaluate. Defaults to sampled discovered pretraining revisions.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names to evaluate. Defaults to all abstractive tasks.",
    )
    parser.add_argument("--num_shots", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--sample_total",
        type=int,
        default=None,
        help="Total number of revisions to sample from all available pretraining revisions.",
    )
    parser.add_argument(
        "--sample_keep_first",
        type=int,
        default=None,
        help="Number of initial revisions to always keep before evenly sampling the rest.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to results/results_olmo2_{model_size}.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip revisions already present in the output file.",
    )
    parser.add_argument(
        "--list_only",
        action="store_true",
        help="Only list discovered/sample revisions and exit.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM tensor parallel size. Defaults by model size: 1b/7b->1, 13b->2, 32b->4.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM GPU memory fraction. Defaults by model size.",
    )
    args = parser.parse_args()

    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Loaded config from {config_path}", flush=True)

    model_size = args.model_size or cfg.get("model_size", "1b")
    num_shots = args.num_shots if args.num_shots is not None else cfg.get("num_shots", 5)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.get("max_new_tokens", 3)
    sample_total = args.sample_total if args.sample_total is not None else cfg.get("sample_total", 25)
    sample_keep_first = args.sample_keep_first if args.sample_keep_first is not None else cfg.get("sample_keep_first", 5)
    tasks = args.tasks or cfg.get("tasks") or ALL_TASKS
    output_path = Path(
        args.output
        or cfg.get("output")
        or f"results/results_olmo2_{model_size}_vllm.json"
    )

    all_revisions = list_all_pretrain_revisions(model_size)

    if args.revisions:
        requested = set(args.revisions)
        revisions = [x for x in all_revisions if x["revision"] in requested]
        missing = requested - {x["revision"] for x in revisions}
        if missing:
            print(f"Warning: requested revisions not found: {sorted(missing)}", flush=True)
    else:
        revisions = sample_revisions_evenly(
            all_revisions,
            keep_first_n=sample_keep_first,
            total_n=sample_total,
        )

    print(f"Model size:   OLMo-2 {model_size}", flush=True)
    print(f"Repos:        {MODEL_REPOS[model_size]}", flush=True)
    print(f"All revs:     {len(all_revisions)} total discovered", flush=True)
    print(f"Using revs:   {len(revisions)} total", flush=True)
    print(f"Sampling:     keep first {sample_keep_first}, total {sample_total}", flush=True)
    print(f"Tasks:        {tasks}", flush=True)
    print(f"Task dir:     {TASK_DIR}", flush=True)
    print(f"Output:       {output_path}", flush=True)

    if args.list_only:
        print("\nSelected pretraining revisions:", flush=True)
        for item in revisions:
            print(f"  {item['repo_id']}::{item['revision']}", flush=True)
        return

    evaluate(
        model_size=model_size,
        revisions=revisions,
        tasks=tasks,
        num_shots=num_shots,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        resume=args.resume,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"\nDone. Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"\nTotal elapsed time: {(end - start)/60:.1f} mins")
