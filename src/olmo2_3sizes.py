"""
Evaluate OLMo 2 pretraining checkpoints on abstractive ICL tasks.

Example usage:
    python src/eval_olmo2.py
    python src/eval_olmo2.py --config config/eval_olmo2_config.json
    python src/eval_olmo2.py --model_size 7b
    python src/eval_olmo2.py --model_size 1b --revisions stage1-step1000-tokens2B stage1-step20000-tokens42B
    python src/eval_olmo2.py --tasks antonym country-capital
    python src/eval_olmo2.py --resume

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
from tqdm import tqdm
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.utils import HfHubHTTPError

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
PROJECT_ROOT = Path("/projectnb/cs505am/students/amao/icl-without-copying")
# DATASET_ROOT = PROJECT_ROOT / "dataset"
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
            # print(
            #     f"  Deleted cache for {model_name} @ {revision} "
            #     f"(freed {strategy.expected_freed_size_str})",
            #     flush=True,
            # )

            tqdm.write(
            f"  Deleted cache for {model_name} @ {revision} "
            f"(freed {strategy.expected_freed_size_str})"
         )
    except Exception as e:
        # print(
        #     f"  Warning: could not delete cache for {model_name} @ {revision}: {e}",
        #     flush=True,
        # )
        tqdm.write(
        f"  Warning: could not delete cache for {model_name} @ {revision}: {e}"
        )

def load_model(repo_id: str, revision: str):
    # print(f"  Loading {repo_id} @ {revision} ...", flush=True)
    tqdm.write(f"  Loading {repo_id} @ {revision} ...")

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)

    # OLMo 2 may not define a pad token.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 2048

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Only resize if we actually added a new token.
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"  Using device: {device}", flush=True)
    tqdm.write(f"  Using device: {device}")
    return model.to(device), tokenizer


def evaluate(
    model_size: str,
    revisions: list[dict],
    tasks: list[str],
    num_shots: int = 5,
    batch_size: int = 32,
    max_new_tokens: int = 3,
    output_path: Optional[Path] = None,
    resume: bool = False,
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

        # revisions = remaining
        # revisions = remaining
        # revisions_to_run = revisions
        revisions_to_run = remaining
        # print(
        #     f"Resuming: skipping {len(skipped)} completed revisions, "
        #     f"{len(revisions)} remaining.",
        #     flush=True,
        # )
        # tqdm.write(
        # f"Resuming: skipping {len(skipped)} completed revisions, "
        # f"{len(revisions)} remaining."
        # )
        tqdm.write(
        f"Resuming: skipping {len(skipped)} completed revisions, "
        f"{len(revisions_to_run)} remaining."
        )
        # if not revisions:
        if not revisions_to_run:
            # print("All revisions already completed.", flush=True)
            tqdm.write("All revisions already completed.")
            return results, timing

    total_start = time.time()

    # for item in revisions:
    # for item in tqdm(revisions, desc="Checkpoints", position=0):
    for item in tqdm(revisions_to_run, desc="Checkpoints", position=0):
        repo_id = item["repo_id"]
        revision = item["revision"]
        revision_key = f"{repo_id}::{revision}"

        # print(f"\n=== {revision_key} ===", flush=True)
        tqdm.write(f"\n=== {revision_key} ===")
        ckpt_start = time.time()

        model, tokenizer = load_model(repo_id, revision)
        results[revision_key] = {}
        outputs[revision_key] = {}
        timing[revision_key] = {}

        # for task in tasks:
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

            # print(f"  {task:30s}  acc={acc:.4f}  ({task_elapsed:.1f}s)", flush=True)
            tqdm.write(f"  {task:30s}  acc={acc:.4f}  ({task_elapsed:.1f}s)")

        ckpt_elapsed = time.time() - ckpt_start
        timing[revision_key]["_total"] = ckpt_elapsed
        # print(f"  revision total: {ckpt_elapsed:.1f}s", flush=True)
        tqdm.write(f"  revision total: {ckpt_elapsed:.1f}s")
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"results": results, "timing": timing}, f, indent=2)
            with open(outputs_path, "w") as f:
                json.dump(outputs, f, indent=2)
            # print(f"  Saved to {output_path} and {outputs_path}", flush=True)
            tqdm.write(f"  Saved to {output_path} and {outputs_path}")

        del model
        torch.cuda.empty_cache()
        delete_checkpoint_cache(repo_id, revision)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"{'Revision':<60} {'Total(s)':>10}")
    print("-" * 80)
    # for item in revisions:
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
        choices=["1b", "7b", "13b"],
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
    args = parser.parse_args()

    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Loaded config from {config_path}", flush=True)

    model_size = args.model_size or cfg.get("model_size", "1b")
    # num_shots = args.num_shots or cfg.get("num_shots", 5)
    # batch_size = args.batch_size or cfg.get("batch_size", 32)
    # max_new_tokens = args.max_new_tokens or cfg.get("max_new_tokens", 3)
    # sample_total = args.sample_total or cfg.get("sample_total", 25)
    # sample_keep_first = args.sample_keep_first or cfg.get("sample_keep_first", 5)
    num_shots = args.num_shots if args.num_shots is not None else cfg.get("num_shots", 5)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.get("max_new_tokens", 3)
    sample_total = args.sample_total if args.sample_total is not None else cfg.get("sample_total", 25)
    sample_keep_first = args.sample_keep_first if args.sample_keep_first is not None else cfg.get("sample_keep_first", 5)
    tasks = args.tasks or cfg.get("tasks") or ALL_TASKS
    output_path = Path(
        args.output
        or cfg.get("output")
        or f"results/results_olmo2_{model_size}.json"
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
    )

    print(f"\nDone. Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"\nTotal elapsed time: {(end - start)/60:.1f} mins")