import torch
import wandb

import torch
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, Literal


def random_sequence_repetition_accuracy(
    model,
    tokenizer,
    num_of_samples=5000,
    seq_len=50,
    batch_size=64,
):

    model.eval()

    vocab_size = tokenizer.vocab_size
    random_sequence = torch.stack(
        [torch.randperm(vocab_size - 1)[:seq_len] + 1 for _ in range(num_of_samples)]
    )
    random_repetitive_sequence = torch.cat([random_sequence, random_sequence], dim=1)

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i in range(0, num_of_samples, batch_size):
            begin_index = i
            end_index = min(i + batch_size, num_of_samples)
            batch = random_repetitive_sequence[begin_index:end_index, :]

            input_ids = batch[:, :-1].to(model.device)
            target_ids = batch[:, -1].to(model.device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            predicted_ids = torch.argmax(logits, dim=-1)

            correct_predictions += (predicted_ids == target_ids).sum().item()
            total_predictions += target_ids.size(0)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return accuracy


def random_sequence_repetition_accuracy_with_ci(
    model,
    tokenizer,
    num_of_samples=5000,
    seq_len=50,
    batch_size=64,
    confidence_level=0.95,
) -> Dict[str, float]:
    accuracy = random_sequence_repetition_accuracy(
        model, tokenizer, num_of_samples, seq_len, batch_size
    )

    z = stats.norm.ppf((1 + confidence_level) / 2)
    n = num_of_samples

    se = np.sqrt(accuracy * (1 - accuracy) / n)

    margin = z * se

    ci_lower = max(0.0, accuracy - margin)
    ci_upper = min(1.0, accuracy + margin)

    return {
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": se,
        "n_samples": n,
    }

def natural_text_accuracy(
    model,
    tokenizer,
    dataset_name="wikitext",
    num_of_samples=5000,
    seq_len=50,
    batch_size=64,
    device=None,
    dataset=None,
):
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Please install 'datasets' library to use this function: pip install datasets"
        )
        return None

    if device is None:
        device = next(model.parameters()).device

    text_column = "text"
    if dataset is None:
        print(f"Loading dataset {dataset_name}")
        if dataset_name == "wikitext":
            dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="train",
            )
        elif dataset_name == "openwebtext":
            dataset = load_dataset(
                "openwebtext",
                split="train",
                streaming=True,
            )
        elif dataset_name == "gutenberg":
            dataset = load_dataset(
                "gutenberg",
                split="en",
            )

    sequences = []
    samples_collected = 0

    for item in dataset:
        if samples_collected >= num_of_samples:
            break

        text = item[text_column]
        if not text or len(text.strip()) == 0:
            continue

        tokens = tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).squeeze()

        if len(tokens) < seq_len:
            continue

        for start_idx in range(0, len(tokens) - seq_len + 1, seq_len):
            if samples_collected >= num_of_samples:
                break
            sequences.append(tokens[start_idx : start_idx + seq_len])
            samples_collected += 1

    sequences = torch.stack(sequences).to(device)

    correct_predictions = 0
    total_predictions = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            begin_index = i
            end_index = min(i + batch_size, len(sequences))
            batch = sequences[begin_index:end_index, :]
            input_ids = batch[:, :-1].to(model.device)
            target_ids = batch[:, -1].to(model.device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            predicted_ids = torch.argmax(logits, dim=-1)

            correct_predictions += (predicted_ids == target_ids).sum().item()
            total_predictions += target_ids.size(0)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return accuracy


def natural_text_repetition_accuracy(
    model,
    tokenizer,
    dataset_name="wikitext",
    num_of_samples=5000,
    seq_len=50,
    batch_size=64,
    device=None,
    use_wandb=True,
    return_dataset_size=False,
    dataset=None,
):
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Please install 'datasets' library to use this function: pip install datasets"
        )
        return None

    if device is None:
        device = next(model.parameters()).device

    text_column = "text"
    if dataset is None:
        print(f"Loading dataset {dataset_name}")
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        elif dataset_name == "openwebtext":
            dataset = load_dataset("openwebtext", split="train", streaming=True)
        elif dataset_name == "gutenberg":
            dataset = load_dataset("gutenberg", split="en")

    sequences = []
    samples_collected = 0

    for item in dataset:
        if samples_collected >= num_of_samples:
            break

        text = item[text_column]
        if not text or len(text.strip()) == 0:
            continue

        tokens = tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).squeeze()

        if len(tokens) < seq_len:
            continue

        for start_idx in range(0, len(tokens) - seq_len + 1, seq_len):
            if samples_collected >= num_of_samples:
                break
            sequences.append(tokens[start_idx : start_idx + seq_len])
            samples_collected += 1

    if use_wandb:
        wandb.summary[f"samples_for_{dataset_name}_repetition"] = samples_collected

    sequences = torch.stack(sequences).to(device)

    repetitive_sequences = torch.cat([sequences, sequences], dim=1)

    correct_predictions = 0
    total_predictions = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            begin_index = i
            end_index = min(i + batch_size, len(sequences))
            batch = repetitive_sequences[begin_index:end_index, :]
            input_ids = batch[:, :-1].to(model.device)
            target_ids = batch[:, -1].to(model.device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            predicted_ids = torch.argmax(logits, dim=-1)

            correct_predictions += (predicted_ids == target_ids).sum().item()
            total_predictions += target_ids.size(0)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    if return_dataset_size:
        return accuracy, total_predictions
    else:
        return accuracy


def natural_text_repetition_accuracy_with_ci(
    model,
    tokenizer,
    dataset_name="wikitext",
    num_of_samples=5000,
    seq_len=50,
    batch_size=64,
    confidence_level=0.95,
    dataset=None,
) -> Dict[str, float]:
    accuracy, n = natural_text_repetition_accuracy(
        model,
        tokenizer,
        dataset_name,
        num_of_samples,
        seq_len,
        batch_size,
        use_wandb=False,
        return_dataset_size=True,
        dataset=dataset,
    )

    z = stats.norm.ppf((1 + confidence_level) / 2)

    se = np.sqrt(accuracy * (1 - accuracy) / n)

    margin = z * se

    ci_lower = max(0.0, accuracy - margin)
    ci_upper = min(1.0, accuracy + margin)

    return {
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": se,
        "n_samples": n,
    }
