import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nnsight import LanguageModel
from typing import List, Tuple
from transformers import GPTNeoXForCausalLM
import os


def previous_token_mask(input_ids, model):
    B, S = input_ids.shape
    device = input_ids.device

    causal = torch.tril(torch.zeros(S, S, device=device, dtype=model.dtype))
    causal = causal + torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=model.dtype), diagonal=1
    )
    masks = causal.unsqueeze(0).expand(B, S, S).clone()

    row = torch.arange(1, S, device=device)
    col = row - 1
    masks[:, row, col] = float("-inf")

    return masks.unsqueeze(1)


def visualize_head_scores(scores, threshold=None, title="Head Scores", fmt=".3f"):
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    else:
        scores = np.array(scores)

    num_layers, num_heads = scores.shape

    plt.figure(figsize=(12, 8))

    ax = sns.heatmap(
        scores,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        cbar_kws={"label": "Score"},
        vmin=0,
        vmax=1,
        xticklabels=[f"Head {i}" for i in range(num_heads)],
        yticklabels=[f"Layer {i}" for i in range(num_layers)],
    )

    if threshold is not None:
        for i in range(num_layers):
            for j in range(num_heads):
                if scores[i, j] > threshold:
                    ax.add_patch(
                        plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=3)
                    )

        plt.title(f"{title}\n(Red boxes indicate scores > {threshold:.3f})")
    else:
        plt.title(title)

    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.show()


def visualize_head_scores_publication(
    scores,
    threshold=None,
    title="Head Scores",
    fmt=".2f",
    save_dir="plot_attention_figures",
    model_type="none",
    color_map="Purples",
    vmin=0,
    vmax=1,
    layer_start=0,
    score_names=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    else:
        scores = np.array(scores)

    num_layers, num_heads = scores.shape

    scores = scores[layer_start:, :].T

    plt.figure(figsize=(scores.shape[1] / 2, num_heads / 2))
    sns.set(font_scale=1.2, style="white")

    ax = sns.heatmap(
        scores,
        cmap=color_map,
        square=True,
        cbar_kws={"label": score_names},
        vmin=vmin,
        vmax=vmax,
        xticklabels=[f"L{i}" for i in range(layer_start, num_layers)],
        yticklabels=[f"H{i}" for i in range(num_heads)],
    )

    if threshold is not None:
        for i in range(num_heads):
            for j in range(scores.shape[1]):
                if scores[i, j] > threshold:
                    ax.scatter(j + 0.5, i + 0.5, marker=".", color="red", s=60)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Head Index")
    ax.set_title(title, fontsize=14, pad=12)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        f"{save_dir}/{model_type}_head_scores.pdf", dpi=600, bbox_inches="tight"
    )
    plt.savefig(
        f"{save_dir}/{model_type}_head_scores.png", dpi=600, bbox_inches="tight"
    )
    plt.close()
    plt.show()


def compute_verbatim_induction_scores(
    model,
    tokenizer,
    num_of_samples=100,
    seq_len=25,
    batch_size=16,
    device=None,
    precomputed_sequences=None,
):
    if device is None:
        device = next(model.parameters()).device

    original_attn_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"

    induction_scores = torch.zeros(
        model.config.num_hidden_layers, model.config.num_attention_heads
    ).to(device)

    if precomputed_sequences is None:
        vocab_size = tokenizer.vocab_size
        random_sequence = torch.randint(1, vocab_size, (num_of_samples, seq_len))
        random_repetitive_sequence = torch.cat(
            [random_sequence, random_sequence], dim=1
        )
    else:
        random_repetitive_sequence = precomputed_sequences

    model.eval()
    with torch.no_grad():
        for i in range(0, num_of_samples, batch_size):
            begin_index = i
            end_index = min(i + batch_size, num_of_samples)
            batch = random_repetitive_sequence[begin_index:end_index, :]
            input_data = {"input_ids": batch.to(device)}
            result = model(**input_data, output_attentions=True)
            for layer in range(model.config.num_hidden_layers):
                layer_values = result.attentions[layer]
                curr_ind_scores = (
                    layer_values.diagonal(offset=-seq_len + 1, dim1=-2, dim2=-1)[
                        ..., 1:
                    ]
                    .mean(dim=-1)
                    .sum(dim=0)
                )
                induction_scores[layer] += curr_ind_scores

    induction_scores /= num_of_samples

    model.config._attn_implementation = original_attn_impl

    return induction_scores


def ablate_heads(model, heads_to_ablate):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_size = hidden_size // num_heads
    head_qkv_size = head_size * 3

    for layer_idx, head_idx in heads_to_ablate:
        layer = model.gpt_neox.layers[layer_idx]
        attn = layer.attention
        with torch.no_grad():
            head_start = head_idx * head_qkv_size
            head_end = (head_idx + 1) * head_qkv_size

            head_start_dense = head_idx * head_size
            head_end_dense = (head_idx + 1) * head_size

            attn.query_key_value.weight[head_start:head_end, :].zero_()
            attn.query_key_value.bias[head_start:head_end].zero_()
            attn.dense.weight[:, head_start_dense:head_end_dense].zero_()
            attn.dense.bias[head_start_dense:head_end_dense].zero_()
