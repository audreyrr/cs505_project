import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle


def apply_iclr_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "figure.figsize": (7, 4),
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "legend.columnspacing": 1.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_attention_heads(attention_weights, tokens=None, tokenizer=None):
    if isinstance(attention_weights[0], torch.Tensor):
        attention_weights = [w[0].detach().cpu().numpy() for w in attention_weights]

    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[0]

    layers_to_show = num_layers
    heads_to_show = num_heads

    fig, axes = plt.subplots(layers_to_show, heads_to_show, figsize=(12, 12))
    if layers_to_show == 1:
        axes = axes.reshape(1, -1)

    induction_positions = []
    if tokens is not None and tokenizer is not None:
        token_strings = [tokenizer.decode(i.item()) for i in tokens]
        induction_positions = find_induction_positions(token_strings)

    for layer_idx in range(layers_to_show):
        for head_idx in range(heads_to_show):
            ax = axes[layer_idx, head_idx]
            attn = attention_weights[layer_idx][head_idx]

            ax.imshow(attn, cmap="Blues")

            for query_pos, key_pos in induction_positions:
                rect = Rectangle(
                    (key_pos - 0.5, query_pos - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

            ax.set_title(f"L{layer_idx}H{head_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_attention_head(
    attention_weights, layer_idx, head_idx, tokens=None, tokenizer=None
):
    if isinstance(attention_weights[layer_idx], torch.Tensor):
        attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    else:
        attn = attention_weights[layer_idx][0, head_idx]

    plt.figure(figsize=(8, 8))
    plt.imshow(attn, cmap="Blues")
    plt.colorbar()

    token_strings = None
    if tokens is not None and tokenizer is not None:
        token_strings = [tokenizer.decode(i.item()) for i in tokens]
        plt.xticks(range(len(token_strings)), token_strings, rotation=45, ha="right")
        plt.yticks(range(len(token_strings)), token_strings)

        induction_positions = find_induction_positions(token_strings)

        for query_pos, key_pos in induction_positions:
            rect = Rectangle(
                (key_pos - 0.5, query_pos - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
            )
            plt.gca().add_patch(rect)

    plt.title(f"Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.show()


def find_induction_positions(token_strings):
    induction_positions = []

    for i, token in enumerate(token_strings):
        for j in range(i):
            if token_strings[j] == token:
                if j + 1 < len(token_strings):
                    induction_positions.append((i, j + 1))

    return induction_positions
