"""
Regenerate the example figures embedded in the README.

    ./.venv/bin/python docs/make_figures.py

Saves fig-attention.png, fig-temperature.png and fig-residual.png next to this
file. Every figure is a real output of pretrained GPT-2.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True, output_hidden_states=True)
model.eval()


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def labels(ids):
    return [tok.decode([t]).strip()[:8] or "_" for t in ids]


def fig_attention():
    ids = tok("the cat sat on the mat", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids)
    attn = out.attentions[11][0, 0].numpy()
    n = attn.shape[0]
    labs = labels(ids[0].tolist())
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labs)
    ax.set_title("attention - layer 11, head 0")
    ax.set_xlabel("attended-to (key)"); ax.set_ylabel("attending (query)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save(fig, "fig-attention.png")


def fig_temperature():
    ids = tok("the capital of france is", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    idx = torch.topk(F.softmax(logits, dim=-1), 8).indices
    words = [repr(tok.decode([i])) for i in idx.tolist()]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, T in zip(axes, (0.3, 0.8, 1.5)):
        probs = F.softmax(logits / T, dim=-1)[idx] * 100
        ax.barh(range(8)[::-1], probs.tolist(), color="#185FA5")
        ax.set_title(f"temperature {T}")
        ax.set_xlabel("probability (%)")
    axes[0].set_yticks(range(8)[::-1]); axes[0].set_yticklabels(words)
    fig.suptitle("same logits, three temperatures: low = peaked, high = flat", y=1.02)
    save(fig, "fig-temperature.png")


def fig_residual():
    ids = tok("the cat sat on the mat", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids)
    norms = [float(hs[0, -1].norm()) for hs in out.hidden_states]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(norms)), norms, marker="o", color="#185FA5")
    ax.set_title("residual stream - last token vector norm grows each block")
    ax.set_xlabel("after block (0 = embedding)"); ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)
    save(fig, "fig-residual.png")


if __name__ == "__main__":
    fig_attention()
    fig_temperature()
    fig_residual()
    print("done")
