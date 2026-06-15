"""
live.py — watch GPT-2 generate one token at a time.

Open http://127.0.0.1:7861 and press ⚡ Generate. At every step six panels
update live: a plain-English narration, the growing sentence, candidate words,
attention weights, a confidence meter, and a running decision log. All numbers
come from the real pretrained model.

Run:
    ./.venv/bin/python live.py
"""

import time

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from panels import (
    attention_panel,
    candidates_panel,
    confidence_panel,
    done_narration,
    log_panel,
    narration_panel,
    text_panel,
)
from styles import PAGE_HEADER, TOGGLE_JS

MODEL = "gpt2"
print(f"loading {MODEL} (cached after first run)…")
tok   = GPT2TokenizerFast.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL, output_attentions=True)
model.eval()
N_LAYER = model.config.n_layer


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _short_labels(id_list: list) -> list:
    out = []
    for t in id_list:
        s = tok.decode([t]).strip()[:7]
        out.append(s if s else "_")
    return out


# ── Streaming generator ──────────────────────────────────────────────────────────

def stream(prompt, temperature, max_new, delay, do_sample, layer):
    layer  = int(layer)
    prompt = prompt.strip() or "the meaning of life is"
    enc    = tok(prompt, return_tensors="pt")
    ids    = enc.input_ids
    plen   = ids.shape[1]
    total  = int(max_new)
    log    = []

    for step in range(1, total + 1):
        with torch.no_grad():
            out = model(ids)

        logits = out.logits[0, -1]
        probs  = F.softmax(logits / temperature, dim=-1)

        attn_last = out.attentions[layer][0].mean(0)[-1]
        labels  = _short_labels(ids[0].tolist())
        weights = attn_last.tolist()
        top_attn = labels[weights.index(max(weights))]

        top  = torch.topk(probs, 8)
        idxs = top.indices.tolist()
        cands      = [tok.decode([t]) for t in idxs]
        cand_probs = top.values.tolist()

        nxt = torch.multinomial(probs, 1).item() if do_sample else idxs[0]
        if nxt in idxs:
            chosen_idx = idxs.index(nxt)
        else:
            cands.append(tok.decode([nxt]))
            cand_probs.append(float(probs[nxt]))
            chosen_idx = len(cands) - 1

        entropy  = float(-(probs * probs.clamp_min(1e-12).log2()).sum())
        top_prob = float(probs.max())
        chosen   = tok.decode([nxt])

        prompt_text = tok.decode(ids[0, :plen].tolist())
        gen_text    = tok.decode(ids[0, plen:].tolist()) if ids.shape[1] > plen else ""

        yield (
            narration_panel(step, total, chosen, top_prob, top_attn),
            text_panel(prompt_text, gen_text, chosen, False),
            candidates_panel(cands, cand_probs, chosen_idx),
            attention_panel(labels[-20:], weights[-20:], layer),
            confidence_panel(top_prob, entropy),
            log_panel(log),
        )
        time.sleep(float(delay))

        log.append((step, chosen, top_prob))
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if nxt == tok.eos_token_id:
            break

    prompt_text = tok.decode(ids[0, :plen].tolist())
    gen_text    = tok.decode(ids[0, plen:].tolist())
    yield (
        done_narration(),
        text_panel(prompt_text, gen_text, "", True),
        gr.update(),
        gr.update(),
        gr.update(),
        log_panel(log),
    )


# ── Gradio layout ────────────────────────────────────────────────────────────────

with gr.Blocks(title="LLM X-Ray", theme=gr.themes.Base(), js=TOGGLE_JS) as demo:

    gr.HTML(PAGE_HEADER)

    with gr.Row():
        prompt = gr.Textbox(
            value="the meaning of life is",
            label="Start a sentence",
            placeholder="type any sentence fragment…",
            scale=4,
        )
        run  = gr.Button("⚡ Generate", variant="primary", scale=1, min_width=120)
        stop = gr.Button("■ Stop",      variant="stop",    scale=1, min_width=100)

    with gr.Row():
        temperature = gr.Slider(
            0.1, 2.0, value=0.8, step=0.1,
            label="temperature  —  low = safe & predictable · high = wild & creative",
        )
        max_new = gr.Slider(
            5, 50, value=20, step=1,
            label="words to generate",
        )
    with gr.Row():
        delay = gr.Slider(
            0.05, 2.0, value=0.65, step=0.05,
            label="seconds per word  (drag left to speed up)",
        )
        layer = gr.Slider(
            0, N_LAYER - 1, value=N_LAYER - 1, step=1,
            label="attention layer to inspect  (0 = earliest · 11 = final)",
        )

    do_sample = gr.Checkbox(
        value=True,
        label="sample randomly  —  uncheck to always pick the top word (fully deterministic)",
    )

    narration_out = gr.HTML()
    text_out      = gr.HTML()

    with gr.Row(equal_height=True):
        cand_out = gr.HTML()
        attn_out = gr.HTML()

    conf_out = gr.HTML()
    log_out  = gr.HTML()

    ev = run.click(
        stream,
        inputs=[prompt, temperature, max_new, delay, do_sample, layer],
        outputs=[narration_out, text_out, cand_out, attn_out, conf_out, log_out],
    )
    stop.click(None, None, None, cancels=[ev])


if __name__ == "__main__":
    demo.queue().launch(server_port=7861)
