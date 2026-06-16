"""
live.py — watch GPT-2 generate one token at a time.

Open http://127.0.0.1:7861 and press Generate. Six readouts update on every
step: a status line, the growing sentence (hero), next-token candidates,
attention, a confidence meter, and a running trace. Every number is from the
real pretrained model.

Run:
    ./.venv/bin/python live.py
"""

import time

import gradio as gr
import torch
import torch.nn.functional as F

from core import XRayModel
from panels import (
    attention_panel,
    candidates_panel,
    confidence_panel,
    done_narration,
    embedding_panel,
    ffn_panel,
    heads_panel,
    log_panel,
    logit_lens_panel,
    narration_panel,
    text_panel,
)
from styles import PAGE_HEADER, TOGGLE_JS

xm = XRayModel()
N_LAYER = xm.n_layer


def stream(prompt, temperature, max_new, delay, do_sample, layer):
    layer = int(layer)
    prompt = prompt.strip() or "the meaning of life is"
    ids = xm.encode(prompt)
    plen = ids.shape[1]
    total = int(max_new)
    log = []

    for step in range(1, total + 1):
        out, ffn_act = xm.forward_capturing(ids, layer)

        logits = out.logits[0, -1]
        probs = F.softmax(logits / temperature, dim=-1)

        labels = xm.short_labels(ids[0].tolist())

        # 00 input embedding (last token): token + position vectors, first 48 dims
        tok_vec, pos_vec = xm.input_embedding(ids)
        emb = embedding_panel(tok_vec[:48].tolist(), pos_vec[:48].tolist())

        # 01 attention summary (mean over heads) + 02 per-head detail
        attn_layer = out.attentions[layer][0]            # [heads, seq, seq]
        attn_mean = attn_layer.mean(0)[-1]
        weights = attn_mean.tolist()
        top_attn = labels[weights.index(max(weights))]
        hlabels = labels[-14:]
        per_head = [(hh, attn_layer[hh, -1].tolist()[-14:])
                    for hh in range(attn_layer.shape[0])]

        # 03 feed-forward neurons (post-GELU activations, last token)
        nvals, nidx = torch.topk(ffn_act, 8)
        top_neurons = list(zip(nidx.tolist(), nvals.tolist()))
        n_active = int((ffn_act > 1.0).sum())

        # 04 logit lens
        lens = xm.logit_lens(out.hidden_states)

        # 05 next-token candidates
        top = torch.topk(probs, 8)
        idxs = top.indices.tolist()
        cands = [xm.decode([t]) for t in idxs]
        cand_probs = top.values.tolist()

        nxt = torch.multinomial(probs, 1).item() if do_sample else idxs[0]
        if nxt in idxs:
            chosen_idx = idxs.index(nxt)
        else:
            cands.append(xm.decode([nxt]))
            cand_probs.append(float(probs[nxt]))
            chosen_idx = len(cands) - 1

        entropy = float(-(probs * probs.clamp_min(1e-12).log2()).sum())
        top_prob = float(probs.max())
        chosen = xm.decode([nxt])

        prompt_text = xm.decode(ids[0, :plen].tolist())
        gen_text = xm.decode(ids[0, plen:].tolist()) if ids.shape[1] > plen else ""

        yield (
            narration_panel(step, total, chosen, top_prob, top_attn),
            text_panel(prompt_text, gen_text, chosen, False),
            emb,
            attention_panel(labels[-20:], weights[-20:], layer),
            heads_panel(per_head, hlabels, layer),
            ffn_panel(top_neurons, n_active, ffn_act.shape[0], layer),
            logit_lens_panel(lens, chosen),
            candidates_panel(cands, cand_probs, chosen_idx),
            confidence_panel(top_prob, entropy),
            log_panel(log),
        )
        time.sleep(float(delay))

        log.append((step, chosen, top_prob))
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if nxt == xm.eos_id:
            break

    prompt_text = xm.decode(ids[0, :plen].tolist())
    gen_text = xm.decode(ids[0, plen:].tolist())
    yield (
        done_narration(),
        text_panel(prompt_text, gen_text, "", True),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(),
        log_panel(log),
    )


# ── Theme: IBM Plex type + paper/ink palette, explicit in BOTH modes so Gradio's
#    own chrome tracks our panels instead of fighting them. ────────────────────────
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F2EEE6",
    body_background_fill_dark="#0E1116",
    body_text_color="#1A1916",
    body_text_color_dark="#ECEAE3",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#161A21",
    block_border_color="#E5E0D5",
    block_border_color_dark="#272D38",
    block_label_text_color="#6E6A62",
    block_label_text_color_dark="#989284",
    block_title_text_color="#1A1916",
    block_title_text_color_dark="#ECEAE3",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#161A21",
    input_border_color="#E5E0D5",
    input_border_color_dark="#272D38",
    border_color_primary="#E5E0D5",
    border_color_primary_dark="#272D38",
    button_primary_background_fill="#2C53D9",
    button_primary_background_fill_dark="#3B63E8",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
)

CSS = """
.gradio-container { max-width: 940px !important; margin: 0 auto !important;
  -webkit-font-smoothing: antialiased; }
footer { display: none !important; }
"""

with gr.Blocks(title="LLM X-Ray", theme=THEME, js=TOGGLE_JS, css=CSS) as demo:

    gr.HTML(PAGE_HEADER)

    with gr.Row():
        prompt = gr.Textbox(
            value="the meaning of life is",
            label="prompt",
            placeholder="type any sentence fragment…",
            scale=4,
            container=True,
        )
        run = gr.Button("Generate", variant="primary", scale=1, min_width=110)
        stop = gr.Button("Stop", variant="secondary", scale=1, min_width=90)

    with gr.Row():
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1,
                                label="temperature  ·  low = safe   high = wild")
        max_new = gr.Slider(5, 50, value=20, step=1, label="words to generate")
    with gr.Row():
        delay = gr.Slider(0.05, 2.0, value=0.6, step=0.05,
                          label="seconds per word  ·  drag left to speed up")
        layer = gr.Slider(0, N_LAYER - 1, value=N_LAYER - 1, step=1,
                          label="attention layer  ·  0 early → 11 final")
    do_sample = gr.Checkbox(value=True,
                            label="sample randomly  ·  uncheck to always take the top word")

    # always visible (beginner spine)
    status_out = gr.HTML()
    text_out = gr.HTML()

    # intermediate+ : the mechanism opens up
    gr.HTML('<div class="xr-divider">the machine, this step — one full forward pass, in order</div>',
            elem_classes="xr-lvl-inter")
    with gr.Row(equal_height=True, elem_classes="xr-lvl-inter"):
        emb_out = gr.HTML()
        attn_out = gr.HTML()
    # expert only : deep internals
    with gr.Row(equal_height=True, elem_classes="xr-lvl-expert"):
        heads_out = gr.HTML()
        ffn_out = gr.HTML()
    lens_out = gr.HTML(elem_classes="xr-lvl-inter")
    # beginner : the outcome
    with gr.Row(equal_height=True):
        cand_out = gr.HTML()
        conf_out = gr.HTML()
    log_out = gr.HTML()

    ev = run.click(
        stream,
        inputs=[prompt, temperature, max_new, delay, do_sample, layer],
        outputs=[status_out, text_out, emb_out, attn_out, heads_out, ffn_out,
                 lens_out, cand_out, conf_out, log_out],
    )
    stop.click(None, None, None, cancels=[ev])


if __name__ == "__main__":
    demo.queue().launch(server_port=7861)
