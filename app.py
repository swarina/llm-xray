"""
app.py - a browser UI for watching a real GPT-2 think.

Same five stages as llm_xray.py, but interactive: type a prompt, drag the layer /
head / temperature sliders, and every panel updates from the ACTUAL pretrained
model. Nothing is faked.

Run:
    ./.venv/bin/python app.py
then open the http://127.0.0.1:7860 link it prints.
"""

import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import gradio as gr

from core import XRayModel

xm = XRayModel()
tok, model = xm.tok, xm.model
N_LAYER, N_HEAD = xm.n_layer, xm.n_head


def short_labels(id_list):
    return xm.short_labels(id_list, width=8)


def analyze(text, layer, head, temperature, topk):
    layer, head, topk = int(layer), int(head), int(topk)
    if not text.strip():
        text = "the cat sat on the"
    enc = tok(text, return_tensors="pt")
    ids = enc.input_ids[:, -xm.max_positions:]  # stay within the context window
    id_list = ids[0].tolist()
    pieces = tok.convert_ids_to_tokens(id_list)
    labels = short_labels(id_list)

    with torch.no_grad():
        out = model(ids)

    # --- Stage 0: tokens ---
    rows = "".join(
        f"<tr><td>{i}</td><td><code>{tid}</code></td>"
        f"<td><code>{html.escape(p)}</code></td>"
        f"<td>{html.escape(repr(tok.decode([tid])))}</td></tr>"
        for i, (tid, p) in enumerate(zip(id_list, pieces))
    )
    tokens_html = (
        f"<p><b>{len(id_list)} tokens</b> &middot; vocab size {model.config.vocab_size:,} "
        f"&middot; embedding dim {model.config.n_embd}</p>"
        "<table><tr><th>#</th><th>id</th><th>piece</th><th>decoded</th></tr>"
        f"{rows}</table>"
        "<p style='color:#888'>GPT-2 uses byte-level BPE; a leading space shows as "
        "the glyph &lsquo;Ġ&rsquo;.</p>"
    )

    # --- Stage 2: attention ---
    attn = out.attentions[layer][0, head].numpy()
    seq = attn.shape[0]
    fa, axa = plt.subplots(figsize=(5, 4.2))
    im = axa.imshow(attn, cmap="Blues", vmin=0, vmax=1)
    axa.set_xticks(range(seq)); axa.set_xticklabels(labels, rotation=45, ha="right")
    axa.set_yticks(range(seq)); axa.set_yticklabels(labels)
    axa.set_title(f"attention - layer {layer}, head {head}")
    axa.set_xlabel("attended-to (key)"); axa.set_ylabel("attending (query)")
    fa.colorbar(im, fraction=0.046, pad=0.04)
    fa.tight_layout()

    # --- Stage 3: residual stream ---
    norms = [float(hs[0, -1].norm()) for hs in out.hidden_states]
    fr, axr = plt.subplots(figsize=(5, 4.2))
    axr.plot(range(len(norms)), norms, marker="o", color="#185FA5")
    axr.set_title("residual stream - last token vector norm")
    axr.set_xlabel("after block (0 = embedding)"); axr.set_ylabel("L2 norm")
    axr.grid(alpha=0.3)
    fr.tight_layout()

    # --- Stage 4: next-token probabilities (real softmax at this temperature) ---
    logits = out.logits[0, -1]
    probs = F.softmax(logits / temperature, dim=-1)
    top = torch.topk(probs, topk)
    cand = [tok.decode([t]) for t in top.indices.tolist()]
    vals = [v * 100 for v in top.values.tolist()]
    fp, axp = plt.subplots(figsize=(5, 4.2))
    axp.barh(range(len(cand))[::-1], vals, color="#185FA5")
    axp.set_yticks(range(len(cand))[::-1]); axp.set_yticklabels([repr(c) for c in cand])
    axp.set_xlabel("probability (%)")
    axp.set_title(f"next token - temperature {temperature:.1f}")
    fp.tight_layout()

    # one real sample + greedy continuation
    sampled = torch.multinomial(probs, 1).item()
    with torch.no_grad():
        gen = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=10, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    summary = (
        f"**sampled at T={temperature:.1f}:** {text!r} + {tok.decode([sampled])!r}\n\n"
        f"**greedy continuation:** {tok.decode(gen[0], skip_special_tokens=True)!r}"
    )

    return tokens_html, fa, fr, fp, summary


with gr.Blocks(title="GPT-2 x-ray") as demo:
    gr.Markdown(
        "# GPT-2 x-ray\n"
        "Watch a **real** GPT-2 process your text, stage by stage. Every panel is "
        "the actual model's internals - drag the sliders and they all update."
    )
    with gr.Row():
        text = gr.Textbox(value="the cat sat on the", label="prompt", scale=3)
        run = gr.Button("analyze", variant="primary", scale=1)
    with gr.Row():
        layer = gr.Slider(0, N_LAYER - 1, value=N_LAYER - 1, step=1, label="attention layer")
        head = gr.Slider(0, N_HEAD - 1, value=0, step=1, label="attention head")
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="temperature")
        topk = gr.Slider(3, 15, value=8, step=1, label="top-k candidates")

    gr.Markdown("### stage 0 - tokenization")
    tokens_out = gr.HTML()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### stage 2 - attention  *(causal mask = blank upper-right)*")
            attn_out = gr.Plot()
        with gr.Column():
            gr.Markdown("### stage 3 - residual stream  *(blocks add, never replace)*")
            resid_out = gr.Plot()
    gr.Markdown("### stage 4 - next token  *(real logits -> softmax at your temperature)*")
    probs_out = gr.Plot()
    summary_out = gr.Markdown()

    inputs = [text, layer, head, temperature, topk]
    outputs = [tokens_out, attn_out, resid_out, probs_out, summary_out]
    run.click(analyze, inputs, outputs)
    for c in (layer, head, temperature, topk):
        c.release(analyze, inputs, outputs)
    demo.load(analyze, inputs, outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
