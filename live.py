"""
live.py - watch GPT-2 generate one token at a time, explained for beginners.

Press "generate" and the real model builds a sentence word by word. Every step is
narrated in plain English and broken into panels: the sentence so far, the words
it's choosing between, which earlier words it looked at, how sure it is, and a
running log of every decision. All numbers come from the real pretrained model.

Run:
    ./.venv/bin/python live.py
then open http://127.0.0.1:7861 and press "generate".
"""

import html
import time

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL = "gpt2"
print(f"loading {MODEL} (cached after first run)...")
tok = GPT2TokenizerFast.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL, output_attentions=True)
model.eval()
N_LAYER = model.config.n_layer

INK = "#111827"
MUTE = "#6b7280"
FAINT = "#9ca3af"
TRACK = "#eef1f4"
GREEN = "#16a34a"
BLUE = "#2563eb"
AMBER = "#d97706"
RED = "#dc2626"


def esc(s):
    return html.escape(s)


def chip(s):
    return html.escape(s).replace(" ", "&nbsp;")


def confidence_style(p):
    if p > 0.6:
        return GREEN, "#e7f6ec", "very confident"
    if p > 0.3:
        return AMBER, "#fdf2e2", "somewhat unsure"
    return RED, "#fdecec", "very unsure - lots of options"


def card(title, subtitle, inner, accent=BLUE):
    return (
        f"<div style='background:#ffffff;border:1px solid #e5e7eb;border-left:4px solid {accent};"
        "border-radius:12px;padding:15px 18px;box-sizing:border-box'>"
        f"<div style='font-size:15px;font-weight:600;color:{INK}'>{title}</div>"
        f"<div style='font-size:12.5px;color:{MUTE};margin:3px 0 13px'>{subtitle}</div>"
        f"{inner}</div>"
    )


INTRO = (
    "<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:15px 18px;"
    f"color:{INK};font-size:14px;line-height:1.7'>"
    "<b>How a language model writes:</b> it adds one word at a time. To pick each word it "
    "(1)&nbsp;scores every word in its vocabulary, (2)&nbsp;looks back at the words so far to "
    "decide, then (3)&nbsp;chooses one, adds it, and repeats. Press <b>generate</b> below and "
    "watch all three happen, slowed down so you can follow."
    "<div style='margin-top:10px;font-size:13px;color:#374151'>"
    "<b>Colors:</b> "
    "<span style='background:#fde68a;padding:1px 6px;border-radius:4px'>yellow</span>&nbsp;= word being added &nbsp;&middot;&nbsp; "
    "<span style='color:#15803d;font-weight:600'>green</span>&nbsp;= the word it picked &nbsp;&middot;&nbsp; "
    "<span style='color:#1d4ed8;font-weight:600'>blue</span>&nbsp;= attention (what it looked at)"
    "</div></div>"
)


def narration_panel(step, total, chosen, top_prob, top_attn):
    color, bg, word = confidence_style(top_prob)
    return (
        f"<div style='background:{bg};border:1px solid {color};border-radius:12px;padding:16px 18px;"
        f"font-size:16px;color:{INK};line-height:1.6'>"
        f"<b>Step {step} of {total}.</b> The model re-read the sentence, paid most attention to "
        f"<b>&lsquo;{esc(top_attn)}&rsquo;</b>, and is <b style='color:{color}'>{top_prob*100:.0f}% sure</b> "
        f"({word}). It's adding the word "
        f"<span style='background:#fde68a;border-radius:4px;padding:2px 7px;font-weight:600'>{chip(chosen)}</span>"
        "</div>"
    )


def text_panel(prompt_text, gen_text, incoming, done):
    cursor = "" if done else "<span style='color:#2563eb;font-weight:700'>&#9612;</span>"
    inc = ""
    if incoming and not done:
        inc = (f"<span style='background:#fde68a;border-radius:4px;padding:2px 4px;color:#92400e;"
               f"font-weight:600'>{chip(incoming)}</span>")
    body = (
        "<div style='font-size:19px;line-height:2;font-family:Georgia,serif'>"
        f"<span style='color:{FAINT}'>{esc(prompt_text)}</span>"
        f"<span style='color:{INK}'>{esc(gen_text)}</span>{inc}{cursor}</div>"
    )
    sub = "Gray = your prompt &nbsp;&middot;&nbsp; black = what the model wrote &nbsp;&middot;&nbsp; yellow = being added now"
    return card("1 &middot; The sentence, growing live", sub, body, accent=AMBER)


def candidates_panel(cands, probs, chosen_idx):
    rows = []
    for i, (c, p) in enumerate(zip(cands, probs)):
        picked = i == chosen_idx
        disp = c.strip() or c
        bar = GREEN if picked else "#bcd3f7"
        lab_color = GREEN if picked else "#4b5563"
        badge = (f"<span style='color:{GREEN};font-weight:600'>&nbsp;&#10003; picked</span>"
                 if picked else "")
        rows.append(
            "<div style='display:flex;align-items:center;gap:10px;margin:5px 0'>"
            f"<div style='width:84px;text-align:right;font-family:monospace;font-size:14px;"
            f"font-weight:{600 if picked else 400};color:{lab_color}'>{esc(disp)}</div>"
            f"<div style='flex:1;background:{TRACK};border-radius:5px;height:18px'>"
            f"<div style='width:{max(p*100,1.5):.1f}%;background:{bar};height:100%;border-radius:5px'></div></div>"
            f"<div style='width:104px;font-size:13px;color:{MUTE}'>{p*100:4.1f}%{badge}</div>"
            "</div>"
        )
    sub = ("It scores all 50,257 words; here are the top few. Taller green bar = the one it chose. "
           "Higher temperature flattens these bars.")
    return card("2 &middot; Which word comes next?", sub, "".join(rows), accent=GREEN)


def attention_panel(labels, weights, layer):
    mx = max(weights) if weights else 1.0
    top = labels[weights.index(mx)] if weights else "-"
    cells = []
    for lab, w in zip(labels, weights):
        op = max(0.05, min(w * 1.15, 1.0))
        dark = w > 0.45
        cells.append(
            f"<div title='{w:.2f}' style='min-width:50px;padding:7px 5px;text-align:center;"
            f"border:1px solid #e5e7eb;border-radius:6px;background:rgba(37,99,235,{op:.2f});"
            f"color:{'#ffffff' if dark else '#1f2937'};font-size:11.5px'>"
            f"{esc(lab)}<br><b>{w:.2f}</b></div>"
        )
    sub = (f"Darker blue = looked at it more (layer {layer}). Strongest here: "
           f"<b>&lsquo;{esc(top)}&rsquo;</b>. This is what the model 're-read' to decide.")
    inner = "<div style='display:flex;gap:6px;flex-wrap:wrap'>" + "".join(cells) + "</div>"
    return card("3 &middot; Which earlier words did it look at?", sub, inner, accent=BLUE)


def confidence_panel(top_prob, entropy_bits):
    color, _, word = confidence_style(top_prob)
    inner = (
        f"<div style='font-size:15px;color:{INK};margin-bottom:8px'>It is "
        f"<b style='color:{color}'>{top_prob*100:.0f}% sure</b> of its top choice &nbsp;({word})</div>"
        f"<div style='background:{TRACK};border-radius:6px;height:14px;width:100%'>"
        f"<div style='width:{top_prob*100:.0f}%;background:{color};height:100%;border-radius:6px'></div></div>"
        f"<div style='font-size:12.5px;color:{MUTE};margin-top:8px'>"
        f"uncertainty: <b>{entropy_bits:.1f} bits</b> &nbsp;(0 = certain, higher = more options in play)</div>"
    )
    return card("How sure is it this step?", "One tall bar above = confident. Many short bars = unsure.",
                inner, accent=color)


def log_panel(entries):
    if not entries:
        inner = f"<div style='color:{FAINT};font-size:13px'>each word it commits will appear here&hellip;</div>"
    else:
        rows = []
        for step, word, p in entries:
            color, _, _ = confidence_style(p)
            rows.append(
                "<div style='display:flex;align-items:center;gap:10px;font-size:13px;padding:4px 0;"
                "border-bottom:1px solid #f3f4f6'>"
                f"<span style='width:30px;color:{FAINT}'>#{step}</span>"
                f"<span style='flex:1;font-family:monospace;color:{INK}'>&hellip; &rarr; "
                f"<b>{esc(word.strip() or word)}</b></span>"
                f"<span style='width:120px;background:{TRACK};border-radius:4px;height:9px'>"
                f"<span style='display:block;height:100%;width:{p*100:.0f}%;background:{color};border-radius:4px'></span></span>"
                f"<span style='width:42px;text-align:right;color:{MUTE}'>{p*100:.0f}%</span></div>"
            )
        inner = "".join(rows)
    return card("4 &middot; Decision log", "Every word it committed, with how sure it was.", inner, accent="#6b7280")


def short_labels(id_list):
    out = []
    for t in id_list:
        s = tok.decode([t]).strip()[:7]
        out.append(s if s else "_")
    return out


def stream(prompt, temperature, max_new, delay, do_sample, layer):
    layer = int(layer)
    prompt = prompt.strip() or "the meaning of life is"
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids
    prompt_len = ids.shape[1]
    total = int(max_new)
    log = []

    for step in range(1, total + 1):
        with torch.no_grad():
            out = model(ids)
        logits = out.logits[0, -1]
        probs = F.softmax(logits / temperature, dim=-1)

        attn_last = out.attentions[layer][0].mean(0)[-1]
        labels = short_labels(ids[0].tolist())
        weights = attn_last.tolist()
        top_attn = labels[weights.index(max(weights))]

        top = torch.topk(probs, 8)
        idxs = top.indices.tolist()
        cand = [tok.decode([t]) for t in idxs]
        cand_probs = top.values.tolist()

        nxt = torch.multinomial(probs, 1).item() if do_sample else idxs[0]
        if nxt in idxs:
            chosen_idx = idxs.index(nxt)
        else:
            cand.append(tok.decode([nxt]))
            cand_probs.append(float(probs[nxt]))
            chosen_idx = len(cand) - 1

        entropy = float(-(probs * probs.clamp_min(1e-12).log2()).sum())
        top_prob = float(probs.max())
        chosen = tok.decode([nxt])

        prompt_text = tok.decode(ids[0, :prompt_len].tolist())
        gen_text = tok.decode(ids[0, prompt_len:].tolist()) if ids.shape[1] > prompt_len else ""

        yield (
            narration_panel(step, total, chosen, top_prob, top_attn),
            text_panel(prompt_text, gen_text, chosen, False),
            candidates_panel(cand, cand_probs, chosen_idx),
            attention_panel(labels[-22:], weights[-22:], layer),
            confidence_panel(top_prob, entropy),
            log_panel(log),
        )
        time.sleep(float(delay))

        log.append((step, chosen, top_prob))
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if nxt == tok.eos_token_id:
            break

    prompt_text = tok.decode(ids[0, :prompt_len].tolist())
    gen_text = tok.decode(ids[0, prompt_len:].tolist())
    final = (
        f"<div style='background:#e7f6ec;border:1px solid {GREEN};border-radius:12px;padding:16px 18px;"
        f"font-size:16px;color:{INK}'>&#10003; <b>Done.</b> The loop above repeated for every word to "
        "build the whole sentence. Change the temperature or prompt and run it again.</div>"
    )
    yield (
        final,
        text_panel(prompt_text, gen_text, "", True),
        gr.update(),
        gr.update(),
        gr.update(),
        log_panel(log),
    )


with gr.Blocks(title="GPT-2 live", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Watch GPT-2 write, one word at a time")
    gr.HTML(INTRO)

    with gr.Row():
        prompt = gr.Textbox(value="the meaning of life is", label="Start a sentence", scale=3)
        run = gr.Button("generate", variant="primary", scale=1)
        stop = gr.Button("stop", scale=1)
    with gr.Row():
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1,
                                label="temperature (low = safe, high = wild)")
        max_new = gr.Slider(5, 40, value=20, step=1, label="how many words")
        delay = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="seconds per word (watch speed)")
        layer = gr.Slider(0, N_LAYER - 1, value=N_LAYER - 1, step=1, label="attention layer")
    do_sample = gr.Checkbox(value=True, label="roll the dice (uncheck = always pick the top word)")

    narration_out = gr.HTML()
    text_out = gr.HTML()
    with gr.Row():
        cand_out = gr.HTML()
        attn_out = gr.HTML()
    conf_out = gr.HTML()
    log_out = gr.HTML()

    ev = run.click(
        stream,
        [prompt, temperature, max_new, delay, do_sample, layer],
        [narration_out, text_out, cand_out, attn_out, conf_out, log_out],
    )
    stop.click(None, None, None, cancels=[ev])


if __name__ == "__main__":
    demo.queue().launch(server_port=7861)
