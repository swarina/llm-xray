"""
panels.py — HTML panel generators for the LLM X-Ray live UI.

Each function returns a self-contained HTML string built on the classes and CSS
variables defined in styles.py. No hardcoded colours — everything resolves
through var(--xr-*) so the theme switch propagates with zero duplication.

Colour grammar:
  cobalt  (--xr-accent) → data / attention / "where it looked"
  marigold(--xr-live)   → the word it is choosing right now
  cobalt / marigold / clay encode confidence (near-certain / leaning / wide open)
"""

import html as _html


def esc(s) -> str:
    return _html.escape(str(s))


def _conf(p: float):
    """Probability → (css-colour-var, one-word label). Reuses brand colours."""
    if p > 0.6:
        return "var(--xr-accent)", "near-certain"
    if p > 0.3:
        return "var(--xr-live)", "leaning"
    return "var(--xr-warn)", "wide open"


def _panel(idx, label, note, inner) -> str:
    return f"""
<section class="xr-panel">
  <header class="xr-phead">
    <span class="xr-idx">{idx}</span>
    <span class="xr-plabel">{label}</span>
    <span class="xr-pnote">{note}</span>
  </header>
  {inner}
</section>"""


def _strip(vals, scale):
    """A row of diverging-coloured cells for a slice of a vector."""
    cells = []
    for v in vals:
        t = max(-1.0, min(1.0, v / scale)) if scale else 0.0
        c = "var(--xr-accent)" if t >= 0 else "var(--xr-warn)"
        p = int(max(6, abs(t) * 100))
        cells.append(f'<span class="xr-cell" style="background:color-mix(in srgb,{c} {p}%,transparent)"></span>')
    return "".join(cells)


# ── 00 · Input embedding ────────────────────────────────────────────────────────

def embedding_panel(tok_vals, pos_vals) -> str:
    scale = max(1e-6, max((abs(x) for x in tok_vals + pos_vals), default=1.0))
    inner = (
        f'<div class="xr-strip-lab">token embedding (wte) &mdash; 768 dims, first {len(tok_vals)} shown</div>'
        f'<div class="xr-strip">{_strip(tok_vals, scale)}</div>'
        '<div class="xr-strip-lab" style="margin-top:13px">+ position embedding (wpe)</div>'
        f'<div class="xr-strip">{_strip(pos_vals, scale)}</div>'
        '<div class="xr-foot">The token id picks a row of the embedding table; GPT-2 adds a learned '
        'position vector. Their sum is the 768-number vector that enters block&nbsp;0. '
        '<i>Blue = positive, clay = negative.</i></div>'
    )
    return _panel("00", "INPUT", "id &rarr; vector", inner)


# ── 02 · Attention heads ────────────────────────────────────────────────────────

def heads_panel(per_head, labels, layer) -> str:
    boxes = []
    for hh, weights in per_head:
        hm = max(weights) or 1.0
        cells = []
        for lab, w in zip(labels, weights):
            p = int(max(6, (w / hm) * 100))
            cells.append(
                f'<span class="xr-mini" title="{esc(lab)}: {w:.2f}" '
                f'style="background:color-mix(in srgb,var(--xr-accent) {p}%,transparent)"></span>'
            )
        boxes.append(
            f'<div class="xr-head-box"><div class="xr-head-id">H{hh:02d}</div>'
            f'<div class="xr-head-strip">{"".join(cells)}</div></div>'
        )
    foot = ('<div class="xr-foot">Attention runs in 12 parallel heads, each with its own Q/K/V '
            'projection. Each strip is one head\'s attention over recent tokens (its own scale) '
            '&mdash; they <i>specialise</i>: some fixate on the first token, others spread. '
            'The summary above averages all twelve.</div>')
    inner = f'<div class="xr-heads-grid">{"".join(boxes)}</div>{foot}'
    return _panel("02", "HEADS", f"layer {layer} · 12 heads", inner)


# ── 03 · Feed-forward network ───────────────────────────────────────────────────

def ffn_panel(top_neurons, n_active, total, layer) -> str:
    mx = max((a for _, a in top_neurons), default=1.0) or 1.0
    rows = []
    for idx, a in top_neurons:
        rows.append(
            f'<div class="xr-neuron"><span class="xr-neuron-id">neuron&nbsp;{idx}</span>'
            f'<span class="xr-neuron-bar"><i style="width:{max(a/mx*100, 2):.0f}%"></i></span>'
            f'<span class="xr-neuron-val">{a:.1f}</span></div>'
        )
    flow = (
        '<div class="xr-ffn-flow">'
        '<span class="xr-ffn-dim">768</span><span class="xr-ffn-arrow">expand &rarr;</span>'
        '<span class="xr-ffn-dim">3072</span><span class="xr-ffn-arrow">GELU, contract &rarr;</span>'
        '<span class="xr-ffn-dim">768</span></div>'
        f'<div class="xr-ffn-meta"><b>{n_active}</b> of {total:,} neurons firing strongly this step '
        '&mdash; top activations:</div>'
    )
    foot = ("<div class=\"xr-foot\">The feed-forward network holds most of GPT-2's parameters. "
            "It expands each vector, applies GELU, then contracts it &mdash; specific neurons fire "
            "for specific patterns and facts (where edit methods like ROME act).</div>")
    return _panel("03", "FEED-FORWARD", f"layer {layer} · MLP", flow + "".join(rows) + foot)


# ── Status strip (per-step narration) ───────────────────────────────────────────

def narration_panel(step, total, chosen, top_prob, top_attn) -> str:
    color, word = _conf(top_prob)
    pct = int(round(top_prob * 100))
    tok = chosen.strip() or chosen
    return f"""
<div class="xr-status">
  <span class="xr-status-step">STEP {step:02d} / {total}</span>
  <span class="xr-status-sep"></span>
  <span class="xr-status-text">
    <b style="color:{color}">{pct}% {word}</b> &nbsp;·&nbsp;
    attended to <b>&lsquo;{esc(top_attn)}&rsquo;</b> &nbsp;·&nbsp;
    adding <span class="xr-now">{esc(tok)}</span>
  </span>
</div>"""


def done_narration() -> str:
    return """
<div class="xr-status">
  <span class="xr-status-step">DONE</span>
  <span class="xr-status-sep"></span>
  <span class="xr-status-text">
    Reached the end of the run. Adjust the controls and generate again.
  </span>
</div>"""


# ── Hero: the growing sentence ──────────────────────────────────────────────────

def text_panel(prompt_text, gen_text, incoming, done) -> str:
    caret = "" if done else '<span class="xr-caret"></span>'
    now = (f'<span class="xr-now">{esc(incoming)}</span>'
           if (incoming and not done) else "")
    meta = "complete" if done else "writing"
    return f"""
<div class="xr-hero">
  <div class="xr-hero-top">
    <span class="xr-hero-label">OUTPUT</span>
    <span class="xr-hero-meta">{meta}</span>
  </div>
  <p class="xr-hero-text"><span class="xr-hero-prompt">{esc(prompt_text)}</span>{esc(gen_text)}{now}{caret}</p>
</div>"""


# ── 01 · Next-token candidates ──────────────────────────────────────────────────

def candidates_panel(cands, probs, chosen_idx) -> str:
    rows = []
    for i, (c, p) in enumerate(zip(cands, probs)):
        picked = i == chosen_idx
        lab = c.strip() or c
        fill = "var(--xr-live)" if picked else "var(--xr-accent)"
        op = "1" if picked else ".42"
        tok_cls = "xr-tok picked" if picked else "xr-tok"
        mark = '<span class="xr-pick">picked</span>' if picked else ""
        rows.append(f"""
<div class="xr-row">
  <span class="{tok_cls}">{esc(lab)}</span>
  <span class="xr-meter"><i style="width:{max(p*100, 1):.1f}%;background:{fill};opacity:{op}"></i></span>
  <span class="xr-val">{p*100:.1f}<small>%</small>{mark}</span>
</div>""")
    return _panel("05", "NEXT&nbsp;TOKEN", "top 8 of 50,257", "".join(rows))


# ── 02 · Attention ──────────────────────────────────────────────────────────────

def attention_panel(labels, weights, layer) -> str:
    if not weights:
        return ""
    mx = max(weights)
    top = labels[weights.index(mx)]
    chips = []
    for lab, w in zip(labels, weights):
        rel = (w / mx) if mx > 0 else 0
        pct = int(max(6, rel * 90))
        text = "var(--xr-surface)" if rel > 0.55 else "var(--xr-text)"
        chips.append(
            f'<span class="xr-chip" title="weight {w:.3f}" '
            f'style="background:color-mix(in srgb, var(--xr-accent) {pct}%, transparent);color:{text}">'
            f"{esc(lab)}<i>{w:.2f}</i></span>"
        )
    inner = (
        f'<div class="xr-chips">{"".join(chips)}</div>'
        '<div class="xr-foot">High weight shows where it <i>looked</i>, not proof of what '
        'mattered &mdash; the first token often acts as an attention &lsquo;sink&rsquo;. '
        'This is averaged over all 12 heads; individual heads specialise far more.</div>'
    )
    return _panel("01", "ATTENTION", f"layer {layer} · strongest &lsquo;{esc(top)}&rsquo;", inner)


# ── 03 · Confidence ─────────────────────────────────────────────────────────────

def confidence_panel(top_prob, entropy_bits) -> str:
    color, word = _conf(top_prob)
    pct = int(round(top_prob * 100))
    max_bits = 15.62  # log2(50257) — the entropy of a uniform draw over the vocab
    entr_pct = min(entropy_bits / max_bits * 100, 100)
    inner = f"""
<div class="xr-conf">
  <div>
    <div class="xr-conf-num" style="color:{color}">{pct}<small>%</small></div>
    <div class="xr-conf-word" style="color:{color}">{word}</div>
    <div class="xr-track"><i style="width:{pct}%;background:{color}"></i></div>
  </div>
  <div class="xr-conf-side">
    <div class="xr-conf-k">entropy</div>
    <div class="xr-conf-v">{entropy_bits:.1f}<small>bits</small></div>
    <div class="xr-track"><i style="width:{entr_pct:.0f}%;background:var(--xr-accent);opacity:.5"></i></div>
    <div class="xr-conf-hint">0 = certain &nbsp;·&nbsp; {max_bits:.0f} = pure guess</div>
  </div>
</div>"""
    return _panel("06", "CONFIDENCE", "how peaked is the choice", inner)


# ── 04 · Trace (decision log) ───────────────────────────────────────────────────

def logit_lens_panel(rows, final_token) -> str:
    """rows: list of (layer_index, token_str, prob). Read top→bottom: the guess
    forming with depth and locking onto the final word."""
    final = (final_token or "").strip()
    out = []
    locked = False
    for li, word, p in rows:
        lab = word.strip() or word
        match = (lab == final) and final != ""
        if match:
            locked = True
        tok_color = "var(--xr-live)" if (match and locked) else "var(--xr-muted)"
        weight = "600" if match else "400"
        bar = "var(--xr-live)" if match else "var(--xr-accent)"
        out.append(f"""
<div class="xr-lens-row">
  <span class="xr-lens-blk">blk {li:02d}</span>
  <span class="xr-lens-tok" style="color:{tok_color};font-weight:{weight}">{esc(lab)}</span>
  <span class="xr-lens-bar"><i style="width:{p*100:.0f}%;background:{bar};opacity:{'1' if match else '.5'}"></i></span>
  <span class="xr-lens-val">{p*100:.0f}%</span>
</div>""")
    foot = ('<div class="xr-foot">Each layer makes a provisional guess (its residual read through the '
            'final norm + unembedding). Watch it <i>settle</i> with depth &mdash; '
            'marigold marks where it locks onto the word it ships.</div>')
    return _panel("04", "LOGIT&nbsp;LENS", "what each layer is guessing", "".join(out) + foot)


def log_panel(entries) -> str:
    if not entries:
        inner = '<div class="xr-empty">committed words stream here as it writes…</div>'
    else:
        rows = []
        for step, word, p in entries:
            color, _ = _conf(p)
            lab = word.strip() or word
            rows.append(f"""
<div class="xr-trow">
  <span class="xr-tstep">{step:02d}</span>
  <span class="xr-tword">{esc(lab)}</span>
  <span class="xr-tbar"><i style="width:{p*100:.0f}%;background:{color}"></i></span>
  <span class="xr-tval">{p*100:.0f}%</span>
</div>""")
        inner = "".join(rows)
    return _panel("07", "TRACE", "every committed word", inner)
