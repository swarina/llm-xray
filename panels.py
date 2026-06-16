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
    return _panel("01", "NEXT&nbsp;TOKEN", "top 8 of 50,257", "".join(rows))


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
    return _panel("02", "ATTENTION", f"layer {layer} · strongest &lsquo;{esc(top)}&rsquo;", inner)


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
    return _panel("03", "CONFIDENCE", "how peaked is the choice", inner)


# ── 04 · Trace (decision log) ───────────────────────────────────────────────────

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
    return _panel("04", "TRACE", "every committed word", inner)
