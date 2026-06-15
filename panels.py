"""
panels.py — HTML panel generators for the LLM X-Ray live UI.

Every function returns a self-contained HTML string that uses the CSS variables
defined in styles.py. No hardcoded colours — all colour references go through
var(--xr-*) so the dark/light toggle propagates automatically.
"""

import html as _html


def esc(s: str) -> str:
    return _html.escape(str(s))


def _conf(p: float) -> tuple:
    """Map a probability to (color-var, bg-var, label)."""
    if p > 0.6:
        return "var(--xr-green)", "var(--xr-green-bg)", "very confident"
    if p > 0.3:
        return "var(--xr-amber)", "var(--xr-amber-bg)", "somewhat unsure"
    return "var(--xr-red)", "var(--xr-red-bg)", "very unsure"


# ── Step narration ──────────────────────────────────────────────────────────────

def narration_panel(step: int, total: int, chosen: str, top_prob: float, top_attn: str) -> str:
    color, bg, label = _conf(top_prob)
    pct = int(top_prob * 100)
    word = chosen.strip() or chosen
    return f"""
<div class="xr-card" style="border-left:4px solid {color};background:{bg};border-color:{color}">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
    <div>
      <div class="xr-eyebrow">step {step} of {total}</div>
      <div style="font-size:17px;font-weight:700;color:{color};margin-bottom:6px">{pct}% confident — {label}</div>
      <div style="font-size:15px;color:var(--xr-text);line-height:1.7">
        Paid most attention to
        <strong>&lsquo;{esc(top_attn)}&rsquo;</strong>,
        then decided to add&nbsp;
        <span style="background:#fef08a;color:#713f12;border-radius:5px;
                     padding:2px 9px;font-weight:700;font-size:16px">{esc(word)}</span>
      </div>
    </div>
    <div style="flex-shrink:0;text-align:right;min-width:90px">
      <div style="font-size:32px;font-weight:800;color:{color};line-height:1">{pct}%</div>
      <div style="background:rgba(0,0,0,.10);border-radius:6px;height:8px;
                  overflow:hidden;margin-top:6px;width:90px">
        <div style="width:{pct}%;height:100%;background:{color};
                    border-radius:6px;transition:width .35s ease"></div>
      </div>
    </div>
  </div>
</div>"""


def done_narration() -> str:
    return """
<div class="xr-card" style="border-left:4px solid var(--xr-green);background:var(--xr-green-bg);
                             border-color:var(--xr-green)">
  <div style="font-size:17px;font-weight:700;color:var(--xr-green);margin-bottom:6px">
    ✓ Generation complete
  </div>
  <div style="font-size:14px;color:var(--xr-mute);line-height:1.65">
    The loop above repeated for every word to build the full sentence.
    Change the prompt or temperature and run again.
  </div>
</div>"""


# ── Growing sentence ────────────────────────────────────────────────────────────

def text_panel(prompt_text: str, gen_text: str, incoming: str, done: bool) -> str:
    cursor = "" if done else \
        "<span style='color:var(--xr-blue);animation:xr-blink 1s step-end infinite'>▌</span>"
    inc = ""
    if incoming and not done:
        inc = (f"<span style='background:#fef08a;color:#713f12;border-radius:5px;"
               f"padding:2px 7px;font-weight:700'>{esc(incoming)}</span>")
    return f"""
<div class="xr-card" style="border-left:4px solid var(--xr-amber)">
  <div class="xr-eyebrow">the sentence</div>
  <div class="xr-heading">Growing live</div>
  <div class="xr-sub">
    <span style="color:var(--xr-hint)">■</span> gray = your prompt &nbsp;·&nbsp;
    black = generated &nbsp;·&nbsp;
    <span style="background:#fef08a;color:#713f12;border-radius:3px;padding:0 5px">yellow</span>
    = being added now
  </div>
  <hr class="xr-hr">
  <div class="xr-sentence">
    <span style="color:var(--xr-hint)">{esc(prompt_text)}</span>{esc(gen_text)}{inc}{cursor}
  </div>
</div>"""


# ── Candidate words ─────────────────────────────────────────────────────────────

def candidates_panel(cands: list, probs: list, chosen_idx: int) -> str:
    rows = []
    for i, (c, p) in enumerate(zip(cands, probs)):
        picked = i == chosen_idx
        label  = c.strip() or c
        fill   = "var(--xr-green)" if picked else "var(--xr-blue)"
        lcolor = "var(--xr-green)" if picked else "var(--xr-mute)"
        weight = "700" if picked else "400"
        badge  = ("&nbsp;<span style='color:var(--xr-green);font-size:11px;font-weight:600'>"
                  "✓ picked</span>") if picked else ""
        rows.append(f"""
<div class="xr-bar-row">
  <div class="xr-bar-label" style="color:{lcolor};font-weight:{weight}">{esc(label)}</div>
  <div class="xr-bar-track">
    <div class="xr-bar-fill" style="width:{max(p*100, 1):.1f}%;background:{fill}"></div>
  </div>
  <div class="xr-bar-pct">{p*100:4.1f}%{badge}</div>
</div>""")
    return f"""
<div class="xr-card" style="border-left:4px solid var(--xr-green)">
  <div class="xr-eyebrow">candidates</div>
  <div class="xr-heading">Which word comes next?</div>
  <div class="xr-sub">
    Top options out of 50,257 words. Green&nbsp;✓ = what it chose.
    Drag temperature up to flatten the bars; down to spike them.
  </div>
  <hr class="xr-hr">
  {"".join(rows)}
</div>"""


# ── Attention chips ─────────────────────────────────────────────────────────────

def attention_panel(labels: list, weights: list, layer: int) -> str:
    if not weights:
        return ""
    mx  = max(weights)
    top = labels[weights.index(mx)]
    chips = []
    for lab, w in zip(labels, weights):
        rel   = (w / mx) if mx > 0 else 0
        alpha = max(0.07, rel * 0.88)
        dark  = rel > 0.55
        text  = "var(--xr-surface)" if dark else "var(--xr-text)"
        chips.append(
            f"<div class='xr-chip' title='weight: {w:.3f}' style='"
            f"background:rgba(59,130,246,{alpha:.2f});"
            f"color:{text};"
            f"border-color:rgba(59,130,246,{min(alpha+.15,1):.2f})'>"
            f"{esc(lab)}<br><strong>{w:.2f}</strong></div>"
        )
    return f"""
<div class="xr-card" style="border-left:4px solid var(--xr-blue)">
  <div class="xr-eyebrow">attention &mdash; layer {layer}</div>
  <div class="xr-heading">What did it look at?</div>
  <div class="xr-sub">
    Darker blue = looked at it more. Strongest: <strong>&lsquo;{esc(top)}&rsquo;</strong>.
    Hover any chip for the exact weight.
  </div>
  <hr class="xr-hr">
  <div style="display:flex;gap:6px;flex-wrap:wrap">{"".join(chips)}</div>
</div>"""


# ── Confidence & entropy ────────────────────────────────────────────────────────

def confidence_panel(top_prob: float, entropy_bits: float) -> str:
    color, _, label = _conf(top_prob)
    pct        = int(top_prob * 100)
    max_bits   = 15.9                         # log2(50257)
    entr_pct   = min(entropy_bits / max_bits * 100, 100)
    return f"""
<div class="xr-card" style="border-left:4px solid {color}">
  <div class="xr-eyebrow">confidence &amp; uncertainty</div>
  <div class="xr-heading">How sure is it this step?</div>
  <hr class="xr-hr">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">

    <div>
      <div style="font-size:13px;color:var(--xr-mute);margin-bottom:6px">
        Confidence in top choice</div>
      <div style="font-size:34px;font-weight:800;color:{color};line-height:1">{pct}%</div>
      <div style="font-size:12.5px;color:{color};margin:3px 0 8px">{label}</div>
      <div class="xr-meter-track">
        <div style="width:{pct}%;height:100%;background:{color};
                    border-radius:8px;transition:width .35s ease"></div>
      </div>
    </div>

    <div>
      <div style="font-size:13px;color:var(--xr-mute);margin-bottom:6px">
        Uncertainty (entropy)</div>
      <div style="font-size:34px;font-weight:800;color:var(--xr-text);line-height:1">
        {entropy_bits:.1f}</div>
      <div style="font-size:12.5px;color:var(--xr-mute);margin:3px 0 8px">
        bits &mdash; 0 = totally certain, {max_bits:.0f} = random</div>
      <div class="xr-meter-track">
        <div style="width:{entr_pct:.0f}%;height:100%;background:var(--xr-blue);
                    border-radius:8px;transition:width .35s ease"></div>
      </div>
    </div>

  </div>
</div>"""


# ── Decision log ────────────────────────────────────────────────────────────────

def log_panel(entries: list) -> str:
    if not entries:
        inner = ("<div style='color:var(--xr-hint);font-size:13px;padding:8px 0'>"
                 "Each word the model commits will appear here&hellip;</div>")
    else:
        rows = []
        for step, word, p in entries:
            color, _, _ = _conf(p)
            label = word.strip() or word
            rows.append(f"""
<div class="xr-log-row">
  <span class="xr-log-step">#{step}</span>
  <span class="xr-log-word">{esc(label)}</span>
  <div class="xr-log-bar">
    <div style="height:100%;width:{p*100:.0f}%;background:{color};border-radius:4px"></div>
  </div>
  <span class="xr-log-pct">{p*100:.0f}%</span>
</div>""")
        inner = "".join(rows)
    return f"""
<div class="xr-card">
  <div class="xr-eyebrow">decision log</div>
  <div class="xr-heading">Every word it committed</div>
  <div class="xr-sub">
    Step, word chosen, confidence bar.
    Green = confident · amber = unsure · red = genuinely guessing.
  </div>
  <hr class="xr-hr">
  {inner}
</div>"""
