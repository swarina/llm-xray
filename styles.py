"""
styles.py — visual system for LLM X-Ray.

Design language: a "reading instrument." IBM Plex type (Mono for data/labels,
Sans for prose, Serif for the generated sentence), a warm paper/ink neutral
palette, one cool accent (cobalt) for data/attention and one warm accent
(marigold) that always means "the word being chosen now."

Theming is driven by a single `data-xr` attribute on <html> plus Gradio's own
`.dark` class — TOGGLE_JS keeps both in sync so the chrome and our panels never
disagree (the bug that made light-mode text vanish on dark-system machines).
"""

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&display=swap');

:root {
  --xr-page:#F1ECE1; --xr-surface:#FDFCF9; --xr-raised:#F6F2EA;
  --xr-text:#1A1815; --xr-muted:#6B675E; --xr-faint:#A7A195;
  --xr-line:#E3DDD0; --xr-line-strong:#D6CFBF;
  --xr-accent:#2A4DB8; --xr-accent-weak:#E7ECF8; --xr-on-accent:#FFFFFF;
  --xr-focus:rgba(42,77,184,.28);
  --xr-live:#BC6E27; --xr-live-weak:#F5E7D2; --xr-live-ink:#5B2F0E;
  --xr-warn:#AC463A;
  --xr-mono:'IBM Plex Mono',ui-monospace,monospace;
  --xr-sans:'IBM Plex Sans',ui-sans-serif,system-ui,sans-serif;
  --xr-serif:'IBM Plex Serif',Georgia,serif;
}
html[data-xr="dark"] {
  --xr-page:#0D1014; --xr-surface:#161A20; --xr-raised:#1C222C;
  --xr-text:#ECE9E2; --xr-muted:#969084; --xr-faint:#5C5950;
  --xr-line:#262C36; --xr-line-strong:#333B47;
  --xr-accent:#7E9CFF; --xr-accent-weak:#18213A; --xr-on-accent:#0D1014;
  --xr-focus:rgba(126,156,255,.32);
  --xr-live:#E3A350; --xr-live-weak:#2A2012; --xr-live-ink:#F4D7A4;
  --xr-warn:#E0796B;
}

/* ── Header ─────────────────────────────────────────────── */
.xr-head { display:flex; align-items:flex-start; justify-content:space-between;
  padding:4px 2px 16px; border-bottom:1px solid var(--xr-line); margin-bottom:18px; }
.xr-brand { display:flex; align-items:center; gap:13px; }
.xr-mark { color:var(--xr-accent); display:flex; align-items:center; }
.xr-wordmark { font-family:var(--xr-mono); font-size:18px; font-weight:600;
  letter-spacing:.05em; color:var(--xr-text); line-height:1.1; }
.xr-caption { font-family:var(--xr-sans); font-size:12px; color:var(--xr-muted); margin-top:3px; }

/* theme switch */
.xr-switch { display:inline-flex; align-items:center; gap:9px; background:none;
  border:none; cursor:pointer; padding:5px 2px; font-family:var(--xr-mono);
  font-size:10px; letter-spacing:.12em; }
.xr-switch .l { color:var(--xr-text); }
.xr-switch .d { color:var(--xr-faint); }
html[data-xr="dark"] .xr-switch .l { color:var(--xr-faint); }
html[data-xr="dark"] .xr-switch .d { color:var(--xr-text); }
.xr-switch-track { position:relative; width:40px; height:22px; border-radius:11px;
  background:var(--xr-line); transition:background .2s; }
html[data-xr="dark"] .xr-switch-track { background:var(--xr-accent); }
.xr-switch-knob { position:absolute; top:3px; left:3px; width:16px; height:16px;
  border-radius:50%; background:var(--xr-surface); box-shadow:0 1px 2px rgba(0,0,0,.3);
  transition:transform .2s; }
html[data-xr="dark"] .xr-switch-knob { transform:translateX(18px); }

/* ── Intro line ─────────────────────────────────────────── */
.xr-intro { display:flex; align-items:center; justify-content:space-between;
  gap:20px; flex-wrap:wrap; padding:0 2px; margin-bottom:18px; }
.xr-intro-text { font-family:var(--xr-sans); font-size:13.5px; color:var(--xr-muted);
  line-height:1.65; max-width:62ch; }
.xr-intro-text b { color:var(--xr-text); font-weight:600; }
.xr-legend { display:flex; gap:16px; font-family:var(--xr-mono); font-size:11px;
  color:var(--xr-muted); white-space:nowrap; }
.xr-sw { display:inline-block; width:10px; height:10px; border-radius:3px;
  margin-right:7px; vertical-align:-1px; }

/* ── Status strip ───────────────────────────────────────── */
.xr-status { display:flex; align-items:center; gap:13px; padding:11px 16px;
  background:var(--xr-raised); border:1px solid var(--xr-line); border-radius:10px;
  font-family:var(--xr-sans); font-size:13px; color:var(--xr-muted); }
.xr-status-step { font-family:var(--xr-mono); font-size:11px; font-weight:600;
  letter-spacing:.08em; color:var(--xr-text); white-space:nowrap; }
.xr-status-sep { width:1px; height:15px; background:var(--xr-line); flex-shrink:0; }
.xr-status-text b { color:var(--xr-text); font-weight:600; }

/* ── Hero (generated sentence) ──────────────────────────── */
.xr-hero { background:var(--xr-surface); border:1px solid var(--xr-line);
  border-radius:16px; padding:26px 32px 30px;
  box-shadow:0 12px 34px -20px rgba(24,18,10,.28); }
.xr-hero-top { display:flex; align-items:center; justify-content:space-between;
  margin-bottom:14px; }
.xr-hero-label { font-family:var(--xr-mono); font-size:11px; font-weight:600;
  letter-spacing:.18em; color:var(--xr-accent); }
.xr-hero-meta { font-family:var(--xr-mono); font-size:10.5px; color:var(--xr-faint);
  letter-spacing:.06em; }
.xr-hero-text { font-family:var(--xr-serif); font-size:25px; line-height:1.82;
  color:var(--xr-text); margin:0; word-wrap:break-word; }
.xr-hero-prompt { color:var(--xr-faint); }
.xr-now { background:var(--xr-live-weak); color:var(--xr-live-ink); border-radius:5px;
  padding:1px 8px; font-weight:600; box-decoration-break:clone;
  -webkit-box-decoration-break:clone; }
.xr-caret { display:inline-block; width:2px; height:1.05em; background:var(--xr-accent);
  margin-left:3px; vertical-align:-2px; animation:xr-blink 1.05s step-end infinite; }
@keyframes xr-blink { 50% { opacity:0; } }

/* ── Generic panel ──────────────────────────────────────── */
.xr-panel { background:var(--xr-surface); border:1px solid var(--xr-line);
  border-radius:12px; padding:16px 18px; height:100%; box-sizing:border-box; }
.xr-phead { display:flex; align-items:baseline; gap:11px; margin-bottom:14px;
  padding-bottom:11px; border-bottom:1px solid var(--xr-line); }
.xr-idx { font-family:var(--xr-mono); font-size:11px; font-weight:600; color:var(--xr-accent); }
.xr-plabel { font-family:var(--xr-mono); font-size:11px; font-weight:600;
  letter-spacing:.15em; color:var(--xr-text); }
.xr-pnote { font-family:var(--xr-mono); font-size:10.5px; color:var(--xr-faint);
  margin-left:auto; letter-spacing:.03em; }

/* ── Candidate rows ─────────────────────────────────────── */
.xr-row { display:flex; align-items:center; gap:12px; padding:5px 0; }
.xr-tok { flex:0 0 88px; text-align:right; font-family:var(--xr-mono); font-size:13px;
  color:var(--xr-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.xr-tok.picked { color:var(--xr-live); font-weight:600; }
.xr-meter { flex:1; height:9px; background:var(--xr-line); border-radius:5px; overflow:hidden; }
.xr-meter i { display:block; height:100%; border-radius:5px; transition:width .35s ease; }
.xr-val { flex:0 0 92px; text-align:right; font-family:var(--xr-mono); font-size:12px;
  color:var(--xr-muted); font-variant-numeric:tabular-nums; }
.xr-val small { font-size:9px; color:var(--xr-faint); margin-left:1px; }
.xr-pick { font-size:9px; letter-spacing:.07em; color:var(--xr-live);
  text-transform:uppercase; margin-left:6px; }

/* ── Attention chips ────────────────────────────────────── */
.xr-chips { display:flex; flex-wrap:wrap; gap:6px; }
.xr-chip { display:inline-flex; flex-direction:column; align-items:center;
  min-width:52px; padding:7px 6px; border-radius:8px; border:1px solid var(--xr-line);
  font-family:var(--xr-mono); font-size:11px; line-height:1.35; }
.xr-chip i { font-style:normal; font-weight:600; margin-top:2px; }

/* ── Confidence ─────────────────────────────────────────── */
.xr-conf { display:grid; grid-template-columns:1.05fr 1fr; gap:22px; }
.xr-conf-num { font-family:var(--xr-mono); font-size:40px; font-weight:600; line-height:1; }
.xr-conf-num small { font-size:18px; margin-left:2px; }
.xr-conf-word { font-family:var(--xr-sans); font-size:13px; margin:5px 0 11px; }
.xr-track { height:8px; background:var(--xr-line); border-radius:4px; overflow:hidden; }
.xr-track i { display:block; height:100%; border-radius:4px; transition:width .35s ease; }
.xr-conf-side { border-left:1px solid var(--xr-line); padding-left:22px; }
.xr-conf-k { font-family:var(--xr-mono); font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--xr-faint); margin-bottom:5px; }
.xr-conf-v { font-family:var(--xr-mono); font-size:30px; font-weight:500;
  color:var(--xr-text); line-height:1; margin-bottom:9px; }
.xr-conf-v small { font-size:13px; color:var(--xr-faint); margin-left:4px; }
.xr-conf-hint { font-family:var(--xr-mono); font-size:10px; color:var(--xr-faint); margin-top:7px; }

/* ── Trace log ──────────────────────────────────────────── */
.xr-trow { display:flex; align-items:center; gap:11px; padding:5px 0;
  border-bottom:1px solid var(--xr-line); }
.xr-trow:last-child { border-bottom:none; }
.xr-tstep { font-family:var(--xr-mono); font-size:11px; color:var(--xr-faint); width:22px; }
.xr-tword { flex:0 0 94px; font-family:var(--xr-mono); font-size:12.5px; font-weight:600;
  color:var(--xr-text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.xr-tbar { flex:1; height:6px; background:var(--xr-line); border-radius:3px; overflow:hidden; }
.xr-tbar i { display:block; height:100%; border-radius:3px; }
.xr-tval { font-family:var(--xr-mono); font-size:11px; color:var(--xr-muted);
  width:36px; text-align:right; }
.xr-empty { font-family:var(--xr-mono); font-size:12px; color:var(--xr-faint); padding:8px 0; }
.xr-foot { font-family:var(--xr-sans); font-size:11.5px; color:var(--xr-faint); line-height:1.5;
  margin-top:12px; padding-top:10px; border-top:1px solid var(--xr-line); }
.xr-foot i { font-style:italic; }

/* ── Detail-level control (sliding segmented control) ───── */
.xr-depth { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin:2px 2px 22px; }
.xr-depth-lab { font-family:var(--xr-mono); font-size:10px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--xr-faint); }
.xr-seg { position:relative; display:inline-grid; grid-template-columns:repeat(3, minmax(96px, 1fr));
  padding:3px; border:1px solid var(--xr-line); border-radius:11px; background:var(--xr-raised); }
.xr-seg-thumb { position:absolute; top:3px; bottom:3px; left:3px; width:calc((100% - 6px) / 3);
  background:var(--xr-accent); border-radius:8px; z-index:0; box-shadow:0 1px 2px rgba(0,0,0,.14);
  transition:transform .28s cubic-bezier(.32,.72,0,1); }
html[data-xr-level="intermediate"] .xr-seg-thumb { transform:translateX(100%); }
html[data-xr-level="beginner"]     .xr-seg-thumb { transform:translateX(200%); }
.xr-seg-btn { position:relative; z-index:1; background:none; border:none; cursor:pointer;
  padding:7px 6px; font-family:var(--xr-mono); font-size:12px; letter-spacing:.02em;
  color:var(--xr-muted); text-align:center; transition:color .2s; }
.xr-seg-btn:hover { color:var(--xr-text); }
html[data-xr-level="beginner"]     .xr-seg-btn[data-lvl="beginner"],
html[data-xr-level="intermediate"] .xr-seg-btn[data-lvl="intermediate"],
html[data-xr-level="expert"]       .xr-seg-btn[data-lvl="expert"] { color:var(--xr-on-accent); }
.xr-depth-hint { font-family:var(--xr-sans); font-size:12px; color:var(--xr-muted); }
.xr-depth-hint::after { content:"every stage of the machine, in order"; }
html[data-xr-level="intermediate"] .xr-depth-hint::after { content:"the key parts of how each word forms"; }
html[data-xr-level="beginner"]     .xr-depth-hint::after { content:"just the story — the words it picks"; }

/* ── Control-group label ────────────────────────────────── */
.xr-controls-lab { font-family:var(--xr-mono); font-size:10px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--xr-faint); margin:10px 2px 2px; }

/* progressive disclosure: hide stages above the chosen level */
html[data-xr-level="beginner"] .xr-lvl-inter,
html[data-xr-level="beginner"] .xr-lvl-expert,
html[data-xr-level="intermediate"] .xr-lvl-expert { display:none !important; }

/* ── Section divider ────────────────────────────────────── */
.xr-divider { font-family:var(--xr-mono); font-size:11px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--xr-faint); margin:18px 2px 0;
  padding-top:15px; border-top:1px solid var(--xr-line); }
.xr-divider::before { content:""; display:inline-block; width:6px; height:6px;
  background:var(--xr-accent); border-radius:1px; margin-right:9px; vertical-align:1px; }

/* ── Input embedding strip ──────────────────────────────── */
.xr-strip-lab { font-family:var(--xr-mono); font-size:10.5px; color:var(--xr-faint); margin-bottom:5px; }
.xr-strip { display:flex; gap:2px; flex-wrap:nowrap; overflow:hidden; }
.xr-cell { width:11px; height:20px; border-radius:2px; flex:0 0 auto; border:1px solid var(--xr-line); }

/* ── Attention heads grid ───────────────────────────────── */
.xr-heads-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; }
.xr-head-box { border:1px solid var(--xr-line); border-radius:8px; padding:8px 10px; }
.xr-head-id { font-family:var(--xr-mono); font-size:10px; font-weight:600; color:var(--xr-accent); }
.xr-head-strip { display:flex; gap:1px; margin-top:5px; }
.xr-mini { flex:1; height:16px; border-radius:2px; min-width:3px; border:1px solid var(--xr-line); }

/* ── Feed-forward ───────────────────────────────────────── */
.xr-ffn-flow { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-bottom:10px; }
.xr-ffn-dim { font-family:var(--xr-mono); font-size:14px; font-weight:600; color:var(--xr-text);
  background:var(--xr-raised); border:1px solid var(--xr-line); border-radius:6px; padding:3px 10px; }
.xr-ffn-arrow { font-family:var(--xr-mono); font-size:10.5px; color:var(--xr-faint); letter-spacing:.03em; }
.xr-ffn-meta { font-family:var(--xr-sans); font-size:12.5px; color:var(--xr-muted); margin-bottom:8px; }
.xr-ffn-meta b { color:var(--xr-live); font-weight:600; }
.xr-neuron { display:flex; align-items:center; gap:10px; padding:3px 0; }
.xr-neuron-id { font-family:var(--xr-mono); font-size:11px; color:var(--xr-faint); width:74px; }
.xr-neuron-bar { flex:1; height:8px; background:var(--xr-line); border-radius:4px; overflow:hidden; }
.xr-neuron-bar i { display:block; height:100%; background:var(--xr-live); border-radius:4px; transition:width .35s ease; }
.xr-neuron-val { font-family:var(--xr-mono); font-size:11px; color:var(--xr-muted); width:46px; text-align:right; }

/* ── Logit lens ─────────────────────────────────────────── */
.xr-lens-row { display:flex; align-items:center; gap:11px; padding:3px 0; }
.xr-lens-blk { font-family:var(--xr-mono); font-size:11px; color:var(--xr-faint); width:42px; }
.xr-lens-tok { flex:0 0 96px; font-family:var(--xr-mono); font-size:13px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.xr-lens-bar { flex:1; height:7px; background:var(--xr-line); border-radius:4px; overflow:hidden; }
.xr-lens-bar i { display:block; height:100%; border-radius:4px; transition:width .35s ease; }
.xr-lens-val { font-family:var(--xr-mono); font-size:11px; color:var(--xr-muted);
  width:34px; text-align:right; }

/* ── Gradio controls, restyled to match the instrument ──── */
.gradio-container input[type="range"] { accent-color:var(--xr-accent); height:5px; cursor:pointer; }
.gradio-container input[type="checkbox"] { accent-color:var(--xr-accent); width:16px; height:16px; cursor:pointer; }
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
  font-family:var(--xr-sans) !important; border-radius:10px !important;
  border-color:var(--xr-line) !important; transition:border-color .15s, box-shadow .15s; }
.gradio-container textarea:focus,
.gradio-container input:focus {
  border-color:var(--xr-accent) !important; box-shadow:0 0 0 3px var(--xr-focus) !important; outline:none !important; }
.gradio-container button.primary,
.gradio-container button.secondary {
  font-family:var(--xr-mono) !important; letter-spacing:.02em; border-radius:10px !important;
  transition:filter .15s, transform .05s; }
.gradio-container button.primary:hover { filter:brightness(1.06); }
.gradio-container button.primary:active,
.gradio-container button.secondary:active { transform:translateY(1px); }

/* ── micro-craft: focus rings + subtle hovers ───────────── */
.xr-seg-btn:focus-visible, .xr-switch:focus-visible {
  outline:2px solid var(--xr-focus); outline-offset:3px; border-radius:8px; }
.xr-panel { transition:border-color .25s; }
.xr-panel:hover { border-color:var(--xr-line-strong); }
.xr-chip { transition:transform .12s; }
.xr-chip:hover { transform:translateY(-1px); }

/* ── Mobile: stack everything to one column ─────────────── */
@media (max-width: 700px) {
  .gradio-container .row { flex-direction:column !important; }
  .xr-conf { grid-template-columns:1fr; gap:16px; }
  .xr-conf-side { border-left:none; border-top:1px solid var(--xr-line); padding-left:0; padding-top:16px; }
  .xr-heads-grid { grid-template-columns:repeat(2, 1fr); }
  .xr-hero { padding:20px 18px 22px; }
  .xr-hero-text { font-size:21px; line-height:1.72; }
  .xr-panel { padding:14px 15px; }
  .xr-depth { gap:10px; }
  .xr-seg { grid-template-columns:repeat(3, minmax(80px, 1fr)); }
}
</style>
"""

_HEADER = """
<div class="xr-head">
  <div class="xr-brand">
    <span class="xr-mark" aria-hidden="true">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <rect x="2"  y="8"  width="3" height="8"  rx="1.5" fill="currentColor"/>
        <rect x="7"  y="3"  width="3" height="18" rx="1.5" fill="currentColor"/>
        <rect x="12" y="10" width="3" height="4"  rx="1.5" fill="currentColor"/>
        <rect x="17" y="6"  width="3" height="12" rx="1.5" fill="currentColor"/>
      </svg>
    </span>
    <div>
      <div class="xr-wordmark">LLM&middot;X-RAY</div>
      <div class="xr-caption">a live look inside GPT-2 as it writes</div>
    </div>
  </div>
  <button class="xr-switch" onclick="xrToggle()" aria-label="Toggle light or dark theme">
    <span class="l">light</span>
    <span class="xr-switch-track"><span class="xr-switch-knob"></span></span>
    <span class="d">dark</span>
  </button>
</div>
"""

_INTRO = """
<div class="xr-intro">
  <div class="xr-intro-text">
    A language model writes <b>one token at a time</b> (a token is a word or word-piece)
    &mdash; score every possible next token, look back at the sentence so far, commit to
    one, repeat. Press <b>Generate</b> and watch the loop run.
  </div>
  <div class="xr-legend">
    <span><span class="xr-sw" style="background:var(--xr-live)"></span>word it picks</span>
    <span><span class="xr-sw" style="background:var(--xr-accent)"></span>where it looks</span>
  </div>
</div>
"""

_LEVELS = """
<div class="xr-depth">
  <span class="xr-depth-lab">detail</span>
  <div class="xr-seg" role="group" aria-label="Detail level">
    <span class="xr-seg-thumb" aria-hidden="true"></span>
    <button class="xr-seg-btn" data-lvl="expert" onclick="xrLevel('expert')">detailed</button>
    <button class="xr-seg-btn" data-lvl="intermediate" onclick="xrLevel('intermediate')">standard</button>
    <button class="xr-seg-btn" data-lvl="beginner" onclick="xrLevel('beginner')">simple</button>
  </div>
  <span class="xr-depth-hint" aria-live="polite"></span>
</div>
"""

PAGE_HEADER = _CSS + _HEADER + _INTRO + _LEVELS

# Runs on page load via gr.Blocks(js=…). Defines window.xrToggle and forces a
# consistent LIGHT default — syncing data-xr (our panels) with Gradio's .dark
# class (its chrome) so the two theme systems can never disagree.
TOGGLE_JS = """
() => {
  const apply = (on) => {
    document.documentElement.setAttribute('data-xr', on ? 'dark' : 'light');
    [document.documentElement, document.body,
     document.querySelector('.gradio-container'),
     document.querySelector('gradio-app')].forEach(el => { if (el) el.classList.toggle('dark', on); });
  };
  window.xrToggle = () => apply(document.documentElement.getAttribute('data-xr') !== 'dark');
  window.xrLevel = (lvl) => document.documentElement.setAttribute('data-xr-level', lvl);
  apply(false);
  window.xrLevel('expert');   // open on the detailed view (full forward pass)
  return [];
}
"""
