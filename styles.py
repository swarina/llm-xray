"""
styles.py — CSS variable theme system, app header, and intro block for LLM X-Ray.

The <style> block in PAGE_HEADER renders globally inside Gradio (not scoped), so
every panel that uses var(--xr-*) names automatically inherits the active theme.
The JS toggle sets data-xr="dark" on <html>, which switches the variable set.
"""

_CSS = """
<style>
/* ── Light theme (default) ─────────────────────────────────────── */
:root {
  --xr-page:      #f1f5f9;
  --xr-surface:   #ffffff;
  --xr-text:      #0f172a;
  --xr-mute:      #475569;
  --xr-hint:      #94a3b8;
  --xr-border:    #e2e8f0;
  --xr-track:     #e8edf2;

  --xr-blue:      #3b82f6;
  --xr-blue-bg:   #eff6ff;
  --xr-green:     #16a34a;
  --xr-green-bg:  #f0fdf4;
  --xr-amber:     #d97706;
  --xr-amber-bg:  #fffbeb;
  --xr-red:       #dc2626;
  --xr-red-bg:    #fef2f2;
  --xr-purple:    #7c3aed;

  --xr-shadow:    0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.04);
  --xr-shadow-md: 0 4px 6px rgba(0,0,0,.06), 0 2px 4px rgba(0,0,0,.04);
}

/* ── Dark theme ────────────────────────────────────────────────── */
[data-xr="dark"] {
  --xr-page:      #0f172a;
  --xr-surface:   #1e293b;
  --xr-text:      #f1f5f9;
  --xr-mute:      #94a3b8;
  --xr-hint:      #475569;
  --xr-border:    #334155;
  --xr-track:     #293548;

  --xr-blue:      #60a5fa;
  --xr-blue-bg:   #172554;
  --xr-green:     #4ade80;
  --xr-green-bg:  #052e16;
  --xr-amber:     #fbbf24;
  --xr-amber-bg:  #451a03;
  --xr-red:       #f87171;
  --xr-red-bg:    #450a0a;
  --xr-purple:    #a78bfa;

  --xr-shadow:    0 1px 3px rgba(0,0,0,.4),  0 1px 2px rgba(0,0,0,.3);
  --xr-shadow-md: 0 4px 6px rgba(0,0,0,.5),  0 2px 4px rgba(0,0,0,.4);
}

/* ── Page background ───────────────────────────────────────────── */
.gradio-container { background: var(--xr-page) !important; transition: background .25s; }

/* ── Base card ─────────────────────────────────────────────────── */
.xr-card {
  background:    var(--xr-surface);
  border:        1px solid var(--xr-border);
  border-radius: 12px;
  padding:       16px 20px;
  box-shadow:    var(--xr-shadow);
  box-sizing:    border-box;
  transition:    background .25s, border-color .25s, box-shadow .25s;
}

/* ── Typography helpers ────────────────────────────────────────── */
.xr-eyebrow {
  font-size: 11px; font-weight: 700; letter-spacing: .07em;
  text-transform: uppercase; color: var(--xr-hint); margin-bottom: 4px;
}
.xr-heading { font-size: 16px; font-weight: 600; color: var(--xr-text); margin-bottom: 3px; }
.xr-sub     { font-size: 12.5px; color: var(--xr-mute); line-height: 1.55; margin-bottom: 13px; }
.xr-hr      { border: none; border-top: 1px solid var(--xr-border); margin: 12px 0; }

/* ── Bar rows (candidates panel) ───────────────────────────────── */
.xr-bar-row   { display: flex; align-items: center; gap: 9px; margin: 5px 0; }
.xr-bar-label {
  width: 90px; text-align: right; font-family: monospace; font-size: 13px;
  color: var(--xr-mute); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.xr-bar-track { flex: 1; background: var(--xr-track); border-radius: 6px; height: 22px; overflow: hidden; }
.xr-bar-fill  { height: 100%; border-radius: 6px; transition: width .35s ease; }
.xr-bar-pct   {
  width: 48px; text-align: right; font-size: 12.5px;
  color: var(--xr-mute); font-variant-numeric: tabular-nums;
}

/* ── Attention chips ───────────────────────────────────────────── */
.xr-chip {
  display: flex; flex-direction: column; align-items: center;
  min-width: 54px; padding: 8px 5px; border-radius: 8px;
  border: 1px solid var(--xr-border); font-size: 11px; text-align: center;
  cursor: default; transition: background .2s, border-color .2s;
}

/* ── Sentence display ──────────────────────────────────────────── */
.xr-sentence {
  font-family: Georgia, serif; font-size: 20px;
  line-height: 2.2; word-wrap: break-word; color: var(--xr-text);
}
@keyframes xr-blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* ── Decision log rows ─────────────────────────────────────────── */
.xr-log-row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 0; border-bottom: 1px solid var(--xr-border); font-size: 13px;
}
.xr-log-step { color: var(--xr-hint); width: 24px; font-variant-numeric: tabular-nums; }
.xr-log-word {
  flex: 0 0 100px; font-family: monospace; font-weight: 600;
  color: var(--xr-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.xr-log-bar  { flex: 1; background: var(--xr-track); border-radius: 4px; height: 8px; overflow: hidden; }
.xr-log-pct  { width: 36px; text-align: right; color: var(--xr-mute); }

/* ── Confidence meter ──────────────────────────────────────────── */
.xr-meter-track {
  background: var(--xr-track); border-radius: 8px; height: 12px; overflow: hidden; margin: 8px 0;
}
</style>
"""

_HEADER = """
<div id="xr-hdr" style="
    background: #0f172a; border-radius: 12px; padding: 14px 22px; margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
    transition: background .25s;">

  <div style="display:flex; align-items:center; gap:14px;">
    <div style="
        width:40px; height:40px; flex-shrink:0; border-radius:10px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        display:flex; align-items:center; justify-content:center; font-size:20px;">
      🔬
    </div>
    <div>
      <div style="font-size:23px; font-weight:800; color:#f8fafc;
                  letter-spacing:-.6px; line-height:1.15;">LLM X-Ray</div>
      <div style="font-size:12px; color:#64748b; margin-top:2px;">
        real GPT-2 &nbsp;·&nbsp; every number from the actual model &nbsp;·&nbsp;
        step by step
      </div>
    </div>
  </div>

  <button id="xr-btn" onclick="xrToggle()" style="
      background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
      border-radius: 9px; padding: 8px 18px; color: #e2e8f0; cursor: pointer;
      font-size: 13px; font-family: inherit; display: flex; align-items: center;
      gap: 7px; transition: background .15s; white-space: nowrap;">
    <span id="xr-ic">🌙</span>
    <span id="xr-lb">Dark mode</span>
  </button>
</div>
"""

# gr.Blocks(js=TOGGLE_JS) runs this on page load, defining window.xrToggle.
# This is necessary because Gradio sets gr.HTML() content via innerHTML, and
# browsers silently drop <script> tags injected that way — onclick handlers
# still fire, but the function they reference would be undefined without this.
TOGGLE_JS = """
() => {
  let dark = false;
  window.xrToggle = function () {
    dark = !dark;
    document.documentElement.setAttribute('data-xr', dark ? 'dark' : '');
    const hdr = document.getElementById('xr-hdr');
    const ic  = document.getElementById('xr-ic');
    const lb  = document.getElementById('xr-lb');
    if (hdr) hdr.style.background = dark ? '#020617' : '#0f172a';
    if (ic)  ic.textContent  = dark ? '☀️' : '🌙';
    if (lb)  lb.textContent  = dark ? 'Light mode' : 'Dark mode';
  };
  return [];
}
"""

_INTRO = """
<div class="xr-card" style="border-left: 4px solid var(--xr-blue); margin-bottom: 4px;">
  <div style="font-size:15px; font-weight:600; color:var(--xr-text); margin-bottom:9px;">
    How a language model writes — in one sentence
  </div>
  <div style="font-size:14px; color:var(--xr-mute); line-height:1.8;">
    It adds <strong style="color:var(--xr-text)">one word at a time</strong>:
    <strong style="color:var(--xr-text)">(1)</strong> score every possible next word,
    <strong style="color:var(--xr-text)">(2)</strong> look back at the sentence so far to decide,
    <strong style="color:var(--xr-text)">(3)</strong> commit to one word, add it, and repeat.
    Press <strong style="color:var(--xr-text)">⚡ Generate</strong> and watch all three happen live.
  </div>
  <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:14px;
              font-size:13px; color:var(--xr-mute); align-items:center;">
    <span>
      <span style="background:#fef08a; color:#713f12; border-radius:4px;
                   padding:1px 8px; font-weight:600;">word</span>
      &ensp;being added now
    </span>
    <span><span style="color:var(--xr-green); font-size:15px;">●</span>&ensp;picked / confident</span>
    <span><span style="color:var(--xr-blue);  font-size:15px;">●</span>&ensp;attention (looked at)</span>
    <span><span style="color:var(--xr-amber); font-size:15px;">●</span>&ensp;somewhat unsure</span>
    <span><span style="color:var(--xr-red);   font-size:15px;">●</span>&ensp;very unsure</span>
  </div>
</div>
"""

PAGE_HEADER = _CSS + _HEADER + _INTRO
