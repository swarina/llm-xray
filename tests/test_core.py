"""
Smoke + correctness + security tests for LLM X-Ray.

Run standalone (no pytest needed):
    ./.venv/bin/python tests/test_core.py
or with pytest if installed:
    ./.venv/bin/pytest tests/
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import panels  # noqa: E402
from core import XRayModel  # noqa: E402

_xm = None


def xm():
    global _xm
    if _xm is None:
        _xm = XRayModel()
    return _xm


# ── correctness ──────────────────────────────────────────────────────────────

def test_logit_lens_final_matches_greedy():
    m = xm()
    ids = m.encode("the capital of france is")
    out, _ = m.forward_capturing(ids, m.n_layer - 1)
    rows = m.logit_lens(out.hidden_states)
    greedy = m.decode([int(out.logits[0, -1].argmax())])
    assert rows[-1][1] == greedy, "logit lens final layer must equal the real greedy token"


def test_ffn_hook_is_faithful():
    m = xm()
    ids = m.encode("hello world today")
    blk = m.model.transformer.h[5]
    store = {}
    h = blk.mlp.register_forward_hook(lambda mod, i, o: store.__setitem__("in", i[0].detach()))
    _, act = m.forward_capturing(ids, 5)
    h.remove()
    manual = blk.mlp.act(blk.mlp.c_fc(store["in"]))[0, -1]
    assert act.shape[0] == 3072
    assert torch.allclose(act, manual, atol=1e-4), "captured FFN activation must match recompute"


def test_input_embedding_dims():
    m = xm()
    tv, pv = m.input_embedding(m.encode("test"))
    assert tv.shape[0] == 768 and pv.shape[0] == 768


def test_softmax_sums_to_one():
    m = xm()
    out, _ = m.forward_capturing(m.encode("one two three"), 0)
    p = F.softmax(out.logits[0, -1] / 0.8, dim=-1)
    assert abs(float(p.sum()) - 1.0) < 1e-4


# ── security: HTML escaping ──────────────────────────────────────────────────

def test_live_panels_escape_html():
    mal = "<b>x</b>"
    assert "&lt;b&gt;" in panels.text_panel("p", mal, mal, False)
    assert "<b>" not in panels.candidates_panel([mal], [0.5], 0)
    assert "<b>" not in panels.heads_panel([(0, [1.0])], [mal], 0)


def test_app_tokens_html_escaped():
    import app
    out = app.analyze("a < b", 11, 0, 0.8, 8)[0]
    assert "&lt;" in out, "app.py token table must HTML-escape token text (XSS guard)"
    assert "<code><" not in out, "no raw '<' token should reach the DOM"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"\nall {len(fns)} tests passed")
