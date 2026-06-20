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
    out, _, _ = m.forward_capturing(ids, m.n_layer - 1)
    rows = m.logit_lens(out.hidden_states)
    greedy = m.decode([int(out.logits[0, -1].argmax())])
    assert rows[-1][1] == greedy, "logit lens final layer must equal the real greedy token"


def test_ffn_hook_is_faithful():
    m = xm()
    ids = m.encode("hello world today")
    blk = m.model.transformer.h[5]
    store = {}
    h = blk.mlp.register_forward_hook(lambda mod, i, o: store.__setitem__("in", i[0].detach()))
    _, act, _ = m.forward_capturing(ids, 5)
    h.remove()
    manual = blk.mlp.act(blk.mlp.c_fc(store["in"]))[0, -1]
    assert act.shape[0] == 3072
    assert torch.allclose(act, manual, atol=1e-4), "captured FFN activation must match recompute"


def test_qk_matches_attention():
    m = xm()
    ids = m.encode("the capital of france is")
    out, _, qkv = m.forward_capturing(ids, 5)
    _, weights = m.qk_scores(qkv, head=0)
    real = out.attentions[5][0, 0, -1]                  # head 0, last token's weights
    assert torch.allclose(weights, real, atol=1e-4), "q·k softmax must equal the model's attention"


def test_input_embedding_dims():
    m = xm()
    tv, pv = m.input_embedding(m.encode("test"))
    assert tv.shape[0] == 768 and pv.shape[0] == 768


def test_residual_stream_grows():
    m = xm()
    out, _, _ = m.forward_capturing(m.encode("the meaning of life is"), 5)
    norms = [float(h[0, -1].norm()) for h in out.hidden_states]
    assert len(norms) == m.n_layer + 1          # embedding + one per block
    assert norms[-1] > norms[0]                  # the residual stream accumulates


def test_softmax_sums_to_one():
    m = xm()
    out, _, _ = m.forward_capturing(m.encode("one two three"), 0)
    p = F.softmax(out.logits[0, -1] / 0.8, dim=-1)
    assert abs(float(p.sum()) - 1.0) < 1e-4


# ── security: HTML escaping ──────────────────────────────────────────────────

def test_live_panels_escape_html():
    mal = "<b>x</b>"
    assert "&lt;b&gt;" in panels.text_panel("p", mal, mal, False)
    assert "<b>" not in panels.candidates_panel([mal], [0.5], 0)
    assert "<b>" not in panels.heads_panel([(0, [1.0])], [mal], 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"\nall {len(fns)} tests passed")
