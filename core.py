"""
core.py — the shared model layer for LLM X-Ray.

One place that loads the model and exposes the extractions every surface needs
(tokens, a forward pass with attentions + hidden states, next-token math, and
the logit lens). The apps import this instead of each re-implementing it, so a
correctness fix lands once.

Model is configurable: XRayModel() uses $XRAY_MODEL or "gpt2". Only the GPT-2
family is supported (gpt2, gpt2-medium/large/xl, distilgpt2) because the
forward-pass extraction (FFN activations, wte/wpe embeddings) is GPT-2-specific;
other architectures fail fast at construction with a clear message.
"""

import os
import threading

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("XRAY_MODEL", "gpt2")


class XRayModel:
    def __init__(self, name: str = DEFAULT_MODEL):
        self.name = name
        print(f"loading {name} (cached after first run)…")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, output_attentions=True, output_hidden_states=True
        )
        self.model.eval()
        cfg = self.model.config
        self.n_layer = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
        self.n_head = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))
        self.vocab = cfg.vocab_size
        self.eos_id = self.tok.eos_token_id
        self.max_positions = getattr(cfg, "n_positions",
                                     getattr(cfg, "max_position_embeddings", 1024))
        self._lock = threading.Lock()  # the model + FFN hook aren't concurrency-safe
        self._require_gpt2_layout()

    def _require_gpt2_layout(self):
        """The internals extraction assumes the GPT-2 block layout. Fail fast
        with a clear message rather than crashing mid-generation."""
        tf = getattr(self.model, "transformer", None)
        ok = (
            tf is not None
            and hasattr(tf, "wte") and hasattr(tf, "wpe") and hasattr(tf, "ln_f")
            and len(getattr(tf, "h", [])) > 0
            and hasattr(tf.h[0], "mlp") and hasattr(tf.h[0].mlp, "act")
        )
        if not ok:
            raise ValueError(
                f"{self.name!r} is not a GPT-2-family model. LLM X-Ray's forward-pass "
                "extraction (FFN activations, wte/wpe embeddings) is GPT-2-specific. "
                "Supported: gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2."
            )

    # ── tokens ──────────────────────────────────────────────────────────────
    def encode(self, text: str):
        return self.tok(text, return_tensors="pt").input_ids

    def decode(self, ids) -> str:
        return self.tok.decode(ids)

    def short_labels(self, id_list, width: int = 7):
        out = []
        for t in id_list:
            s = self.tok.decode([t]).strip()[:width]
            out.append(s if s else "_")
        return out

    # ── forward pass ────────────────────────────────────────────────────────
    def forward(self, ids):
        with self._lock, torch.no_grad():
            return self.model(ids)

    # ── logit lens ──────────────────────────────────────────────────────────
    def _final_norm(self):
        # GPT-2: transformer.ln_f. Other models expose it elsewhere; fall back
        # to identity so the lens still runs (slightly less faithful).
        base = getattr(self.model, "transformer", None)
        return getattr(base, "ln_f", None) if base is not None else None

    def logit_lens(self, hidden_states):
        """Top-1 next-token prediction read off EACH layer's residual.

        Applies the model's final norm + unembedding to every layer's last-token
        vector, so you can watch the prediction settle with depth. Returns a list
        of (layer_index, token_str, probability), layer 1 → n_layer.
        """
        norm = self._final_norm()
        head = self.model.get_output_embeddings()  # the unembedding (lm_head)
        rows = []
        with torch.no_grad():
            for li in range(1, len(hidden_states)):
                h = hidden_states[li][0, -1]
                if norm is not None:
                    h = norm(h)
                probs = F.softmax(head(h), dim=-1)
                p, idx = probs.max(dim=-1)
                rows.append((li, self.tok.decode([int(idx)]), float(p)))
        return rows

    # ── forward pass internals (GPT-2 layout) ───────────────────────────────
    def forward_capturing(self, ids, layer: int):
        """A single forward pass that also captures, for `layer`: the FFN's
        post-activation (3072-dim, last token) and the raw c_attn output (the
        q/k/v for every position) — both via temporary hooks, one forward."""
        store = {}
        block = self.model.transformer.h[layer]

        def ffn_hook(_m, _i, o):
            store["ffn"] = o[0, -1].detach()           # post-GELU, last token

        def attn_hook(_m, _i, o):
            store["qkv"] = o[0].detach()               # c_attn output [seq, 3*n_embd]

        # the lock keeps a concurrent request's forward from firing these hooks
        with self._lock:
            h1 = block.mlp.act.register_forward_hook(ffn_hook)
            h2 = block.attn.c_attn.register_forward_hook(attn_hook)
            try:
                with torch.no_grad():
                    out = self.model(ids)
            finally:
                h1.remove()
                h2.remove()
        return out, store.get("ffn"), store.get("qkv")

    def qk_scores(self, qkv, head: int = 0):
        """From a captured c_attn output, the scaled query·key scores and their
        softmax (= the attention weights) for the LAST token at `head`. This is
        how attention is actually computed, before it becomes the weights."""
        n_embd = self.model.config.n_embd
        hd = n_embd // self.n_head
        q, k, _ = qkv.split(n_embd, dim=-1)            # each [seq, n_embd]
        q = q.view(-1, self.n_head, hd)
        k = k.view(-1, self.n_head, hd)
        scores = (k[:, head] @ q[-1, head]) / (hd ** 0.5)   # [seq]
        return scores, torch.softmax(scores, dim=-1)

    def input_embedding(self, ids):
        """The two vectors that assemble the last token's input: its token
        embedding (wte row) and its learned position embedding (wpe row)."""
        wte = self.model.transformer.wte.weight
        wpe = self.model.transformer.wpe.weight
        last = int(ids[0, -1])
        pos = min(ids.shape[1] - 1, wpe.shape[0] - 1)
        return wte[last].detach(), wpe[pos].detach()
