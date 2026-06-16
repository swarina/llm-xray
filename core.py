"""
core.py — the shared model layer for LLM X-Ray.

One place that loads the model and exposes the extractions every surface needs
(tokens, a forward pass with attentions + hidden states, next-token math, and
the logit lens). The apps import this instead of each re-implementing it, so a
correctness fix lands once.

Model is configurable: XRayModel() uses $XRAY_MODEL or "gpt2". Other small
causal LMs (distilgpt2, gpt2-medium) work too; the logit lens reaches for a
GPT-2-style final norm and falls back gracefully.
"""

import os

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
        with torch.no_grad():
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
