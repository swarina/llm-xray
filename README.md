# 🔬 llm-xray

> See how an LLM works **end to end** — live, on a real GPT-2.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Built with](https://img.shields.io/badge/built%20with-PyTorch%20%2B%20Transformers-ee4c2c)
![Tests](https://github.com/swarina/llm-xray/actions/workflows/tests.yml/badge.svg)

This isn't a diagram of how language models work — it's the **real machine, running**.
Load a real GPT-2 and watch it generate text one token at a time, with every stage
of the forward pass exposed: tokens → embeddings → attention → feed-forward →
layer-by-layer prediction → sampling → repeat. Every number on screen comes from the
actual model. A **detail-level toggle** scales it from *"watch it write"* (beginner)
to per-head attention and per-neuron FFN activations (expert).

It began as a fact-check of the article
[*How LLMs actually work*](https://www.0xkato.xyz/how-llms-actually-work) and grew
into a tool for *seeing* the pipeline instead of just reading about it.

---

## How this differs from other transformer visualizers

There are excellent tools in this space, and this one deliberately doesn't try to
out-polish them on their turf. Honestly:

- **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** (Georgia Tech) does *single-step* next-token prediction beautifully, in-browser.
- **[bbycroft.net/llm](https://bbycroft.net/llm)** is a stunning 3D tour of a tiny toy model's architecture.

What this tool does that they don't:

1. **The generation *loop*, narrated** — most tools show one forward pass; this runs the autoregressive loop token by token, with a decision log. You watch it *write*.
2. **The complete forward pass, including the FFN** — embeddings, per-head attention, *and the feed-forward network* (where most parameters and "facts" live), all extracted live.
3. **A logit lens fused into generation** — watch the prediction form layer by layer, over real generated text. For *"the capital of france is"* it surfaces `Paris` at layers 10–11, then the final layer overrides it to `the`.
4. **Beginner → Expert in one tool** — progressive disclosure, not one fixed depth.
5. **Honest about interpretability** — it flags that attention weights ≠ importance, that the first-token "sink" dominates, that head-averaging hides specialization.
6. **A hackable real-PyTorch lab** — swap models, instrument it, extend it.

---

## The three tools

| Tool | Where | What it is |
|---|---|---|
| [`live.py`](live.py) | browser, **`:7861`** | **the flagship** — live token-by-token generation, full forward pass, depth toggle, dark/light |
| [`app.py`](app.py) | browser, `:7860` | static, slider-driven panels for one prompt (tokens, attention heatmap, residual, next-token) |
| [`llm_xray.py`](llm_xray.py) | terminal | a printed walkthrough of one forward pass + a saved attention heatmap |

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python live.py        # → http://127.0.0.1:7861  (start here)
./.venv/bin/python app.py         # → http://127.0.0.1:7860
./.venv/bin/python llm_xray.py    # terminal
```

First run downloads GPT-2 (~500 MB) and caches it. Swap models *within the GPT-2
family* with `XRAY_MODEL=distilgpt2 ./.venv/bin/python live.py` (or `--model` for
the CLI) — `gpt2-medium`/`large`/`xl`, `distilgpt2`. Other architectures are
rejected at load, since the forward-pass extraction is GPT-2-specific.

Run the tests with `./.venv/bin/python tests/test_core.py`.

---

## What you see in `live.py`, stage by stage

Press **Generate** and each token reveals the whole machine, in pipeline order:

| # | Panel | What it shows (live from the model) |
|---|---|---|
| 00 | **Input** | token embedding (wte) + position embedding (wpe) assembling the input vector |
| 01 | **Attention** | where the last token looked (summary, averaged over heads) |
| 02 | **Heads** | all 12 heads' real attention patterns — they specialize |
| 03 | **Feed-forward** | 768 → 3072 → GELU → 768, with the real top-firing neurons |
| 04 | **Logit lens** | the prediction forming layer by layer |
| 05 | **Next token** | the candidate distribution it samples from |
| 06 | **Confidence** | how peaked the choice is, plus entropy |
| 07 | **Trace** | the running log of every committed word |

…and the **detail level** control gates them:

- **Beginner** — the story: sentence, candidates, confidence, trace
- **Intermediate** — adds input, attention summary, logit lens
- **Expert** — adds per-head attention and the feed-forward neurons

![Temperature reshapes the next-token distribution](docs/fig-temperature.png)

*Low temperature → peaked and safe. High → flat and surprising. The same logits,
three temperatures.*

---

## Concepts, for newcomers

**Tokens** — models read integer ids for *subword* chunks, not letters.
`strawberry` → `['st','raw','berry']`, which is exactly why "how many r's?" is hard.

**Embeddings** — each id looks up a row: a vector of 768 numbers (4,096 in a 7B
model). GPT-2 adds a learned position vector; their sum enters the stack.

**Attention** — each token looks back at earlier tokens and weights them. It runs
in 12 parallel heads that specialize. *(Caveat the app makes: a high weight shows
where it looked, not proof of what mattered.)*

![A real GPT-2 attention head](docs/fig-attention.png)

**Feed-forward** — a per-token MLP that expands to 3,072 dims, applies GELU, and
contracts. Most of GPT-2's parameters live here; specific neurons fire for
specific patterns and facts.

**Residual stream** — each block *adds* to the vector rather than replacing it, so
information has a direct path to the top. You can watch it grow:

![The residual stream grows each block](docs/fig-residual.png)

**Logits, temperature, sampling** — the final vector becomes one score per
vocabulary token; temperature + softmax turn scores into probabilities, and the
model samples one. Low temperature ≈ always take the top word; high ≈ give
long-shots a real chance.

---

## How GPT-2 differs from today's models

GPT-2 (2019) is the most inspectable open model, but predates a few things modern
models treat as standard. The *shape* of the pipeline is identical.

| Component | GPT-2 | Modern (LLaMA-style) |
|---|---|---|
| Position info | learned absolute embeddings | **RoPE** (rotary) |
| Normalization | LayerNorm | **RMSNorm** |
| Attention | full multi-head | **Grouped-Query Attention** + KV cache |
| FFN activation | GELU | **SwiGLU** |

---

## Project structure

```
llm-xray/
├── core.py            # XRayModel — the shared model layer (load + extraction + logit lens)
├── live.py            # flagship live UI (port 7861)
├── styles.py          # the visual system: CSS theme, header, depth toggle, JS
├── panels.py          # HTML panel generators (input, attention, heads, FFN, lens, …)
├── app.py             # static slider UI (port 7860)
├── llm_xray.py        # terminal walkthrough
├── tests/test_core.py # correctness + security smoke tests
├── .github/workflows/ # CI: runs the tests on every push / PR
├── requirements.txt
├── docs/
│   ├── make_figures.py   # regenerates the figures below
│   └── fig-*.png
├── LICENSE
└── README.md
```

Regenerate the README figures with `./.venv/bin/python docs/make_figures.py`.

---

## Credits

- Inspiration / fact-check: [*How LLMs actually work*](https://www.0xkato.xyz/how-llms-actually-work)
- Model: [GPT-2](https://huggingface.co/gpt2) via 🤗 [Transformers](https://github.com/huggingface/transformers)
- UI: [Gradio](https://www.gradio.app/) · type: [IBM Plex](https://www.ibm.com/plex/)

Licensed under the [MIT License](LICENSE).
