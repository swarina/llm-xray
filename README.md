# llm x-ray

Watch a real GPT-2 think, one stage at a time. This is the runnable companion to
the article ["how LLMs actually work"](https://www.0xkato.xyz/how-llms-actually-work):
instead of describing the pipeline, it prints the **actual internals** of a
pretrained model for whatever sentence you type. Nothing is faked — every number
comes out of GPT-2.

## Setup

Already done if you ran the installer, but to redo it:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Three ways to look at it

```bash
./.venv/bin/python llm_xray.py     # terminal: prints all 5 stages + saves a heatmap PNG
./.venv/bin/python app.py          # browser UI at :7860 - sliders + 4 static panels
./.venv/bin/python live.py         # browser UI at :7861 - watch it generate token by token, live
```

First run downloads GPT-2 (~500 MB) and caches it; later runs are fast.

`live.py` is the real-time one: press **generate** and watch the autoregressive
loop unfold - at each step you see the candidate words it's weighing, what it
attended to, and how confident it is, before it commits to a token and feeds it
back in. Use the "seconds per step" slider to slow it down enough to follow.

CLI examples:

```bash
./.venv/bin/python llm_xray.py --text "the capital of france is" --layer 10 --head 7
./.venv/bin/python llm_xray.py --text "i love programming in" --generate 12
./.venv/bin/python llm_xray.py --temperature 0.3
```

## What each stage maps to in the article

| Stage in the script | Article section | What you actually see |
|---|---|---|
| 0 · Tokenization | "models read integer ids" | Your text → subword pieces + ids; why `strawberry` is hard to spell |
| 1 · Embeddings | "each id indexes a long vector" | The real 768-dim embedding matrix; the vector for your last token |
| 2 · Attention | "Q·K, softmax, causal mask" | A real attention matrix for any layer/head + a saved heatmap PNG |
| 3 · Residual stream | "each block adds to the vector" | The last token's vector norm growing block by block |
| 4 · Logits → next token | "last vector → 1 logit per word" | Top-k next words, and how temperature reshapes them (real softmax) |
| 5 · Generation | "repeat the loop" | A greedy continuation |

## Where GPT-2 differs from the article's "modern" model

GPT-2 (2019) is the most inspectable open model, but it predates a few things the
article lists as current best practice. The script calls these out as you hit them:

- **Positional encoding** — GPT-2 uses *learned absolute* position embeddings
  (`wpe`). Modern models use **RoPE**. (Stage 1 prints this.)
- **Normalization** — GPT-2 uses LayerNorm; modern models use **RMSNorm**.
- **Attention** — GPT-2 uses full multi-head attention; modern models add
  **Grouped-Query Attention (GQA)** and a **KV cache** for speed.
- **FFN activation** — GPT-2 uses GELU; modern models use **SwiGLU**.

The *shape* of the pipeline is identical, which is the whole point of the article.

## Knobs

`--text` prompt · `--model gpt2|gpt2-medium|gpt2-large` · `--layer N` `--head N`
(which attention to inspect) · `--temperature` · `--topk` · `--generate N` ·
`--no-plot`.
