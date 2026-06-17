---
title: LLM X-Ray
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: live.py
short_description: Watch a real GPT-2 think — live, token by token, end to end.
license: mit
pinned: false
---

# 🔬 LLM X-Ray

Watch a **real** GPT-2 generate text one token at a time, with every stage of the
forward pass exposed — tokens, embeddings, attention, the feed-forward network,
the prediction forming layer by layer, and sampling. Every number comes from the
actual model. A **detail** control scales it from the full forward pass
(*detailed*, the default) down to just *"watch it write"* (*simple*).

Press **Generate** and pick a detail level.

Source & full write-up: <https://github.com/swarina/llm-xray>

> Auto-synced from GitHub `main` by a GitHub Actions workflow — do not edit here.
