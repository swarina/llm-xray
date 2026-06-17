# Deploying to Hugging Face Spaces (free)

The live app (`live.py`) runs on a **free CPU Space** (2 vCPU, 16 GB RAM) — plenty
for GPT-2. First build takes a few minutes (it installs PyTorch and downloads
GPT-2); after that it's cached.

## What the Space needs

Only the files the live app uses:

```
live.py  core.py  panels.py  styles.py  requirements.txt  README.md
```

…plus a `README.md` whose **YAML front-matter** tells Spaces how to run it:

```yaml
---
title: LLM X-Ray
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: live.py
pinned: false
license: mit
---
```

Two lines matter most:
- `app_file: live.py` — Spaces defaults to `app.py`, which here is the *other* tool.
- `python_version: "3.11"` — Gradio 4.x pulls in `pydub`, which imports the stdlib
  `audioop` module **removed in Python 3.13**. Without this pin the Space builds on
  3.13 and crashes on import. `live.py` binds `0.0.0.0` automatically when it detects
  the `SPACE_ID` env var, so no launch flags are needed.

## Option A — web UI (easiest)

1. Create a free account at <https://huggingface.co/join> (you do this — I can't
   create accounts).
2. Go to <https://huggingface.co/new-space> → choose **Gradio** as the SDK, give it
   a name (e.g. `llm-xray`), pick **CPU basic (free)**.
3. The new Space comes with a `README.md`. Edit its front-matter so
   `app_file: live.py` (and bump `sdk_version` to `4.44.1`).
4. Upload `live.py`, `core.py`, `panels.py`, `styles.py`, `requirements.txt`
   (the "Files" tab → "Add file" → upload).
5. It builds automatically and goes live at
   `https://huggingface.co/spaces/<you>/llm-xray`.

## Option B — git push

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                         # paste a token from hf.co/settings/tokens

huggingface-cli repo create llm-xray --repo-type space --space-sdk gradio
git clone https://huggingface.co/spaces/<you>/llm-xray hf-space

cp live.py core.py panels.py styles.py requirements.txt hf-space/
# create hf-space/README.md with the front-matter block above
cd hf-space && git add -A && git commit -m "deploy llm-xray" && git push
```

## Notes

- **Public by default.** Free Spaces are public; you can set it private in Settings
  (private free Spaces are allowed, with limits).
- **Sleeps when idle.** Free Spaces pause after ~48 h of no traffic and wake on the
  next visit (a short cold start).
- **Speed.** Generation is CPU-only on the free tier — fine for GPT-2, just a little
  slower than local. Lower the "seconds per word" slider to taste.
- **Faster builds (optional).** To pull the smaller CPU-only PyTorch wheel, you can
  pin in `requirements.txt`:
  `torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cpu`
