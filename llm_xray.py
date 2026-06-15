"""
llm_xray.py - watch a real GPT-2 think, one stage at a time.

This walks a single forward pass through GPT-2 and prints the ACTUAL internals
the "how LLMs work" article describes - tokens, embeddings, attention, the
residual stream, and the logits that become the next word. Nothing here is faked:
every number comes out of the real pretrained model.

Run:
    ./.venv/bin/python llm_xray.py
    ./.venv/bin/python llm_xray.py --text "the capital of france is" --layer 10 --head 7
    ./.venv/bin/python llm_xray.py --text "i love programming in" --generate 12
    ./.venv/bin/python llm_xray.py --model gpt2-medium --temperature 0.3

Each section header says which article claim it is showing you.
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def softmax_topk(logits, temperature, k):
    """Real next-token math: divide logits by temperature, softmax, take top-k."""
    probs = F.softmax(logits / temperature, dim=-1)
    top = torch.topk(probs, k)
    return top.values, top.indices


def main():
    ap = argparse.ArgumentParser(description="X-ray a real GPT-2 forward pass.")
    ap.add_argument("--text", default="the cat sat on the",
                    help="prompt to run through the model")
    ap.add_argument("--model", default="gpt2",
                    help="gpt2 | gpt2-medium | gpt2-large (bigger = slower download)")
    ap.add_argument("--layer", type=int, default=None,
                    help="which transformer block's attention to inspect (default: last)")
    ap.add_argument("--head", type=int, default=0, help="which attention head")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="sampling temperature for the logits demo")
    ap.add_argument("--topk", type=int, default=8, help="how many next-token candidates to show")
    ap.add_argument("--generate", type=int, default=0,
                    help="if >0, greedily generate this many continuation tokens")
    ap.add_argument("--no-plot", action="store_true", help="skip saving the attention heatmap PNG")
    args = ap.parse_args()

    torch.manual_seed(0)

    print(f"loading {args.model} (downloads once, then cached)...")
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(
        args.model, output_attentions=True, output_hidden_states=True
    )
    model.eval()

    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    n_embd = model.config.n_embd
    layer = args.layer if args.layer is not None else n_layers - 1
    layer = max(0, min(layer, n_layers - 1))
    head = max(0, min(args.head, n_heads - 1))

    # ----------------------------------------------------------------------
    rule("STAGE 0 - TOKENIZATION   (article: 'models read integer ids, not text')")
    # ----------------------------------------------------------------------
    ids = tok(args.text, return_tensors="pt").input_ids
    id_list = ids[0].tolist()
    pieces = tok.convert_ids_to_tokens(id_list)
    print(f"text:   {args.text!r}")
    print(f"tokens: {len(id_list)}  (vocab size = {model.config.vocab_size:,})")
    print(f"{'#':>4}  {'id':>6}  {'piece':<14}  decoded")
    for i, (tid, pc) in enumerate(zip(id_list, pieces)):
        # GPT-2 uses byte-level BPE; a leading space is shown as the glyph 'Ġ'.
        print(f"{i:>4}  {tid:>6}  {pc:<14}  {tok.decode([tid])!r}")

    # Why "how many r's in strawberry?" is hard: the word is not one unit.
    berry = tok.convert_ids_to_tokens(tok("strawberry").input_ids)
    print(f"\n'strawberry' tokenizes to {berry} - the model never sees the letters,")
    print("which is exactly why letter-counting trips it up.")

    # ----------------------------------------------------------------------
    rule("STAGE 1 - EMBEDDINGS   (article: 'each id indexes a row, a long vector')")
    # ----------------------------------------------------------------------
    wte = model.transformer.wte.weight        # token embedding matrix
    wpe = model.transformer.wpe.weight        # position embedding matrix
    print(f"token embedding matrix wte: {tuple(wte.shape)}  -> {n_embd} numbers per token")
    last_id = id_list[-1]
    vec = wte[last_id]
    print(f"embedding for last token {tok.decode([last_id])!r} (first 8 of {n_embd} dims):")
    print("  " + "  ".join(f"{x:+.3f}" for x in vec[:8].tolist()))
    print(f"\nNOTE: GPT-2 adds a LEARNED position embedding wpe {tuple(wpe.shape)}.")
    print("Modern models (LLaMA etc.) swapped this for RoPE - the article's main")
    print("divergence from what you're looking at here.")

    # ----------------------------------------------------------------------
    rule(f"STAGE 2 - ATTENTION   layer {layer}, head {head}   (article: Q*K, softmax, causal mask)")
    # ----------------------------------------------------------------------
    with torch.no_grad():
        out = model(ids)
    # attentions: tuple over layers, each (batch, heads, seq, seq), already softmaxed.
    attn = out.attentions[layer][0, head]      # (seq, seq)
    seq = attn.shape[0]
    short = [tok.decode([t]).strip()[:6] or "_" for t in id_list]
    print("each row = a token choosing which earlier tokens to read (rows sum to 1):")
    header = "         " + " ".join(f"{s:>6}" for s in short)
    print(header)
    for i in range(seq):
        cells = " ".join(f"{attn[i, j]:6.2f}" if j <= i else f"{'.':>6}" for j in range(seq))
        print(f"{short[i]:>7}  {cells}")
    print(f"\nupper-right is blank -> causal mask (no peeking ahead).")
    print(f"row sums (should all be ~1.00): {[round(float(attn[i].sum()), 2) for i in range(seq)]}")
    print(f"last token attends most to: "
          f"{short[int(attn[-1].argmax())]!r} (weight {float(attn[-1].max()):.2f})")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(0.7 * seq + 2, 0.7 * seq + 1))
            im = ax.imshow(attn.numpy(), cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(seq)); ax.set_xticklabels(short, rotation=45, ha="right")
            ax.set_yticks(range(seq)); ax.set_yticklabels(short)
            ax.set_title(f"GPT-2 attention - layer {layer}, head {head}")
            ax.set_xlabel("attended-to (key)"); ax.set_ylabel("attending (query)")
            fig.colorbar(im, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fname = f"attention_L{layer}_H{head}.png"
            fig.savefig(fname, dpi=130)
            print(f"\nsaved heatmap -> {fname}")
        except Exception as e:
            print(f"(plot skipped: {e})")

    # ----------------------------------------------------------------------
    rule("STAGE 3 - RESIDUAL STREAM   (article: each block ADDS to the vector)")
    # ----------------------------------------------------------------------
    # hidden_states: tuple of (n_layers + 1), each (batch, seq, n_embd).
    # The last token's vector accumulates information layer by layer.
    print("L2 norm of the LAST token's vector as it flows up the residual stream:")
    print(f"{'after':>10}  {'norm':>8}")
    for li, hs in enumerate(out.hidden_states):
        label = "embedding" if li == 0 else f"block {li-1}"
        print(f"{label:>10}  {float(hs[0, -1].norm()):>8.2f}")
    print("the vector is never replaced - blocks add to it, so early info has a")
    print("direct additive path all the way to the top.")

    # ----------------------------------------------------------------------
    rule("STAGE 4 - LOGITS -> NEXT TOKEN   (article: last vector -> 1 logit per word)")
    # ----------------------------------------------------------------------
    logits = out.logits[0, -1]                 # (vocab,)
    print(f"final logits vector: {tuple(logits.shape)} (one score per vocabulary word)\n")
    for temp in (0.2, args.temperature, 1.5):
        vals, idx = softmax_topk(logits, temp, args.topk)
        tag = " <- your --temperature" if abs(temp - args.temperature) < 1e-9 else ""
        print(f"temperature = {temp}{tag}")
        for p, t in zip(vals.tolist(), idx.tolist()):
            bar = "#" * max(1, round(p * 40))
            print(f"    {tok.decode([t])!r:<14} {p*100:5.1f}%  {bar}")
        print()
    print("low temperature -> peaky and safe.  high -> flat and surprising.")

    # one real sample at the chosen temperature
    probs = F.softmax(logits / args.temperature, dim=-1)
    sampled = torch.multinomial(probs, 1).item()
    print(f"\nsampled one token at T={args.temperature}: "
          f"{args.text!r} + {tok.decode([sampled])!r}")

    # ----------------------------------------------------------------------
    if args.generate > 0:
        rule(f"STAGE 5 - GENERATION   ({args.generate} greedy tokens, the loop repeated)")
        with torch.no_grad():
            gen = model.generate(
                ids, max_new_tokens=args.generate, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        print(tok.decode(gen[0], skip_special_tokens=True))

    print("\ndone. re-run with --layer / --head / --text / --temperature to explore.")


if __name__ == "__main__":
    main()
