"""RQ1 step 2: one forward pass per item; save hidden states at every layer.

For each item the agent prompt (system + task + state) is followed by the action
as the assistant's own reply. We record, at every layer (0 = embeddings, then
each decoder layer's output = residual stream), two vectors:
  last : hidden state at the last token of the action
  mean : mean over the action's tokens
Captured with our own forward hooks, so every layer is measured the same way
(transformers' output_hidden_states replaces the last entry with the final-norm
output; we do not use it).

Output (resumable, sharded): <out>/shard_XXXX.npz with arrays last [n, L+1, d],
mean [n, L+1, d] (float16) and ids; plus <out>/meta.json and <out>/spans.jsonl.

Usage (Kaggle): python -m irrev.extract --model Qwen/Qwen2.5-7B-Instruct \
                  --data data/all.jsonl --out results/qwen7b/hidden
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import chat_prompt, system_role_supported, agent_messages, decoder_layers, first_device, hf_token, load_jsonl, load_model, run_info, set_seed
from .runguard import RunGuard


def action_span(tok, item):
    """Token ids of prompt+action, and the index range of the action tokens."""
    msgs = agent_messages(item)
    prefix = chat_prompt(tok, msgs)
    full = prefix + item["action_text"]
    ids_prefix = tok(prefix, add_special_tokens=False)["input_ids"]
    ids_full = tok(full, add_special_tokens=False)["input_ids"]
    start = len(ids_prefix)
    ok = ids_full[:start] == ids_prefix
    if not ok:  # tokenizer merged across the boundary: back off to the longest common prefix
        start = next(i for i, (a, b) in enumerate(zip(ids_full, ids_prefix)) if a != b)
    return ids_full, start, len(ids_full), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N items")
    a = ap.parse_args()
    cfg = {"script": "extract", "model": a.model, "data": a.data, "shard": a.shard, "limit": a.limit,
           "n_items": len(load_jsonl(a.data))}
    with RunGuard(a.out, cfg) as g:
        if g.mode != "done":
            run(a)


def run(a):
    set_seed(0)
    import torch

    items = load_jsonl(a.data)
    if a.limit:
        items = items[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    tok, model, dtype = load_model(a.model, token=hf_token())
    layers = decoder_layers(model)
    L = len(layers)

    cur = {}

    def save_vec(idx, hs):
        h = hs[0] if isinstance(hs, (tuple, list)) else hs        # transformers 5: tensor; 4.x: tuple
        s, e = cur["span"]
        x = h[0, s:e, :].float()
        cur["last"][idx] = x[-1].cpu().numpy()
        cur["mean"][idx] = x.mean(0).cpu().numpy()

    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: save_vec(0, o))]
    for li, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: save_vec(li + 1, o)))

    meta = {"run": run_info(a.model, dtype), "n_layers_plus_embed": L + 1,
            "hidden_size": model.config.hidden_size, "system_role_supported": system_role_supported(tok), "data": a.data, "n_items": len(items),
            "positions": ["last action token", "mean over action tokens"]}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    dev = first_device(model)
    t0 = time.time()
    for s0 in range(0, len(items), a.shard):
        path = os.path.join(a.out, f"shard_{s0 // a.shard:04d}.npz")
        if os.path.exists(path):
            continue
        chunk = items[s0: s0 + a.shard]
        last = np.zeros((len(chunk), L + 1, model.config.hidden_size), np.float16)
        mean = np.zeros_like(last)
        bad = 0
        spans = []
        for j, it in enumerate(chunk):
            ids, s, e, ok = action_span(tok, it)
            cur.update(span=(s, e), last=last[j], mean=mean[j])
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev))
            if not np.isfinite(last[j]).all():
                bad += 1
            spans.append({"id": it["id"], "n_tokens": len(ids), "action_start": s, "action_end": e,
                          "boundary_clean": ok, "action_decoded": tok.decode(ids[s:e])})
        tmp = path[:-4] + ".tmp.npz"                      # write then rename: never a half shard
        np.savez(tmp, last=last, mean=mean, ids=np.array([it["id"] for it in chunk]))
        os.replace(tmp, path)
        with open(os.path.join(a.out, "spans.jsonl"), "a") as sf:   # after the shard exists
            for r in spans:
                sf.write(json.dumps(r) + "\n")
        done = min(s0 + a.shard, len(items))
        print(f"{done}/{len(items)} items | {time.time() - t0:.0f}s | non-finite vectors in shard: {bad}", flush=True)
    for h in hooks:
        h.remove()
    print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
