"""RQ2: ask the model directly whether the action can be undone.

Two measurements per item, both greedy / deterministic:
  options   : probability of the next token being A / B / C after the
              multiple-choice prompt (robust; no parsing)
  free_text : a short answer ending in 'Answer: <label>', parsed by regex.
              Unparseable answers are kept and counted, never dropped.

Usage: python -m irrev.stated --model Qwen/Qwen2.5-7B-Instruct --data data/all.jsonl \
         --out results/qwen7b/stated.jsonl [--no-free-text] [--batch 8]
"""
from __future__ import annotations

import argparse
import os
import re
import time

from .common import (chat_prompt, system_role_supported, STATED_OPTIONS, first_device, hf_token, load_jsonl, load_model, run_info,
                     set_seed, stated_messages, write_jsonl)
from .runguard import RunGuard, read_jsonl_tolerant

PARSE = re.compile(r"answer\s*:\s*\**\s*(partially reversible|partial|irreversible|reversible)", re.I)


def parse_free(text):
    m = PARSE.findall(text)
    if not m:
        return None
    v = m[-1].lower()
    return {"partially reversible": "partial", "partial": "partial",
            "irreversible": "irreversible", "reversible": "reversible"}[v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--no-free-text", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = {"script": "stated", "model": a.model, "data": a.data, "batch": a.batch, "limit": a.limit,
           "max_new_tokens": a.max_new_tokens, "free_text": not a.no_free_text}
    with RunGuard(a.out, cfg, related=(a.out.replace(".jsonl", ".run.jsonl"),)) as g:
        if g.mode != "done":
            run(a)


def run(a):
    set_seed(0)
    import torch

    items = load_jsonl(a.data)
    if a.limit:
        items = items[: a.limit]
    done = {r["id"] for r in read_jsonl_tolerant(a.out)}
    todo = [it for it in items if it["id"] not in done]
    tok, model, dtype = load_model(a.model, token=hf_token())
    tok.padding_side = "left"
    dev = first_device(model)
    letter_ids = {}
    for L in STATED_OPTIONS:
        ids = tok(L, add_special_tokens=False)["input_ids"]
        letter_ids[L] = ids[0]
    write_jsonl(a.out.replace(".jsonl", ".run.jsonl"), [{"run": run_info(a.model, dtype), "letter_ids": letter_ids, "system_role_supported": system_role_supported(tok),
                                                        "resumed_with_done": len(done)}], mode="a")
    t0 = time.time()
    for b0 in range(0, len(todo), a.batch):
        batch = todo[b0: b0 + a.batch]
        # --- option probabilities ---
        prompts = [chat_prompt(tok, stated_messages(it)) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits, -1)
        rows = []
        for j, it in enumerate(batch):
            p = {STATED_OPTIONS[L]: float(probs[j, i]) for L, i in letter_ids.items()}
            mass = sum(p.values())
            pn = {k: v / mass for k, v in p.items()} if mass > 0 else p
            rows.append({"id": it["id"], "label": it["label"], "option_probs": pn, "option_mass": mass,
                         "stated_option": max(pn, key=pn.get)})
        # --- free text ---
        if not a.no_free_text:
            prompts = [chat_prompt(tok, stated_messages(it, free_text=True)) for it in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j, r in enumerate(rows):
                text = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                r["free_text"] = text
                r["stated_free"] = parse_free(text)          # None = parse failure (counted)
        write_jsonl(a.out, rows, mode="a")
        if os.environ.get("IRREV_TEST_FAIL_AFTER_BATCH") and b0 >= a.batch * int(os.environ["IRREV_TEST_FAIL_AFTER_BATCH"]):
            raise RuntimeError("test-only injected failure")      # used by tests/test_resume.sh only
        print(f"{min(b0 + a.batch, len(todo))}/{len(todo)} | {time.time() - t0:.0f}s", flush=True)
    print("STATED DONE", flush=True)


if __name__ == "__main__":
    main()
