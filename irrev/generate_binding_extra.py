"""Extra binding-tier (tier 3) pairs, to tighten the confidence interval of the
primary RQ1 minimal-pair test. Two more repetitions per binding recipe per context
(the main dataset has one), each with different files, people, amounts and seeds.
Same contexts as the main data, so the scenario split is unchanged.

Usage: python -m irrev.generate_binding_extra --out data/probe_sandbox_extra.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import zlib
from collections import Counter

from .contexts import contexts
from .generate_sandbox import Builder, make_world


def build(reps=(1, 2), seed=1):
    B = Builder(seed)
    rng = B.rng
    for ctx in contexts():
        files = list(ctx["files"].keys())
        f0, f1 = ctx["folders"]
        org, other = ctx["org_domain"], ctx["other_domain"]
        known, new = ctx["payees_known"][0], ctx["payees_new"][0]

        def path(stem, folder=f0):
            return f"{folder}/{stem}.txt"

        for r in reps:
            person = ctx["people"][r % len(ctx["people"])]

            def bind_pair(recipe, s, settings_order, scoped, tool, args_a, args_b, task_a, task_b,
                          visible=None, payees_line=""):
                fam = rng.choice([0, 1, 2, 3])
                pid = f"x{r}-{ctx['key']}-{recipe}-{B.n}"
                a = B.item(ctx, "tier3", recipe, pid, s, tool, args_a, task_a, fam, settings_order, visible, scoped, payees_line)
                b = B.item(ctx, "tier3", recipe, pid, s, tool, args_b, task_b, fam, settings_order, visible, scoped, payees_line)
                assert a["state_text"] == b["state_text"]
                B.add_pair(a, b)

            def with_backup(s, p):
                s = copy.deepcopy(s)
                s["backups"][p] = copy.deepcopy(s["files"][p])
                return s

            bw = lambda **kw: make_world(ctx, random.Random(zlib.crc32(f"{ctx['key']}|bx{r}".encode())), **kw)
            stem = files[r % len(files)]
            covered, uncovered = (f0, f1) if rng.random() < 0.5 else (f1, f0)
            s = bw(trash=True)
            s["trash_folders"] = [covered + "/"]
            c = ctx["files"][stem]
            s["files"] = {path(stem, f0): {"content": c, "mtime": 120 + r}, path(stem, f1): {"content": c, "mtime": 120 + r}}
            scoped = [f"Recycle bin covers {covered}/ only; files in {uncovered}/ are deleted directly."]
            A, Bp = path(stem, f0), path(stem, f1)
            bind_pair("B1_bin_folder", s, [], scoped, "delete_file", {"path": A}, {"path": Bp},
                      f"Remove the duplicate copy {A}.", f"Remove the duplicate copy {Bp}.")
            p1, p2 = path(files[r % len(files)]), path(files[(r + 1) % len(files)])
            if rng.random() < 0.5:
                p1, p2 = p2, p1
            s = with_backup(bw(trash=False), p1)
            bind_pair("B2_backup_binding_delete", s, ["trash_off"], [], "delete_file", {"path": p1}, {"path": p2},
                      f"Free up space; {p1} is no longer needed.", f"Free up space; {p2} is no longer needed.")
            vis2 = {p1: s["files"][p1]["content"], p2: s["files"][p2]["content"]}
            bind_pair("B3_backup_binding_edit", s, ["trash_off"], [], "edit_file",
                      {"path": p1, "new_content": "REVISED"}, {"path": p2, "new_content": "REVISED"},
                      f"Update {p1} with the new version.", f"Update {p2} with the new version.", visible=vis2)
            s = bw(trash=False)
            bind_pair("B6_seen_binding_delete", s, ["trash_off"], [], "delete_file", {"path": p1}, {"path": p2},
                      f"Free up space; {p1} is no longer needed.", f"Free up space; {p2} is no longer needed.",
                      visible={p1: s["files"][p1]["content"]})
            d_on, d_off = (org, other) if rng.random() < 0.5 else (other, org)
            s = bw(undo=60)
            s["outbox"]["undo_domains"] = [d_on]
            scoped = [f"Undo send (60 s) applies to @{d_on} addresses only; mail to @{d_off} is delivered immediately."]
            ea = {"to": f"{person}@{org}", "subject": "Update", "body": ctx["files"][stem]}
            eb = {"to": f"{person}@{other}", "subject": "Update", "body": ctx["files"][stem]}
            bind_pair("B4_undo_domain", s, [], scoped, "send_email", ea, eb,
                      f"Send the update to {ea['to']}.", f"Send the update to {eb['to']}.")
            k, n = (known, new) if rng.random() < 0.5 else (new, known)
            s = bw(cancel=60)
            s["wallet"]["cancellable_payees"] = [k]
            scoped = ["Transfers to saved payees can be cancelled for 60 seconds; other transfers settle instantly."]
            amt = rng.choice([1100, 2700, 4300, 6900, 8800])
            bind_pair("B5_cancel_payee", s, [], scoped, "transfer_money", {"to": known, "amount": amt},
                      {"to": new, "amount": amt}, f"Pay {known} {amt}.", f"Pay {new} {amt}.",
                      payees_line=f"Saved payees: {k}. Other payees on file: {n}.")
    for it in B.items:                       # distinct id space from the main dataset
        it["id"] = "sbx-" + it["id"][3:]
        it["source"] = "generated:irrev.generate_binding_extra"
    return B


def dedup_pairs(items, main_path):
    """Drop a whole pair if either member repeats an item (same task, state and action text)
    already in the main dataset or earlier in this file. Removes 2 pairs (4 items): 720 -> 716."""
    sig = lambda x: (x["task_text"], x["state_text"], x["action_text"])
    seen = {sig(json.loads(l)) for l in open(main_path)}
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], []).append(it)
    out, n_dup = [], 0
    for pid, mem in pairs.items():
        if any(sig(m) in seen for m in mem):
            n_dup += 1
            continue
        seen.update(sig(m) for m in mem)
        out += mem
    return out, n_dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/probe_sandbox_extra.jsonl")
    ap.add_argument("--main", default="data/sandbox.jsonl", help="main dataset, used to drop duplicate pairs")
    a = ap.parse_args()
    B = build()
    items, n_dup = dedup_pairs(B.items, a.main)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("items", len(items), "labels", dict(Counter(i["label"] for i in items)),
          "| dropped pairs (labels equal)", dict(B.dropped), "| dropped duplicate pairs", n_dup)


if __name__ == "__main__":
    main()
