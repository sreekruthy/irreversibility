"""Re-search every non-reversible ScienceWorld item with the current labeller.

A 'reversible' label is a certificate (a restoring sequence was found and verified
against the null state), so it cannot be a false positive. 'partial' and
'irreversible' can be search misses, so those are re-searched after any fix to the
candidate generator. The old label is kept in `label_v1` for the record.
Usage: python -m irrev.scienceworld_relabel data/scienceworld.jsonl
"""
import json, os, sys, time
from .scienceworld_label import Env, label_item


def main(path, worker=0, n_workers=1, target="nonreversible"):
    rows = [json.loads(l) for l in open(path)]
    # de-duplicate (the sampler can draw the same state and action twice)
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"]); uniq.append(r)
    rows = uniq
    E = Env()
    changed = 0
    out_part = path + f".relabel.{target}.w{worker}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_part)} if os.path.exists(out_part) else set()
    if target == "noopfix":   # items whose baseline used the 0-tick fallback (outdoor location bug)
        targets = [r for r in rows if r.get("search_info", {}).get("noop") == "look around"]
    elif target == "pass3":   # fixed candidate ranking + outdoor no-op: every non-certified label
        targets = [r for r in rows if r["label"] != "reversible" or r.get("search_info", {}).get("noop") == "look around"]
    else:
        targets = [r for r in rows if r["label"] != "reversible"]
    targets = [r for i, r in enumerate(targets) if i % n_workers == worker]
    pf = open(out_part, "a")
    for r in targets:
        if r["id"] in done:
            continue
        task = r["scenario_id"][3:]
        E.load(task, r["variation"])
        act = json.loads(r["action_text"])["action"]
        t0 = time.time()
        labels, obs, nodes, best, capped, info = label_item(E, r["prefix"], act)
        key = "label_v2" if "label_v1" in r else "label_v1"
        r.update(**{key: r["label"], "labels_by_horizon_" + key[-2:]: r["labels_by_horizon"]}, label=labels["3"],
                 labels_by_horizon=labels, search_nodes=nodes, search_capped=capped, restore_depth=best,
                 search_info=info, relabelled=True)
        changed += r["label"] != r[key]
        pf.write(json.dumps(r) + "\n"); pf.flush()
        print(f"{r['id']} {r[key]} -> {r['label']} nodes={nodes} {time.time()-t0:.1f}s", flush=True)
    print("RELABEL PART DONE changed", changed, flush=True)
    os._exit(0)


def merge(path, n_workers, target="nonreversible"):
    rows = [json.loads(l) for l in open(path)]
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"]); uniq.append(r)
    upd = {}
    for w in range(n_workers):
        p = path + f".relabel.{target}.w{w}.jsonl"
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l); upd[r["id"]] = r
    out = [upd.get(r["id"], r) for r in uniq]
    with open(path + ".tmp", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    print("merged", len(out), "rows,", len(upd), "relabelled")


if __name__ == "__main__":
    tgt = sys.argv[4] if len(sys.argv) > 4 else "nonreversible"
    if sys.argv[2] == "merge":
        merge(sys.argv[1], int(sys.argv[3]), tgt)
    else:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), tgt)
