"""Robustness check of ScienceWorld labels: re-search every item whose search hit the node budget
(capped, label partial/irreversible) with a 5x larger budget and twice as many candidates per node.
Writes data/scienceworld_recheck.w{k}.jsonl with old and new labels; nothing in the dataset is changed.
Usage: python -m irrev.scienceworld_recheck WORKER N_WORKERS
"""
import json, os, sys, time
from . import scienceworld_label as S

S.MAX_NODES = 200
S.TOP_K = 24


def main(worker=0, n=1):
    rows = [json.loads(l) for l in open("data/probe_scienceworld.jsonl")]
    targets = [r for r in rows if r["label"] != "reversible" and r["search_capped"]]
    targets = [r for i, r in enumerate(targets) if i % n == worker]
    out = f"data/scienceworld_recheck.w{worker}.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    E = S.Env()
    f = open(out, "a")
    for r in targets:
        if r["id"] in done:
            continue
        E.load(r["scenario_id"][3:], r["variation"])
        act = json.loads(r["action_text"])["action"]
        t0 = time.time()
        labels, obs, nodes, best, capped, info = S.label_item(E, r["prefix"], act)
        rec = {"id": r["id"], "old": r["label"], "new": labels["3"], "nodes": nodes, "capped": capped,
               "secs": round(time.time() - t0, 1), "action": act}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(rec, flush=True)
    print("RECHECK DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
