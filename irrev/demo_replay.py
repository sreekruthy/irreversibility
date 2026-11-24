"""Demo for the presentation: replay one agent task pair (reversible and irreversible version) from the
saved logs, step by step, re-executing every tool call in the sandbox. No GPU and no model needed.

The model replies are read from results/<model>/agent_neutral.jsonl; the sandbox re-runs each call from the
task's initial state, so the observations shown are recomputed live and checked against the logged ones.

Usage:
  python -m irrev.demo_replay                                   # default pair, Qwen
  python -m irrev.demo_replay --model qwen7b --pair task-game-F4_rewrite_file --pause
  python -m irrev.demo_replay --list                            # list available pairs with a one-line outcome
"""
from __future__ import annotations

import argparse
import json
import textwrap

from .analyze_behaviour import episode_measures
from .labeller import label as rlabel
from .sandbox import Sandbox

STATE_CHANGING = {"write_file", "edit_file", "backup_file", "restore_backup", "delete_backup", "move_to_trash",
                  "delete_file", "restore_from_trash", "purge_from_trash", "empty_trash", "recover_emptied_bin",
                  "send_email", "unsend_email", "schedule_email", "cancel_scheduled", "transfer_money", "cancel_transfer"}
COL = {"reversible": "\033[32m", "partial": "\033[33m", "irreversible": "\033[31m", None: ""}
END = "\033[0m"


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def outcome(ep):
    m = episode_measures(ep)
    return (f"irreversible action: {'YES' if m['took_irreversible'] else 'no'} | strict caution: "
            f"{'yes' if m['strict'] else 'no'} | task success: {'yes' if m['success'] else 'no'} | steps: {ep['n_steps']}")


def replay(task, ep, pause, color):
    print("=" * 100)
    print(f"{task['version'].upper()} VERSION   ({task['id']})")
    print("=" * 100)
    print(textwrap.fill("TASK: " + task["task_text"], 100))
    print("\nSTATE SHOWN TO THE AGENT:\n" + task["state_text"] + "\n")
    env = Sandbox(task["init_state"])
    mismatches = 0
    for t in ep["trace"]:
        if pause:
            input("  [Enter for the next step] ")
        print(f"step {t['step'] + 1}  model output: {t['raw'][:200]}")
        if not t.get("parse_ok"):
            print("         -> could not be parsed as a tool call (counted as a parse failure)\n")
            continue
        tool, args = t["tool"], t["args"]
        lab = None
        if tool in STATE_CHANGING:
            try:
                lab = rlabel(env.s, tool, args, 3).label
            except Exception:
                lab = None
        obs = env.step(tool, args)
        mismatches += obs != t["obs"]
        tag = f"  [{lab}]" if lab else ""
        c = COL[lab] if color else ""
        print(f"         -> {c}{tool}{tag}{END if c else ''}: {obs[:160]}\n")
    print("OUTCOME: " + outcome(ep))
    if mismatches:
        print(f"(warning: {mismatches} recomputed observation(s) differ from the log)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b", choices=["qwen7b", "llama8b"])
    ap.add_argument("--pair", default="task-game-F4_rewrite_file")
    ap.add_argument("--condition", default="neutral", choices=["neutral", "careful"])
    ap.add_argument("--pause", action="store_true", help="wait for Enter between steps (live demo)")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    tasks = {t["id"]: t for t in load("data/tasks.jsonl")}
    eps = load(f"results/{a.model}/agent_{a.condition}.jsonl")
    if a.list:
        for e in sorted(eps, key=lambda e: e["id"]):
            print(f"{e['pair_id']:45s} {e['version']:13s} {outcome(e)}")
        return
    by = {e["version"]: e for e in eps if e["pair_id"] == a.pair}
    if len(by) != 2:
        raise SystemExit(f"pair {a.pair} not found in results/{a.model}/agent_{a.condition}.jsonl (use --list)")
    print(f"\nModel: {a.model}   prompt: {a.condition}   pair: {a.pair}\n")
    for ver in ("reversible", "irreversible"):
        replay(tasks[by[ver]["id"]], by[ver], a.pause, not a.no_color)
    print("SUMMARY")
    for ver in ("reversible", "irreversible"):
        print(f"  {ver:13s} {outcome(by[ver])}")


if __name__ == "__main__":
    main()
