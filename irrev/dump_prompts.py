"""Write every rendered prompt to disk (brief, Section 5: "save every prompt and every model output").

For each item: the probe prompt (agent prompt + action, exactly what extract.py feeds the model)
and the stated-answer prompts (multiple-choice and free-text). For each task episode: the
initial agent prompt (neutral and careful); later steps are the initial prompt plus the logged
model outputs and observations in the agent trace, so they are fully reconstructable.
Needs only the tokenizer (no GPU).

Usage: python -m irrev.dump_prompts --model Qwen/Qwen2.5-7B-Instruct --out results/qwen7b/prompts.jsonl.gz \
          --data data/probe_sandbox.jsonl data/probe_sandbox_extra.jsonl data/probe_scienceworld.jsonl \
          --tasks data/tasks.jsonl data/tasks_shortcut_stated.jsonl
"""
import argparse
import gzip
import json

from transformers import AutoTokenizer

from .common import agent_messages, chat_prompt, hf_token, load_jsonl, stated_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*", default=[])
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, token=hf_token())
    n = 0
    with gzip.open(a.out, "wt") as f:
        for p in a.data + [t for t in a.tasks if "shortcut" in t]:
            for it in load_jsonl(p):
                f.write(json.dumps({"id": it["id"], "kind": "probe", "file": p,
                                    "prompt": chat_prompt(tok, agent_messages(it)) + it["action_text"]}) + "\n")
                f.write(json.dumps({"id": it["id"], "kind": "stated_options", "file": p,
                                    "prompt": chat_prompt(tok, stated_messages(it))}) + "\n")
                f.write(json.dumps({"id": it["id"], "kind": "stated_free", "file": p,
                                    "prompt": chat_prompt(tok, stated_messages(it, free_text=True))}) + "\n")
                n += 3
        for p in [t for t in a.tasks if "shortcut" not in t]:
            for t in load_jsonl(p):
                for careful in (False, True):
                    f.write(json.dumps({"id": t["id"], "kind": "agent_initial_careful" if careful else "agent_initial_neutral",
                                        "file": p, "prompt": chat_prompt(tok, agent_messages(t, careful=careful))}) + "\n")
                    n += 1
    print(f"PROMPTS DONE: {n} prompts -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
