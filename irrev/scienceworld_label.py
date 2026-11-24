"""Reversibility labels for ScienceWorld (state, action) pairs by bounded search.

ScienceWorld has no save/restore, so every searched node is reached by
reset + replay of the action history.

The world changes on its own (heat flows, a fountain respawns water with new
ids), so "restoring the previous state" is defined counterfactually:
after a and a restoring sequence r, the world is compared with the NULL state:
the world reached from s by doing nothing for the same number of simulation
ticks (measured, not assumed). "Doing nothing" = "go to <current room>", which
advances exactly 1 tick and changes nothing; note that ScienceWorld's own
`wait1` advances 2 ticks, so it cannot serve as the 1-tick baseline.
Ambient change (heat flow, respawning water) then cancels out.

  canonical state: object tree with ids dropped (objects named by their path of
                   names) and temperature held out as a soft field
  exact match:     canonical equal and every temperature within TEMP_TOL of null
  soft match:      canonical equal, some temperature differs  -> partial
Search space (pruned, see paper): valid actions whose verb can restore state
and which mention an object named in a, or a's original container. At most
MAX_NODES sequences are tried per item; an item with no match is labelled
irreversible, which is an upper bound on irreversibility (reported).

Usage: python -m irrev.scienceworld_label --out data/scienceworld.jsonl --worker 0 --n-workers 2
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import time
import zlib

TEMP_TOL = 0.5
MAX_NODES = 40
TOP_K = 12          # candidate actions kept per search node
MAX_PREFIX = 25
TASKS = ["boil", "melt", "freeze", "change-the-state-of-matter-of", "chemistry-mix",
         "chemistry-mix-paint-secondary-color", "chemistry-mix-paint-tertiary-color", "use-thermometer",
         "measure-melting-point-known-substance", "test-conductivity", "power-component",
         "find-living-thing", "find-non-living-thing", "grow-plant"]
ACTION_VERBS = ("open", "close", "activate", "deactivate", "pick up", "put down", "move", "pour",
                "mix", "connect", "disconnect", "eat", "dunk", "go to")
RESTORE_VERBS = ("open", "close", "activate", "deactivate", "pick up", "put down", "move", "pour",
                 "connect", "disconnect", "go to")
INVERSE = {"open": "close", "close": "open", "activate": "deactivate", "deactivate": "activate",
           "pick up": "put down", "put down": "pick up", "connect": "disconnect", "disconnect": "connect"}

print = functools.partial(print, flush=True)


def canon(tree):
    """(sorted list of (name-path, props-json), sorted list of (name-path, temperature))."""
    objs, temps = [], []

    def walk(node, path):
        p = path + "/" + str(node.get("name"))
        props = {}
        for k, v in node.items():
            if k in ("uuid", "contents", "name"):
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk == "temperatureC":
                        temps.append((p, float(vv)))
                    else:
                        props[k + "." + kk] = vv
            else:
                props[k] = v
        objs.append((p, json.dumps(props, sort_keys=True, default=str)))
        for child in (node.get("contents") or {}).values():
            walk(child, p)

    walk(tree, "")
    return sorted(objs), sorted(temps)


def compare(c, null):
    if c[0] != null[0]:
        return "none"
    if len(c[1]) == len(null[1]) and all(abs(a[1] - b[1]) <= TEMP_TOL for a, b in zip(c[1], null[1])):
        return "exact"
    return "soft"


def verb_of(a):
    for v in sorted(ACTION_VERBS, key=len, reverse=True):
        if a.startswith(v + " ") or a == v:
            return v
    return None


def objects_of(a):
    """The object(s) an action acts on, without location qualifiers or targets.

    "put down orange in inventory" -> ["orange"]   ("in X" only says where it is)
    "move orange in bowl to kitchen" -> ["orange"] ("to X" is the target)
    "connect red wire terminal 2 to anode in red light bulb" -> ["red wire terminal 2", "anode"]
      (for connect/disconnect both ends can be used to undo)
    """
    v = verb_of(a)
    rest = a[len(v):].strip() if v else a
    parts = re.split(r"\s+(?:to|into|on)\s+", rest, maxsplit=1)

    def core(p):
        p = re.split(r"\s+in\s+", p, maxsplit=1)[0]          # drop "in <location>"
        p = re.sub(r"\s+containing\s+.*$", "", p)            # drop "containing ..."
        return p.strip()
    names = [core(parts[0])]
    if v in ("connect", "disconnect") and len(parts) > 1:
        names.append(core(parts[1]))
    return [n for n in names if n]

class Env:
    def __init__(self):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv("", envStepLimit=1000)
        self.task = None

    def load(self, task, var):
        self.env.load(task, var, "", generateGoldPath=True)
        self.task = (task, var)
        return self.env.get_gold_action_sequence()

    def replay(self, actions):
        self.env.reset()
        for a in actions:
            self.env.step(a)

    def state(self):
        return canon(self.env.getObjectTree())

    def valid(self):
        return list(self.env.get_valid_action_object_combinations())


def parent_name(tree, objname):
    """Name of the container currently holding an object whose name starts with objname."""
    found = []

    def walk(node, parent):
        if str(node.get("name", "")).startswith(objname) and parent is not None:
            found.append(parent)
        for ch in (node.get("contents") or {}).values():
            walk(ch, node.get("name"))

    walk(tree, None)
    return found[0] if found else None


def moved_info(objs_before, objs_after):
    """Real name and original container of the object the action moved (by tree diff)."""
    pb = {p for p, _ in objs_before}
    pa = {p for p, _ in objs_after}
    removed = sorted(pb - pa, key=lambda p: (p.count("/"), p))
    if not removed:
        return None, None
    parts = removed[0].split("/")
    return parts[-1], (parts[-2] if len(parts) >= 3 else None)


def referents(E, name):
    """All unambiguous names ScienceWorld uses for the object called `name`."""
    if not name:
        return []
    lut = E.env.get_all_object_ids_types_referents_LUTJSON()
    lut = json.loads(lut) if isinstance(lut, str) else lut
    count = {}
    for v in lut.values():
        for r in v.get("referents", []):
            count[r] = count.get(r, 0) + 1
    out = set()
    for v in lut.values():
        refs = v.get("referents", [])
        if name in refs:
            # keep ambiguous multi-word names too (ScienceWorld qualifies them in actions,
            # e.g. "cup containing nothing in inventory"); drop bare one-word ones like "cup"
            out.update(r for r in refs if count.get(r, 0) == 1 or " " in r)
    out.add(name)
    return sorted(out, key=len, reverse=True)


def _mentions(x, names):
    return any(n and re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", x) for n in names)


def candidates(valid, a, origin, moved=None, extra_names=(), origin_names=()):
    """Restoring actions to try, best first.

    Kept: actions that mention the moved/acted-on object (any restoring verb), plus
    'go to', 'open' and 'close' actions on the original container/room (to walk back
    or re-open a container). Actions that mention only the origin (e.g. moving some
    OTHER object into it) are not kept: they cannot undo this action and would crowd
    out the right candidates.
    """
    v = verb_of(a)
    names = [n for n in objects_of(a) + [moved] + list(extra_names) if n]
    # a terminal / electrode names its parent object in the undo command ("disconnect orange wire")
    names += [re.sub(r"\s+(terminal \d+|anode|cathode)$", "", n) for n in names]
    names = sorted(set(n for n in names if n and n != "agent"), key=len, reverse=True)
    onames = [n for n in [origin] + list(origin_names) if n]
    inv = INVERSE.get(v)
    elec = v in ("connect", "disconnect")
    agent_moved = (moved == "agent") or v == "go to"
    rel = []
    for x in valid:
        vx = verb_of(x)
        if x == a or vx not in RESTORE_VERBS:
            continue
        if vx in ("connect", "disconnect") and not elec:
            continue
        if _mentions(x, names):
            rel.append(x)
        elif onames and vx in ("go to", "open", "close") and _mentions(x, onames):
            rel.append(x)

    def pri(x):
        if inv and x.startswith(inv) and _mentions(x, names):
            return 0
        if onames and x.startswith("move") and _mentions(x, names) and any(x.endswith(" to " + o) for o in onames):
            return 0
        if agent_moved and onames and x.startswith("go to") and _mentions(x, onames):
            return 0
        if onames and x.startswith(("open", "close")) and _mentions(x, onames):
            return 1
        return 2
    rel.sort(key=lambda x: (pri(x), len(x), x))
    return rel[:TOP_K]

def label_item(E, prefix, a, horizons=(1, 2, 3)):
    Hmax = max(horizons)
    # null trajectory: s then wait 1..Hmax+1 ticks (one replay)
    E.replay(prefix)
    c0 = E.state()
    # the location line is "This room is called the X." indoors but
    # "This outside location is called the outside." outdoors: accept any "This ... is called the X."
    room = re.search(r"This [a-z ]+? is called the ([^.]+)\.", E.env.look())
    if not room:
        raise RuntimeError("cannot read the current location; no valid 1-tick no-op")
    noop = f"go to {room.group(1)}"
    m0 = E.env.get_num_moves()
    null = {0: c0}
    for t in range(1, 2 * (Hmax + 1) + 1):        # nulls for every tick count we may meet
        before = E.env.get_num_moves()
        E.env.step(noop)
        if E.env.get_num_moves() - before != 1:  # guard: the baseline must advance exactly 1 tick
            raise RuntimeError(f"no-op '{noop}' advanced {E.env.get_num_moves() - before} ticks, expected 1")
        null[E.env.get_num_moves() - m0] = E.state()
    if c0 != null[0] or null.get(1) is None:
        raise RuntimeError("null trajectory incomplete")
    # after the action
    E.replay(prefix)
    obs, _, _, _ = E.env.step(a)
    c1 = E.state()
    # diff against the 1-tick null, not s itself, so ambient change (melting, respawning
    # water) is not mistaken for the object the action moved
    moved, origin = moved_info(null[1][0] if 1 in null else c0[0], c1[0])
    mrefs, orefs = referents(E, moved), referents(E, origin)
    best = {"exact": None, "soft": None}

    def ticks():
        return E.env.get_num_moves() - m0

    def vs_null(c, t):
        return compare(c, null[t]) if t in null else "none"
    m = vs_null(c1, ticks())
    if m != "none":
        best[m] = 0
    nodes = 0
    frontier = [[]]
    valid_cache = {(): E.valid()}
    for depth in range(1, Hmax + 1):
        if best["exact"] is not None:
            break
        nxt = []
        for seq in frontier:
            if nodes >= MAX_NODES:
                break
            if tuple(seq) not in valid_cache:
                E.replay(prefix + [a] + seq)
                valid_cache[tuple(seq)] = E.valid()
            for r in candidates(valid_cache[tuple(seq)], a, origin, moved, mrefs, orefs):
                if nodes >= MAX_NODES:
                    break
                E.replay(prefix + [a] + seq + [r])
                nodes += 1
                m = vs_null(E.state(), ticks())
                if m == "exact" and best["exact"] is None:
                    best["exact"] = depth
                if m == "soft" and best["soft"] is None:
                    best["soft"] = depth
                if best["exact"] is not None:
                    break
                nxt.append(seq + [r])
            if best["exact"] is not None:
                break
        frontier = nxt
    labels = {}
    for H in horizons:
        if best["exact"] is not None and best["exact"] <= H:
            labels[str(H)] = "reversible"
        elif best["soft"] is not None and best["soft"] <= H:
            labels[str(H)] = "partial"
        else:
            labels[str(H)] = "irreversible"
    return labels, obs, nodes, best, nodes >= MAX_NODES, {"moved": moved, "origin": origin, "noop": noop}


def state_text(E, taskdesc):
    look = E.env.look()
    inv = E.env.inventory()
    return f"[Task]\n{taskdesc}\n[Observation]\n{look}\n[Inventory]\n{inv}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scienceworld.jsonl")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--states-per-task", type=int, default=16)
    ap.add_argument("--actions-per-state", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out.replace(".jsonl", f".w{a.worker}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    E = Env()
    tasks = [t for i, t in enumerate(TASKS) if i % a.n_workers == a.worker]
    with open(out, "a") as f:
        for task in tasks:
            rng = random.Random(zlib.crc32(f"{task}|{a.seed}".encode()))
            E.load(task, 0)
            nvar = len(E.env.get_variations_train())
            for si in range(a.states_per_task):
                var = E.env.get_variations_train()[rng.randrange(nvar)]
                gold = E.load(task, var)
                k = rng.randrange(0, max(1, min(MAX_PREFIX, len(gold) - 1)))
                prefix = gold[:k]
                E.replay(prefix)
                taskdesc = E.env.get_task_description()
                st_text = state_text(E, taskdesc)
                valid = [x for x in E.valid() if verb_of(x) in ACTION_VERBS]
                acts = []
                if k < len(gold) and verb_of(gold[k]) in ACTION_VERBS:
                    acts.append(gold[k])
                # verb-stratified sampling: pick a verb uniformly, then an action with it,
                # so rare state-changing verbs (mix, pour, eat, activate) are not swamped
                # by the thousands of connect/move combinations
                by_verb = {}
                for x in valid:
                    if x not in acts:
                        by_verb.setdefault(verb_of(x), []).append(x)
                verbs = sorted(v for v in by_verb if v != verb_of(acts[0]) ) if acts else sorted(by_verb)
                rng.shuffle(verbs)
                for v in verbs:
                    if len(acts) >= a.actions_per_state:
                        break
                    acts.append(rng.choice(sorted(by_verb[v])))
                for act in acts:
                    iid = f"sw-{task}-v{var}-k{k}-{zlib.crc32(act.encode()):08x}"
                    if iid in done:
                        continue
                    t0 = time.time()
                    try:
                        labels, obs, nodes, best, capped, info = label_item(E, prefix, act)
                    except Exception as e:      # never drop silently: log the failure
                        print("FAIL", iid, repr(e))
                        continue
                    rec = {"id": iid, "environment": "scienceworld", "scenario_id": f"sw-{task}",
                           "pair_id": None, "tier": "single", "recipe": f"sw:{verb_of(act)}",
                           "domain": "physical", "phrasing_family": None,
                           "task_text": taskdesc, "state_text": st_text,
                           "action_text": json.dumps({"action": act}),
                           "label": labels["3"], "horizon": 3, "labels_by_horizon": labels,
                           "search_nodes": nodes, "search_capped": capped,
                           "restore_depth": best, "observation_after": obs[:300], "search_info": info,
                           "variation": var, "prefix_len": k, "prefix": prefix,
                           "source": "ScienceWorld 1.2.3 (Apache-2.0) + irrev.scienceworld_label"}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{iid} {labels} nodes={nodes} {time.time()-t0:.1f}s")
    os._exit(0)


if __name__ == "__main__":
    main()
