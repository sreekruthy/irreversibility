#!/usr/bin/env bash
# End-to-end resume/restart tests with the fake model (CPU). Run from the repo root.
set -u
M=${FAKE_MODEL:-/home/claude/fake_model}
T=$(mktemp -d); OUT=$T/st.jsonl
run() { python3 -m irrev.stated --model $M --data data/sandbox.jsonl --out $OUT --limit 32 --batch 4 --max-new-tokens 6 "$@" 2>&1 | grep -E "runguard|DONE|FAILED|Error" ; }
state() { python3 -c "import json;print(json.load(open('$OUT.status.json'))['state'])"; }
nrows() { wc -l < $OUT | tr -d ' '; }
uniq_ids() { python3 -c "import json;r=[json.loads(l) for l in open('$OUT')];print(len(r)==len({x['id'] for x in r}))"; }

echo "== 1. kill mid-run (simulated timeout), then resume"
timeout -s KILL 12 python3 -m irrev.stated --model $M --data data/sandbox.jsonl --out $OUT --limit 32 --batch 4 --max-new-tokens 6 >/dev/null 2>&1
echo "after kill: state=$(state) rows=$(nrows)"
run
echo "after resume: state=$(state) rows=$(nrows) unique_ids=$(uniq_ids)"

echo "== 2. rerun when complete"
run

echo "== 3. truncated last line (killed during a write), then resume"
python3 - <<PY
import json; p="$OUT"; L=open(p).read().splitlines(); open(p,"w").write("\n".join(L[:20])+"\n"+L[20][:40])
s=json.load(open(p+".status.json")); s["state"]="running"; json.dump(s,open(p+".status.json","w"))
PY
run
echo "after: state=$(state) rows=$(nrows) unique_ids=$(uniq_ids)"

echo "== 4. error mid-run, then restart from scratch"
rm -rf $T/*; IRREV_TEST_FAIL_AFTER_BATCH=2 run
echo "after error: state=$(state) rows=$(nrows)"
run
echo "after restart: state=$(state) rows=$(nrows) unique_ids=$(uniq_ids) | archived: $(ls -d $T/*failed* | wc -l)"

echo "== 5. settings changed -> archive and restart"
run --no-free-text
echo "after: state=$(state) rows=$(nrows) | archived: $(ls -d $T/*failed* | wc -l)"
rm -rf $T
