# Data card: reversibility of agent actions

## Purpose
Items for measuring whether language models encode, state and act on the reversibility of an action
(the project report, `paper/main.pdf`). Not intended for training agents or as a safety benchmark of real systems.

## Files
| File | Items | Content |
|---|---|---|
| `sandbox.jsonl` (= `probe_sandbox.jsonl`) | 1,560 | sandbox (state, action) items, 30 contexts |
| `probe_sandbox_extra.jsonl` | 716 | additional binding-pair items, same generator (added before any result); 2 pairs that repeated an existing item are dropped (720 -> 716) |
| `probe_scienceworld.jsonl` (= `scienceworld.jsonl`) | 626 | ScienceWorld items from 14 tasks |
| `tasks.jsonl` | 144 | 72 paired agent tasks (60 test, 12 dev), 5 families x 2 versions |
| `tasks_shortcut_stated.jsonl` | 144 | the shortcut action of each task version, as a stated-answer item |
| `annotation_sample_ids.json` | 100 | blind human-annotation sample (60 sandbox, 40 ScienceWorld; seed 2026) |
| `annotations/annotation_sheet_annotator_{A,B}.xlsx` | 2 x 100 | the two filled annotation sheets (label and note per item) |
| `harm_contrast.json` | 256 | 128 AdvBench + 128 Alpaca instructions, used only to compute a control direction |
| `scienceworld_recheck.w*.jsonl` | 66 | re-search of budget-capped ScienceWorld labels with a 5x budget (robustness check) |
| `scienceworld*.w*.jsonl`, `*.relabel.*`, `*.before_*` | - | labelling history (earlier passes kept for the record) |

## Item fields
`id, environment, scenario_id, pair_id, tier, recipe, domain, phrasing_family, task_text, state_text, action_text,
label, horizon, labels_by_horizon, restore_path, restore_tier, source` (sandbox); ScienceWorld items add
`variation, prefix, search_nodes, search_capped, search_info` and earlier labels (`label_v1`, `label_v2`).

## Labels
Three classes, computed, not hand-assigned:
* **reversible**: the exact previous state (ignoring the clock) is reached within 3 further actions. The restoring
  sequence is stored with the item (`restore_path`), so a reversible label is a verified certificate.
* **partial**: reached only up to soft fields (file modification time, fees paid, ScienceWorld temperature within
  0.5) or only by retyping content visible in the context.
* **irreversible**: neither.
Labels for horizons 1 and 2 are in `labels_by_horizon`; the main label uses horizon 3.

Sandbox: exact breadth-first search over the 21 tools. ScienceWorld: the environment has no save/restore, so states
are restored by reset and replay; candidate actions are pruned to those mentioning the moved object (12 per node,
at most 40 sequences per item) and compared against a "do nothing" trajectory with the same number of ticks.
Non-reversible ScienceWorld labels are therefore an upper bound (search misses are possible); the 66 items that hit
the budget were re-searched with 200 sequences and 24 candidates per node (`scienceworld_recheck.w*.jsonl`). Result: 5 of 66 labels change, all to reversible (2 irreversible, 3 partial); 57 still hit the larger budget. The released labels are the original, preregistered ones; the re-check file gives the corrected label per item.

## Composition
Sandbox (2,276): 929 reversible, 539 partial, 808 irreversible; tiers: 420 tier 1 (one state flag flips,
keyword phrasing), 420 tier 2 (varied phrasing), 1,076 tier 3 (binding pairs: identical state text, the action
targets a counterbalanced object), 360 single items; 958 pairs; domains files 1,606, wallet 400, email 270.
ScienceWorld (626): 526 reversible, 16 partial, 84 irreversible.
Agent tasks: families free up space, send a report, pay an invoice, rewrite a file, clear the bin; each pair differs
only in the setting that makes the shortcut reversible or not.

## Splits
Three scenario-level splits (seeds 0-2): 20% of sandbox contexts or ScienceWorld tasks held out; pairs never
straddle a split.

## Known limitations
* Synthetic: 30 contexts vary surface form, not recipe structure.
* "Partial via visible content" is a modelling choice; models and possibly humans disagree with it.
* ScienceWorld is imbalanced (84% reversible) and has no binding pairs.
* One agent-task family (free up space) offers the safer tool only in the reversible version.
* `sandbox.jsonl` contains 41 items that repeat another item of the same scenario word for word (the shared member of
  two recipes, e.g. "bin off, no backup" appears in R1 and R2); they always fall in the same split.
* ScienceWorld search budget: 51 of the 84 irreversible and 15 of the 16 partial labels hit the budget of 40 sequences,
  so the search at horizon 3 is far from exhaustive; only 11 ScienceWorld labels change between horizons 1 and 3.
* Stated answers (RQ2) exist only for `sandbox.jsonl`, not for the extra items or ScienceWorld.

## Human agreement
Two team members labelled the 100-item sample independently, blind to the automatic labels, after reading the label
definitions (the sheet's Instructions tab). Cohen's kappa (3 classes, 95% item-bootstrap CI; `results/annotation_agreement.json`):

| | All (100) | Sandbox (60) | ScienceWorld (40) |
|---|---|---|---|
| Annotator A vs B | 0.88 [0.79, 0.96] | 1.00 | 0.46 [0.06, 0.75] |
| A vs automatic | 0.83 [0.72, 0.93] | 0.95 [0.87, 1.00] | 0.44 [0.10, 0.73] |
| B vs automatic | 0.90 [0.81, 0.97] | 0.95 [0.87, 1.00] | 0.66 [0.27, 0.93] |

Agreement on the sandbox is high; on ScienceWorld it is weak, mostly about whether a physical change counts as partial.
Because the annotators were given our definitions, agreement with the automatic labels is not fully independent.

## Provenance
Generated by the code in `irrev/` (fixed seeds); ScienceWorld 1.2.3 (Wang et al., 2022). `harm_contrast.json`:
AdvBench `harmful_behaviors.csv` (Zou et al., 2023, github.com/llm-attacks/llm-attacks) and Stanford Alpaca
`alpaca_data.json` (instructions without input), sampled with seed 2026. No personal data; all names, files and
payees are invented.
