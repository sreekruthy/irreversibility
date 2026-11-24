# Knowing It Cannot Be Undone? Encoded, Stated and Enacted Reversibility in Open-Weight LLM Agents

Course project (Advanced Deep Neural Networks, Mahindra University, Option 1).
Team: Shri Honeytha, Devi Sandeepya, Sreekruthy, Lalitha Manaswini (contributions: [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md)).
Report: [`paper/main.pdf`](paper/main.pdf). Preregistered analysis plan: [`PREREGISTRATION.md`](PREREGISTRATION.md)
(committed 4 Nov 2026 before any model output existed; deviations are listed in the report, Section "Deviations").

We measure, on the same items, whether Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct
**encode** whether an action can be undone (linear probes on hidden states, RQ1), **state** it when asked (RQ2),
**act** on it as agents (RQ3), and whether steering along the probe direction changes behaviour (RQ4).

## Main results

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Probe, sandbox, binary balanced accuracy at preregistered layer L* | 0.964 (L*=20) | 0.975 (L*=15) |
| Probe, binding pairs (identical state text; bag-of-words = 0.50) | 0.986 | 1.000 |
| Stated answer, 3-class accuracy (same 312 test items) | 0.45 | 0.29 |
| Agent takes the irreversible shortcut (irreversible versions, 60 pairs) | 1.00 | 0.95 |
| Caution, irreversible / reversible version | 0.10 / 0.17 | 0.02 / 0.05 |
| Steering trend (strict caution vs strength), slope per unit | -0.62, p < 1e-9 | -1.04, p < 1e-7 |
| Irreversible action under steering (any strength) | 0.95-1.00 | 0.80-0.95 |

Human check of the labels (100 items, two annotators): Cohen's kappa 0.88 between annotators (sandbox 1.00, ScienceWorld 0.46),
0.83 / 0.90 with the automatic labels (ScienceWorld 0.44 / 0.66); see `results/annotation_agreement.json`.

Limits stated in the report: probes do not transfer between the sandbox and ScienceWorld, and on ScienceWorld
they are no better than bag-of-words on the action; steering is not direction-specific for Qwen
(one random direction has a similar effect) and never changes the irreversible decision.

## Run order (whole project)

Each stage needs the outputs of the stages before it. The commands are in the sections further down.

| # | Stage | Where | Code | Output |
|---|---|---|---|---|
| 0 | Install, run the tests | CPU | `requirements.txt`, `tests/` | — |
| 1 | Build the sandbox items and binding pairs | CPU | `generate_sandbox.py`, `generate_binding_extra.py` (uses `sandbox.py`, `labeller.py`, `contexts.py`) | `data/sandbox.jsonl`, `data/probe_sandbox*.jsonl` |
| 2 | Build the ScienceWorld items and check their labels | CPU, hours | `scienceworld_label.py`, then `scienceworld_relabel.py`, then `scienceworld_recheck.py` | `data/scienceworld.jsonl`, `data/scienceworld_recheck.w*.jsonl` |
| 3 | Build the agent tasks | CPU | `tasks.py` | `data/tasks.jsonl`, `data/tasks_shortcut_stated.jsonl` |
| 4 | Lexical check of the dataset | CPU | `lexical_check.py` | stdout |
| 5 | Hidden states, all layers (for RQ1) | GPU | `extract.py` | `results/<model>/hidden_*` |
| 6 | RQ2 stated answers | GPU | `stated.py` | `results/<model>/stated_*.jsonl` |
| 7 | RQ3 agent runs, neutral and careful prompt | GPU | `agent.py` | `results/<model>/agent_*.jsonl` |
| 8 | RQ1 probes, baselines, ablations, steering directions | CPU | `probes.py` | `results/<model>/probes/` |
| 9 | RQ4 steering: calibrate on dev tasks, then 12 test conditions | GPU | `steer_run.py` (uses `agent.py`) | `results/<model>/steer/` |
| 10 | Exploratory controls: harm direction, held-out tool domain, ScienceWorld predictions | GPU | `extra_analyses.py` | `results/<model>/extra_controls.json`, `steer/test_harm_*` |
| 11 | Save the rendered prompts | CPU | `dump_prompts.py` | `results/<model>/prompts.jsonl.gz` |
| 12 | Statistics, figures, paper | CPU, ~2 min | `reproduce_analysis.sh` (`analyze_*.py`, `plot_*.py`) | `results/<model>/analysis_*.json`, `paper/fig_*`, `paper/main.pdf` |
| 13 | Human annotation agreement (κ) | CPU | `tools/make_annotation_sheet.py` (blank sheets), then `tools/annotation_agreement.py` on the filled sheets | `tools/annotation_sheets/`, `data/annotations/`, `results/annotation_agreement.json` |
| 14 | Supplementary analyses and LaTeX tables | CPU | `supplement.py` (inside `reproduce_analysis.sh`) | `results/supplement.json`, `paper/tables/*.tex` |
| 15 | Demo for the presentation (replay of saved episodes) | CPU | `demo_replay.py` | stdout |

Run stages 5–11 once per model: Qwen first, then Llama. `results/` already contains every output except the
hidden states, so stage 12 runs straight from a fresh clone. `kaggle_runner.ipynb` is the first Kaggle notebook, kept
for the record; the commands below are the current ones. `logs/` holds the ScienceWorld labelling logs.

## Repository layout

```
irrev/            all code (package)
  sandbox.py        deterministic tool sandbox (files, outbox, wallet; 21 tools)
  labeller.py       ground truth by breadth-first search over restoring action sequences (horizons 1-3)
  generate_sandbox.py, generate_binding_extra.py, contexts.py   sandbox probe items (tiers 1-3, binding pairs)
  scienceworld_label.py, scienceworld_relabel.py, scienceworld_recheck.py   ScienceWorld items and label checks
  tasks.py          72 paired agent tasks (60 test, 12 dev) + their shortcut items for RQ2
  common.py         prompts, model loading (fp16 on T4)
  extract.py        hidden states at every layer (last and mean action token)         [GPU]
  stated.py         stated answers (option probabilities + free text)                 [GPU]
  agent.py          agent loop, caution logging, optional steering hook               [GPU]
  steer_run.py      RQ4: calibration on dev tasks, then 12 test conditions            [GPU]
  extra_analyses.py exploratory controls: harm direction, held-out tool domain         [GPU]
  probes.py         RQ1: probes per layer with all baselines, controls and ablations   [CPU]
  analyze_behaviour.py, analyze_probes.py, analyze_steering.py   statistics           [CPU]
  plot_probes.py, plot_steering.py   figures                                           [CPU]
  runguard.py       resume-or-restart for every long run
  supplement.py     exploratory K, power analysis, secondary RQ3 measures, LaTeX tables   [CPU]
  demo_replay.py    presentation demo: replays one task pair from the saved logs          [CPU]
data/             all datasets (see data/DATA_CARD.md); data/annotations/ holds the two filled annotation sheets
results/<model>/  every model output: stated answers, agent traces, probe results, steering runs,
                  rendered prompts (prompts.jsonl.gz), analysis JSONs. Hidden states are not included (~1.5 GB).
paper/            LaTeX source, figures, bibliography
tools/            blank annotation sheets (tools/annotation_sheets/), agreement script, fake model for tests
tests/            42 tests (labeller, agent logic, architectures, resume behaviour)
```

## Setup

Python 3.13. GPU runs and probe fitting used Kaggle (2x Tesla T4, fp16; torch 2.11.0+cu130, transformers 5.22.1,
scikit-learn 1.6.1); the full package list is `results/kaggle_pip_freeze.txt`. Offline analyses: `requirements.txt`.

```bash
pip install -r requirements.txt
python -m pytest -q tests && bash tests/test_resume.sh    # uses a tiny random model from tools/make_fake_model.py
```
Llama-3.1-8B-Instruct is gated: accept its licence on Hugging Face and set `HF_TOKEN`.
ScienceWorld needs Java (unset `JAVA_TOOL_OPTIONS` if it is set).

## Reproducing the tables and figures (CPU, ~2 minutes)

```bash
./reproduce_analysis.sh
```
This rebuilds every number, table and figure in the paper from `results/` and recompiles `paper/main.pdf`.

| Paper element | Command (inside reproduce_analysis.sh) | Output |
|---|---|---|
| Data: bag-of-words per tier | `python -m irrev.lexical_check data/sandbox.jsonl` | stdout |
| RQ2, RQ3, blindness gap | `python -m irrev.analyze_behaviour results/<model>` | `results/<model>/analysis_behaviour.json` |
| RQ1, ablations, probe-vs-stated kappa | `python -m irrev.analyze_probes --tag <model>` | `results/<model>/analysis_probes.json` |
| RQ4 | `python -m irrev.analyze_steering --steer results/<model>/steer --neutral results/<model>/agent_neutral.jsonl --out ...` | `results/<model>/analysis_steering.json` |
| Figures | `python irrev/plot_probes.py`, `python irrev/plot_steering.py` | `paper/fig_*.pdf` |
| Annotation agreement (appendix table) | `python tools/annotation_agreement.py data/annotations/annotation_sheet_annotator_A.xlsx data/annotations/annotation_sheet_annotator_B.xlsx` | `results/annotation_agreement.json` |
| Tables 4-8, exploratory K, power | `python -m irrev.supplement` | `results/supplement.json`, `paper/tables/*.tex` |

## Reproducing the data (CPU)

```bash
python -m irrev.generate_sandbox --out data/sandbox.jsonl --seed 0
python -m irrev.generate_binding_extra --out data/probe_sandbox_extra.jsonl   # drops 2 duplicate pairs: 716 items
python -m irrev.tasks --out data/tasks.jsonl --stated-out data/tasks_shortcut_stated.jsonl
python -m irrev.scienceworld_label --out data/scienceworld.jsonl --worker 0 --n-workers 2   # and worker 1; hours
cp data/sandbox.jsonl data/probe_sandbox.jsonl; cp data/scienceworld.jsonl data/probe_scienceworld.jsonl   # both copies are in the repo
```
The ScienceWorld labels were produced by the labeller and then re-searched in audit passes after fixes to the
candidate generator (`scienceworld_relabel.py`, logs in `logs/`); a fresh run of the current labeller reproduces the
procedure, not necessarily every label (the search is budget-capped).
```bash
```
Seeds are fixed (zlib.crc32 of names, not Python's `hash`), so the files are reproduced exactly.

## Reproducing the GPU runs

Each command resumes where it stopped if interrupted (`runguard.py`). Shown for Qwen; for Llama use
`meta-llama/Llama-3.1-8B-Instruct` and `results/llama8b`.

```bash
M=Qwen/Qwen2.5-7B-Instruct; R=results/qwen7b
for s in "probe_sandbox hidden_sandbox" "probe_sandbox_extra hidden_extra" "probe_scienceworld hidden_sw"; do set -- $s
  python -m irrev.extract --model $M --data data/$1.jsonl --out $R/$2; done                          # hidden states
python -m irrev.stated --model $M --data data/sandbox.jsonl --out $R/stated_sandbox.jsonl             # RQ2 (Llama: --max-new-tokens 256)
python -m irrev.stated --model $M --data data/tasks_shortcut_stated.jsonl --out $R/stated_tasks.jsonl
python -m irrev.agent --model $M --tasks data/tasks.jsonl --split test --out $R/agent_neutral.jsonl    # RQ3
python -m irrev.agent --model $M --tasks data/tasks.jsonl --split test --out $R/agent_careful.jsonl --careful
python -m irrev.probes --out $R/probes --set sandbox:$R/hidden_sandbox:data/probe_sandbox.jsonl \
   --set sandbox:$R/hidden_extra:data/probe_sandbox_extra.jsonl --set scienceworld:$R/hidden_sw:data/probe_scienceworld.jsonl   # RQ1 (CPU)
python -m irrev.steer_run --model $M --probes $R/probes --out $R/steer                                # RQ4
python -m irrev.extra_analyses --model $M --tag qwen7b && python -m irrev.extra_analyses --model $M --tag qwen7b --steer-harm
python -m irrev.dump_prompts --model $M --out $R/prompts.jsonl.gz --data data/probe_sandbox.jsonl \
   data/probe_sandbox_extra.jsonl data/probe_scienceworld.jsonl --tasks data/tasks.jsonl data/tasks_shortcut_stated.jsonl
```
Approximate Kaggle time per model: hidden states 20-30 min, stated answers 30-45 min, agent runs 10-30 min,
probes 1.5-3.5 h (CPU), steering 2-3.5 h.

## Notes

* Probes are fitted in the span of the standardised training vectors (exact for L2 logistic regression;
  see `RowSpace` in `irrev/probes.py`); binary probes use the Newton-Cholesky solver.
* The simulated user's reply and the RQ2 question wording were fixed during the dev pilot, before any test result.
* AI assistance: see the disclosure in the report (Ethics Statement).

## Demo (presentation)

Replays one task pair from the saved agent logs, re-executing every tool call in the sandbox (no GPU):
```bash
python -m irrev.demo_replay --list                                   # all pairs with their outcome
python -m irrev.demo_replay --model qwen7b --pair task-game-F4_rewrite_file --pause   # Enter = next step
```
In the default pair the agent edits the file directly in both versions, although only the reversible version has a backup.

## Hidden states

The hidden states (~1.5 GB per model) are not in the repository. Regenerate them with `extract.py` (stage 5;
20-30 min per model on 2x T4). `results/<model>/probes/` holds every probe score, so no table needs them.

## Data sources

ScienceWorld (Wang et al., 2022) is used through its Python package. The harmful/harmless
instructions in `data/harm_contrast.json` are a fixed sample from AdvBench (Zou et al., 2023) and Stanford Alpaca,
used only to compute an activation direction.
