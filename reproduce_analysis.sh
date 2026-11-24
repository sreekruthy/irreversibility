#!/usr/bin/env bash
# Rebuild every table and figure in the paper from the saved model outputs in results/ (CPU only, ~2 min).
# GPU steps that produce results/ are listed in README.md ("Reproducing the GPU runs").
set -euo pipefail
cd "$(dirname "$0")"
python -m irrev.lexical_check data/sandbox.jsonl                       # Data section: bag-of-words baselines per tier
for T in qwen7b llama8b; do
  python -m irrev.analyze_behaviour results/$T                          # RQ2, RQ3, blindness gap (Tables 2-3)
  python -m irrev.analyze_probes --tag $T                               # RQ1, ablations, probe-vs-stated kappa
  python -m irrev.analyze_steering --steer results/$T/steer --neutral results/$T/agent_neutral.jsonl \
         --out results/$T/analysis_steering.json                        # RQ4 (trend test, controls)
  [ -f results/$T/probes/sw_preds.jsonl ] && python -m irrev.analyze_extra --tag $T   # exploratory controls
done
python irrev/plot_probes.py                                             # Figure: probes per layer
python irrev/plot_steering.py                                           # Figure: steering
python tools/annotation_agreement.py data/annotations/annotation_sheet_annotator_A.xlsx \
       data/annotations/annotation_sheet_annotator_B.xlsx                # Cohen's kappa (data card, appendix)
python -m irrev.supplement                                             # exploratory K, power, secondary RQ3 measures, LaTeX tables
(cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null && echo "paper/main.pdf rebuilt")
