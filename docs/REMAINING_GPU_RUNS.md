# Remaining runs (optional, closes two gaps listed in the report)

Same Kaggle setup as before (2x T4, the repository and `HF_TOKEN` in the notebook). Run once per model
(Qwen first, then Llama with `M=meta-llama/Llama-3.1-8B-Instruct; R=results/llama8b`).

## 1. RQ2 stated answers on ScienceWorld and the extra binding items (GPU, ~1-1.5 h for both models)

```bash
M=Qwen/Qwen2.5-7B-Instruct; R=results/qwen7b
python -m irrev.stated --model $M --data data/probe_scienceworld.jsonl --out $R/stated_sw.jsonl            # Llama: --max-new-tokens 256
python -m irrev.stated --model $M --data data/probe_sandbox_extra.jsonl --out $R/stated_extra.jsonl        # Llama: --max-new-tokens 256
```
Download `stated_sw.jsonl`, `stated_extra.jsonl` and their `.run.jsonl` / `.status.json` files into `results/<model>/`.
Each run takes about 20-35 min per model (626 and 716 items; the 1,560 main items took 30-45 min).

## 2. Horizon ablation on all three splits (CPU on Kaggle, ~1-2 h per model; needs the hidden states)

The hidden states must exist in `$R/hidden_*` (regenerate with `extract.py`, 20-30 min per model on GPU, if the
Kaggle output was deleted). Copy the existing `results/<model>/probes/partial/` into place first so that every
finished sweep is reused and only the two new horizon sweeps are fitted:

```bash
python -m irrev.probes --out $R/probes --horizon-all-seeds \
   --set sandbox:$R/hidden_sandbox:data/probe_sandbox.jsonl \
   --set sandbox:$R/hidden_extra:data/probe_sandbox_extra.jsonl \
   --set scienceworld:$R/hidden_sw:data/probe_scienceworld.jsonl
```
Download `results/<model>/probes/` (probes.json and the new `partial/*_3seeds.json`), then run
`./reproduce_analysis.sh` locally. The ablation table picks up the three-split values automatically;
the "(1 split)" footnote, Table 8 row and Limitations sentence about it then need to be removed.

## Total
About 3-4 hours of Kaggle time for both models, plus setup and downloads; it can run within one Kaggle session per model.
