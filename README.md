# process-aware-verification-mfg

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23197334.svg)](https://doi.org/10.5281/zenodo.23197334)

Code and data for the paper **"Securing AI-Driven Task Planning for Industrial Robots:
Process-Aware Verification Against Prompt Injection in Manufacturing Cells"**
by Andrii Koliada, Volodymyr Tonkonogyi and Liubov Bovnegra
(Odesa Polytechnic National University). The paper is under review; the citation will be
added on publication.

The repository contains a discrete-event simulation of a robot-tended finish-turning cell,
generators for benign orders and ten production-semantic attack classes on LLM task plans,
the defences evaluated in the paper, and every result table and figure.

## Contents

```
sim/                    worst-case evaluation (compromised planner, no LLM in the loop) -> RQ2, RQ3
  mfgsec/model.py         products, process-quality model and simulation assumptions
  mfgsec/plans.py         benign orders, correct plans, attack-plan generators (class level)
  mfgsec/defences.py      D3 safety/authorisation verifier, D4b master-data verifier
  mfgsec/sim.py           cell simulation (SimPy) incl. D4c SPC outcome monitor
  mfgsec/experiment.py    paired evaluation of all defence configurations
  run.py                  regenerates everything in results/
rq1/                    end-to-end experiment with real LLM planners -> RQ1
  prompt.py               prompt assembly: plain, D1 hardened, D4a channel separation
  parse.py, verifier.py   plan parsing/schema check, D2 LLM-as-verifier
  run_rq1.py, analysis.py paired run and statistics
  inject_template.py      structure of the private payload module (see "Withheld material")
results/                simulation tables (CSV) and figures (SVG/PDF/PNG)
results_rq1/            end-to-end per-order results, tables and figure
tools/fig_architecture.py  cell architecture diagram -> paper/figures/
tests/                  offline test of the RQ1 pipeline with mock planners
```

## Reproducing the results

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python sim/run.py                 # rewrites results/ (fixed seeds, reproducible)
python -m rq1.analysis            # RQ1 tables and figure from results_rq1/raw_rq1.csv
python tools/fig_architecture.py  # architecture diagram
```

`python sim/run.py --n 50 --n_sens 10 --out /tmp/quick` gives a fast smoke run that does not
touch `results/`.

The process parameters in `sim/mfgsec/model.py` (`results/table_assumptions.csv`,
`results/table_products.csv`) are literature-typical simulation assumptions reviewed by the
manufacturing co-author, not measurements from a plant.

## Withheld material

For responsible disclosure, the injection payloads are not released. `rq1/inject.py`, which
builds the injected work-order text, is withheld; `rq1/inject_template.py` documents its
interface. The prompts, model completions and injected content of the end-to-end run are
withheld as well. Released are the per-order results (`results_rq1/raw_rq1.csv`), the
statistics and the plan audit (`table_rq1_audit.csv`, derived by `rq1.audit` from the
withheld log). Re-running `rq1.run_rq1` therefore requires your own
implementation of `inject.py`; the offline test is skipped without it.

The end-to-end run used locally hosted models via Ollama (Qwen3 8B and Llama 3.1 8B as
planners, Gemma 3 12B as verifier); the exact configuration and model digests are in
`results_rq1/run_meta_rq1.json`.

## License

Code: MIT (`LICENSE`). Result data and figures in `results/`, `results_rq1/` and
`paper/figures/`: CC BY 4.0 (`LICENSE-DATA`).

## Citation

Archived on Zenodo: https://doi.org/10.5281/zenodo.23197334 (all versions). See `CITATION.cff`.
Please cite the paper once published.
