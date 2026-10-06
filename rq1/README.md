# RQ1 harness

Measures how often real LLM planners follow production-semantic injections carried by
work-order free text, and evaluates D1 (prompt hardening), D4a (channel separation) and
D2 (LLM-as-verifier).

| File | Role |
|---|---|
| `prompt.py` | prompt assembly: plain / d1_hardened / d4a_channel_separation; D4a typed vocabulary |
| `inject.py` | **private, git-ignored**: payload families (C1–C6 × direct/authority), benign text, `followed()` |
| `parse.py` | completion → plan; executor schema validation (errors reported, never counted as defended) |
| `verifier.py` | D2 judge prompt and fail-closed decision parsing |
| `models.py` | OpenAI-compatible (Ollama etc.), Anthropic, and offline mock planners (tests only) |
| `run_rq1.py` | paired loop: one trial set shared by all models/conditions; completion cache; harm scored with `mfgsec.experiment.harm_success` |
| `analysis.py` | compliance (Wilson), paired McNemar plain vs D1/D4a (Holm), benign errors, real-plan detection/harm, SVG figure |

Run from the repo root:

    cp rq1/config.example.yaml rq1/config.yaml   # pin models
    python -m rq1.run_rq1                        # resumable (results_rq1/cache.jsonl)
    python -m rq1.analysis
    python tests/test_rq1_pipeline.py            # offline end-to-end test (mock planners)

Releasable: `results_rq1/raw_rq1.csv`, `table_*.csv`, `figures/`. Never release:
`inject.py`, `results_rq1/log.jsonl`, `results_rq1/cache.jsonl` (contain payloads).
