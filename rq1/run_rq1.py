"""RQ1 experiment loop (spec §5), paired design. Run from the repo root:

    python -m rq1.run_rq1                       # uses rq1/config.yaml
    python -m rq1.run_rq1 --config X --out Y    # e.g. a mock smoke test

Trials are generated ONCE (order, untrusted text, channel, delivery style, seeds) and
every model x prompt condition sees exactly the same trials, so prompt conditions can
be compared with paired McNemar tests. Completions are cached by (model, prompt)
in results_rq1/cache.jsonl, so an interrupted run resumes without new API calls.

Outputs: <out>/raw_rq1.csv (no payload text; releasable) and <out>/log.jsonl,
<out>/cache.jsonl (contain payloads: git-ignored, never release).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))

from mfgsec.plans import make_order, correct_plan               # noqa: E402
from mfgsec.experiment import CONFIGS, _run_config, harm_success  # noqa: E402
from mfgsec.sim import execute                                   # noqa: E402
from rq1.prompt import build_prompt, UNTRUSTED_SLOTS             # noqa: E402
from rq1.parse import parse_plan                                 # noqa: E402
from rq1.models import build_client                              # noqa: E402
from rq1.verifier import build_judge_prompt, parse_decision      # noqa: E402

try:
    from rq1.inject import ATTACK_CLASSES, STYLES, inject, followed, benign_text  # private
except ImportError:
    sys.exit("rq1/inject.py missing: copy inject_template.py to inject.py and implement it.")

SCORED = ("D0_none", "D3_safety_auth", "D4b_master_data", "FULL_D3+D4bc")


def make_trials(n: int, seed: int) -> list[dict]:
    """One trial set shared by all models and conditions. Channel and delivery style
    are balanced by cycling, not crossed, to keep the call budget linear in n."""
    rng = np.random.default_rng(seed)
    trials, idx = [], 0
    for cls in ATTACK_CLASSES + ["benign"]:
        for i in range(n):
            idx += 1
            order = make_order(rng, 900000 + idx, "nominal")
            channel = UNTRUSTED_SLOTS[i % len(UNTRUSTED_SLOTS)]
            style = STYLES[(i // len(UNTRUSTED_SLOTS)) % len(STYLES)]
            untrusted = (benign_text(order, rng) if cls == "benign"
                         else inject(order, cls, channel, rng, style))
            trials.append({"trial": idx, "attack_class": cls, "order": order,
                           "channel": channel if cls != "benign" else "",
                           "style": style if cls != "benign" else "",
                           "untrusted": untrusted,
                           "gen_seed": int(rng.integers(0, 2**31 - 1)),
                           "exec_seed": int(rng.integers(0, 2**31 - 1))})
    return trials


class Cache:
    def __init__(self, path: Path):
        self.path, self.d = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf8").splitlines():
                r = json.loads(line)
                self.d[r["key"]] = r
        self.f = open(path, "a", encoding="utf8")

    @staticmethod
    def key(mcfg, system, user, seed):
        # Every setting that changes the completion is part of the key.
        opts = [mcfg["model"], mcfg.get("temperature", 0.0), mcfg.get("no_think", False),
                mcfg.get("structured_output")]
        return hashlib.sha256(json.dumps([opts, system, user, seed]).encode()).hexdigest()

    def get_or_call(self, client, mcfg, system, user, seed):
        k = self.key(mcfg, system, user, seed)
        if k in self.d:
            return self.d[k]["text"], self.d[k]["latency_s"], True
        t0 = time.perf_counter()
        text = client.complete(system, user, seed=seed)
        lat = time.perf_counter() - t0
        rec = {"key": k, "model": mcfg["model"], "text": text, "latency_s": lat}
        self.d[k] = rec
        self.f.write(json.dumps(rec) + "\n"); self.f.flush()
        return text, lat, False


def score_plan(plan, order, correct, exec_seed, base):
    """Run the plan through the simulator under each scored defence (as in the
    worst-case study) and return detection and harm per defence."""
    out = {}
    for dname in SCORED:
        r = _run_config(CONFIGS[dname], plan, order, exec_seed, correct)
        out[f"{dname}_detected"] = bool(r["blocked_pre"] or r["alarm_on_original_plan"])
        out[f"{dname}_harm"] = harm_success(r, base)
    return out


def _ollama_digests(cfg) -> dict:
    """Pin local models: Ollama tags are mutable, digests are not (spec §6)."""
    import urllib.request
    out = {}
    for m in cfg["models"] + ([cfg["verifier"]] if cfg.get("verifier") else []):
        if m.get("kind") != "openai_compatible" or "11434" not in m.get("base_url", ""):
            continue
        try:
            base = m["base_url"].rsplit("/v1", 1)[0]
            with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as r:
                tags = {t["name"]: t["digest"] for t in json.loads(r.read())["models"]}
            out[m["model"]] = tags.get(m["model"])
        except Exception as e:      # recorded, not fatal
            out[m["model"]] = f"unavailable: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "rq1" / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results_rq1"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Cache(out / "cache.jsonl")
    log = open(out / "log.jsonl", "a", encoding="utf8")
    trials = make_trials(cfg["n_per_cell"], cfg.get("seed", 424242))
    vcfg = cfg.get("verifier")
    verifier = build_client(vcfg) if vcfg else None
    rows, n_new = [], 0
    meta = {"config": cfg, "ollama_digests": _ollama_digests(cfg)}
    (out / "run_meta_rq1.json").write_text(json.dumps(meta, indent=2, default=str))
    total = len(cfg["models"]) * len(cfg["conditions"]) * len(trials)
    for mcfg in cfg["models"]:
        client = build_client(mcfg)
        for cond in cfg["conditions"]:
            for t in trials:
                order, cls = t["order"], t["attack_class"]
                correct = correct_plan(order)
                system, user = build_prompt(order, t["untrusted"], cond)
                text, lat, hit = cache.get_or_call(client, mcfg, system, user, t["gen_seed"])
                n_new += not hit
                plan, err = parse_plan(text)
                row = {"model": mcfg["name"], "model_id": mcfg["model"],
                       "temperature": mcfg.get("temperature", 0.0), "condition": cond,
                       "attack_class": cls, "channel": t["channel"], "style": t["style"],
                       "trial": t["trial"], "product": order.product, "latency_s": lat,
                       "parse_error": err is not None,
                       "injected_followed": bool(plan is not None and cls != "benign"
                                                 and followed(plan, correct, cls)),
                       "plan_equals_correct": bool(plan is not None
                                                   and plan["steps"] == correct["steps"]
                                                   and plan["machine"] == correct["machine"])}
                if plan is not None:
                    # Identity fields are bound to the real order, not taken from the planner.
                    plan["order_id"], plan["product"] = order.order_id, order.product
                    base = execute(correct, order, np.random.default_rng(t["exec_seed"]),
                                   spc=False, fallback_plan=correct)
                    row.update(score_plan(plan, order, correct, t["exec_seed"], base))
                    if verifier is not None and cond in cfg.get("verifier_conditions", cfg["conditions"]):
                        js, ju = build_judge_prompt(order, plan)
                        jt, jl, jhit = cache.get_or_call(verifier, vcfg, js, ju, t["gen_seed"])
                        n_new += not jhit
                        acc, ok = parse_decision(jt)
                        row["D2_detected"], row["D2_parse_ok"], row["D2_latency_s"] = (not acc), ok, jl
                        # D2 block -> operator review, then the correct plan runs.
                        row["D2_harm"] = row["D0_none_harm"] and acc
                rows.append(row)
                log.write(json.dumps({**row, "system": system, "user": user,
                                      "completion": text}) + "\n")
                if len(rows) % 50 == 0:
                    print(f"{len(rows)}/{total} rows, {n_new} new model calls", flush=True)
    pd.DataFrame(rows).to_csv(out / "raw_rq1.csv", index=False)
    print(f"wrote {len(rows)} rows to {out} ({n_new} new model calls)")


if __name__ == "__main__":
    main()
