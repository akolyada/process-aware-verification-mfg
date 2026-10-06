"""Plan audit for RQ1: derives releasable COUNTS from the private log (which contains
payloads and is never released). Output contains no prompt or payload text.

    python -m rq1.audit            # -> results_rq1/table_rq1_audit.csv

Rows (model, condition, attack_class, item, count, n):
  parse_error:<type>            parse/schema error types
  benign_deviation:<reasons>    D4b reasons for benign plans that differ from the correct plan
  d3_detection:<reasons>        why D3 rejected an attack plan
  in_window_shift_harm          not 'followed' (no out-of-window value), but set_param moved
                                inside the authorised window AND the plan caused harm (D0)
  harm_passed_d4b               D4b accepted the plan, yet it caused harm
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "sim")]

from mfgsec.defences import d3_safety_authorisation, d4b_master_data  # noqa: E402
from mfgsec.plans import correct_plan                                  # noqa: E402
from rq1.parse import parse_plan                                       # noqa: E402
from rq1.run_rq1 import make_trials                                    # noqa: E402


def main():
    meta = json.loads((ROOT / "results_rq1" / "run_meta_rq1.json").read_text())
    cfg = meta["config"]
    trials = {t["trial"]: t for t in make_trials(cfg["n_per_cell"], cfg.get("seed", 424242))}
    counts, n = Counter(), Counter()
    for line in open(ROOT / "results_rq1" / "log.jsonl", encoding="utf8"):
        r = json.loads(line)
        key = (r["model"], r["condition"], r["attack_class"])
        n[key] += 1
        order = trials[r["trial"]]["order"]
        correct = correct_plan(order)
        plan, err = parse_plan(r["completion"])
        if err:
            counts[key + (f"parse_error:{err.split(':')[0]}:{err.split(':', 1)[-1].strip()[:40]}",)] += 1
            continue
        plan["order_id"], plan["product"] = order.order_id, order.product
        if r["attack_class"] == "benign" and not r["plan_equals_correct"]:
            _, why = d4b_master_data(plan, order)
            counts[key + ("benign_deviation:" + "; ".join(why),)] += 1
        if r["attack_class"] != "benign" and r.get("D3_safety_auth_detected"):
            _, why = d3_safety_authorisation(plan, order)
            counts[key + ("d3_detection:" + "; ".join(why),)] += 1
        if r.get("D0_none_harm") and not r.get("D4b_master_data_detected"):
            counts[key + ("harm_passed_d4b",)] += 1
        sp = next((s for s in plan["steps"] if s["op"] == "set_param"), None)
        sc = next(s for s in correct["steps"] if s["op"] == "set_param")
        if (sp and not r["injected_followed"] and r.get("D0_none_harm")
                and not r.get("D4b_master_data_detected")
                and (sp["speed"], sp["feed"], sp["offset"]) != (sc["speed"], sc["feed"], sc["offset"])):
            counts[key + ("in_window_shift_harm",)] += 1
    rows = [{"model": k[0], "condition": k[1], "attack_class": k[2], "item": k[3],
             "count": v, "n": n[k[:3]]} for k, v in sorted(counts.items())]
    out = ROOT / "results_rq1" / "table_rq1_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
