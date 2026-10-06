"""Run the full worst-case evaluation and write results/.

    python sim/run.py                          # full run (n=300 per condition) -> results/
    python sim/run.py --n 50 --out /tmp/quick  # quick run elsewhere
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mfgsec.model import ASSUMPTIONS, PRODUCTS, SAFETY_LIMITS, rederive_feed_recipe
from mfgsec.experiment import run_attacks, run_benign
from mfgsec.plans import ATTACK_PARAMS
from mfgsec.profiles import PROFILES, apply_profile
from mfgsec import analysis as an

PROD_CLASSES = ["C1_wrong_variant", "C2_inspection_omission",
                "C3_param_outside_quality_window", "C4_operation_reorder",
                "C5_throughput_sabotage", "C6_denial_of_service_holds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_sens", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default=None, choices=sorted(PROFILES),
                    help="parameter profile (default: original model); never writes results/")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    out = args.out or str(root / ("results" if not args.profile else f"results_{args.profile}"))
    if args.profile and Path(out).resolve() == (root / "results").resolve():
        raise SystemExit("a profile run must not overwrite results/; pass another --out")
    apply_profile(args.profile)
    os.makedirs(f"{out}/figures", exist_ok=True)
    t0 = time.time()

    # ---------------- Assumptions table ---------------------------------- #
    pd.DataFrame([{"parameter": k, "value": str(v)} for k, v in ASSUMPTIONS.items()]
                 + [{"parameter": f"safety.{k}", "value": str(v)}
                    for k, v in SAFETY_LIMITS.items()]
                 ).to_csv(f"{out}/table_assumptions.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in p.__dict__.items() if k != "programs"},
                   "programs": ",".join(p.programs.values())}
                  for p in PRODUCTS.values()]).to_csv(f"{out}/table_products.csv", index=False)
    pd.DataFrame([{"parameter": k, "value": str(v)} for k, v in ATTACK_PARAMS.items()]
                 ).to_csv(f"{out}/table_attack_params.csv", index=False)

    # ---------------- Main attack experiment ----------------------------- #
    att = run_attacks(args.n)
    att.to_csv(f"{out}/raw_attacks.csv", index=False)
    det = an.rate_table(att, "detected")
    harm = an.rate_table(att, "harm_success")
    det.to_csv(f"{out}/table_detection.csv", index=False)
    harm.to_csv(f"{out}/table_harm_success.csv", index=False)
    an.harm_magnitude(att).to_csv(f"{out}/table_harm_magnitude.csv", index=False)

    tests = pd.concat([
        an.paired_tests(att, "harm_success", "D3_safety_auth", "D4bc_process_aware"),
        an.paired_tests(att, "harm_success", "D4b_master_data", "D4bc_process_aware"),
        an.paired_tests(att, "harm_success", "D4c_spc", "D4bc_process_aware"),
    ])
    tests.to_csv(f"{out}/table_mcnemar.csv", index=False)

    # ---------------- Derived quantities quoted in the text --------------- #
    derived = []

    def _add(name, k, n, note=""):
        lo, hi = an.wilson(k, n)
        derived.append({"quantity": name, "k": k, "n": n, "value": k / n if n else float("nan"),
                        "CI_low": lo, "CI_high": hi, "note": note})

    def _pool(col, classes, dfn):
        x = att[(att.attack_class.isin(classes)) & (att.defence == dfn)][col]
        return int(x.sum()), len(x)

    for dfn in ["D0_none", "D3_safety_auth", "D4b_master_data", "D4c_spc",
                "D4bc_process_aware", "FULL_D3+D4bc"]:
        _add(f"harm_C1-C6_pooled_{dfn}", *_pool("harm_success", PROD_CLASSES, dfn))
    _add("detected_C1-C6_pooled_D3_safety_auth", *_pool("detected", PROD_CLASSES, "D3_safety_auth"))
    _add("detected_S1-S2_pooled_D3_safety_auth",
         *_pool("detected", ["S1_overspeed", "S2_missing_fixture"], "D3_safety_auth"))
    _add("detected_C1-C6_pooled_D4b_master_data", *_pool("detected", PROD_CLASSES, "D4b_master_data"))
    a1 = att[(att.attack_class == "A1_adaptive_offset_in_window") & (att.defence == "D4c_spc")]
    q = a1.spc_alarm_part.dropna()
    for nm, v in [("median", q.median()), ("q25", q.quantile(0.25)), ("q75", q.quantile(0.75)),
                  ("p95", q.quantile(0.95)), ("max", q.max())]:
        derived.append({"quantity": f"A1_D4c_alarm_part_{nm}", "value": v, "n": len(q)})
    a2 = att[(att.attack_class == "A2_adaptive_slow_in_window") & (att.defence == "D0_none")]
    for pid, g in a2.groupby("product"):
        derived.append({"quantity": f"A2_D0_oee_loss_pp_mean_{pid}",
                        "value": float((100 * (g.base_oee - g.oee_ex_defence)).mean()), "n": len(g)})
        _add(f"A2_D0_harm_{pid}", int(g.harm_success.sum()), len(g))
    # Window headroom: upper offset limit + wear expected at the recipe tool
    # change, vs tolerance (quoted in the Discussion; model-derived, no RNG).
    from mfgsec.model import expected_wear_dev
    for p in PRODUCTS.values():
        w = expected_wear_dev(p, p.tool_change_every)
        derived.append({"quantity": f"offset_win_hi_plus_wear_at_tool_change_mm_{p.pid}",
                        "value": p.offset_win[1] + w, "note": f"tolerance {p.tol}"})
    # Recipe roughness check: predicted Ra at the recipe tool change, at nominal
    # feed and at the feed-window top, as a fraction of spec (design rule:
    # window top <= ra_window_margin).
    from mfgsec.model import ra_at_tool_change
    for p in PRODUCTS.values():
        for nm, f in [("nominal", p.feed_nom), ("window_top", p.feed_win[1])]:
            derived.append({"quantity": f"ra_at_tool_change_frac_of_spec_{nm}_{p.pid}",
                            "value": ra_at_tool_change(p, f) / p.ra_max})
        if ra_at_tool_change(p, p.feed_win[1]) > ASSUMPTIONS["ra_window_margin"] * p.ra_max:
            print(f"WARNING: {p.pid} feed-window top violates the Ra window margin")
    # (benign-derived rows are appended after the benign run below)

    # ---------------- Benign corpus -------------------------------------- #
    ben = run_benign(args.n)
    ben.to_csv(f"{out}/raw_benign.csv", index=False)
    rows = []
    for (bt, dfn), g in ben.groupby(["benign_type", "defence"]):
        n = len(g)
        fb = int(g.false_block.sum())
        al = int(g.spc_alarm.sum())
        al_up = int(g.spc_alarm_after_upset.sum())
        rows.append({"benign_type": bt, "defence": dfn, "n": n,
                     "false_block": fb, "false_block_rate": fb / n,
                     "fb_CI_low": an.wilson(fb, n)[0], "fb_CI_high": an.wilson(fb, n)[1],
                     "spc_alarms": al, "spc_alarms_after_upset": al_up,
                     "spc_alarms_no_upset": al - al_up,
                     "oee_mean": g.oee.mean(), "defence_time_mean_min": g.defence_time.mean()})
    pd.DataFrame(rows).to_csv(f"{out}/table_benign_costs.csv", index=False)
    s = ben[ben.defence == "D4c_spc"]
    _add("benign_spc_alarm_all_types_D4c", int(s.spc_alarm.sum()), len(s))
    _add("benign_spc_alarm_without_upset_all_types_D4c",
         int((s.spc_alarm & ~s.spc_alarm_after_upset).sum()), len(s))
    _add("benign_spc_alarms_preceded_by_upset_share", int(s.spc_alarm_after_upset.sum()),
         int(s.spc_alarm.sum()))
    dv = ben[(ben.defence == "D4b_master_data") & (ben.benign_type != "nominal")]
    _add("benign_false_block_legit_deviations_pooled_D4b", int(dv.false_block.sum()), len(dv))
    f = ben[ben.defence == "FULL_D3+D4bc"]
    derived.append({"quantity": "benign_defence_time_min_mean_all_types_FULL",
                    "value": float(f.defence_time.mean()), "n": len(f)})
    pd.DataFrame(derived).to_csv(f"{out}/table_derived.csv", index=False)

    # ---------------- Master-data lag sweep ------------------------------ #
    sweep = []
    for p in [0.0, 0.02, 0.05, 0.10, 0.20]:
        b = run_benign(args.n, p_lag=p, seed=11)
        g = b[(b.defence == "D4b_master_data") & (b.benign_type != "nominal")]
        k, n = int(g.false_block.sum()), len(g)
        lo, hi = an.wilson(k, n)
        sweep.append({"p_lag": p, "n": n, "false_blocks": k, "rate": k / n,
                      "CI_low": lo, "CI_high": hi})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(f"{out}/table_lag_sweep.csv", index=False)

    # ---------------- Sensitivity analysis ------------------------------- #
    base_vals = dict(ASSUMPTIONS)
    # +/-30% on each coefficient. Detection probabilities are varied through
    # their miss rate (1 - p), which keeps them valid and symmetric. Integer
    # thresholds are rounded (2 -> 1 and 3). In-process detection is set by the
    # gauge model, so its noise and failure rate are varied instead of a p.
    # When a roughness coefficient makes the feed recipe violate the recipe
    # roughness rule, the feed recipe is re-derived (rederive_feed_recipe).
    sens_params = ["noise_sigma", "meas_sigma", "delta_wear_max", "p_gauge_fail",
                   "p_detect_final", "p_upset", "k_Ra", "ra0_um", "ewma_L",
                   "harm_oee_loss_min", "harm_nonconf_min"]
    base_products = dict(PRODUCTS)
    sens = []
    for prm in [None] + sens_params:
        for fac in ([1.0] if prm is None else [0.7, 1.3]):
            ASSUMPTIONS.clear(); ASSUMPTIONS.update(base_vals)
            PRODUCTS.clear(); PRODUCTS.update(base_products)
            if prm:
                if prm.startswith("p_detect"):
                    v = 1.0 - (1.0 - base_vals[prm]) * fac
                elif prm == "harm_nonconf_min":
                    v = max(1, round(base_vals[prm] * (0.5 if fac < 1 else 1.5)))
                else:
                    v = base_vals[prm] * fac
                ASSUMPTIONS[prm] = v
            scales = rederive_feed_recipe()
            if scales:
                print(f"sensitivity {prm} x{fac}: feed recipe scaled",
                      {k: round(x, 3) for k, x in scales.items()})
            a = run_attacks(args.n_sens, seed=99)
            b = run_benign(args.n_sens, seed=98)
            pc = a[a.attack_class.isin(PROD_CLASSES)]
            sens.append({
                "parameter": prm or "baseline", "factor": fac,
                "value": ASSUMPTIONS[prm] if prm else "",
                "D0_prod_harm": pc[pc.defence == "D0_none"].harm_success.mean(),
                "D3_prod_harm": pc[pc.defence == "D3_safety_auth"].harm_success.mean(),
                "FULL_prod_harm": pc[pc.defence == "FULL_D3+D4bc"].harm_success.mean(),
                "A1_D4b_harm": a[(a.attack_class == "A1_adaptive_offset_in_window")
                                 & (a.defence == "D4b_master_data")].harm_success.mean(),
                "A1_FULL_harm": a[(a.attack_class == "A1_adaptive_offset_in_window")
                                  & (a.defence == "FULL_D3+D4bc")].harm_success.mean(),
                "C2_D0_harm": a[(a.attack_class == "C2_inspection_omission")
                                & (a.defence == "D0_none")].harm_success.mean(),
                "A2_D0_harm": a[(a.attack_class == "A2_adaptive_slow_in_window")
                                & (a.defence == "D0_none")].harm_success.mean(),
                "A2_FULL_harm": a[(a.attack_class == "A2_adaptive_slow_in_window")
                                  & (a.defence == "FULL_D3+D4bc")].harm_success.mean(),
                "benign_nominal_spc_alarm_no_upset": (
                    lambda g: (g.spc_alarm & ~g.spc_alarm_after_upset).mean())(
                    b[(b.benign_type == "nominal") & (b.defence == "D4c_spc")]),
                **{f"feed_scale_{pid}": scales.get(pid, 1.0) for pid in base_products},
            })
    ASSUMPTIONS.clear(); ASSUMPTIONS.update(base_vals)
    PRODUCTS.clear(); PRODUCTS.update(base_products)
    pd.DataFrame(sens).to_csv(f"{out}/table_sensitivity.csv", index=False)

    # ---------------- Upset-rate sweep ----------------------------------- #
    # p_upset has no citable source; report results across a plausible range.
    sweep_u = []
    for pu in [0.001, 0.002, 0.005, 0.01]:
        ASSUMPTIONS.clear(); ASSUMPTIONS.update(base_vals); ASSUMPTIONS["p_upset"] = pu
        a = run_attacks(args.n_sens, seed=97)
        b = run_benign(args.n_sens, seed=96)
        pc = a[a.attack_class.isin(PROD_CLASSES)]
        bn = b[(b.benign_type == "nominal") & (b.defence == "D4c_spc")]
        sel = lambda c, d: a[(a.attack_class == c) & (a.defence == d)].harm_success.mean()
        sweep_u.append({
            "p_upset": pu,
            "D0_prod_harm": pc[pc.defence == "D0_none"].harm_success.mean(),
            "FULL_prod_harm": pc[pc.defence == "FULL_D3+D4bc"].harm_success.mean(),
            "C2_D0_harm": sel("C2_inspection_omission", "D0_none"),
            "C2_D0_excess_escape_mean": (lambda g: (g.escape - g.base_escape).mean())(
                a[(a.attack_class == "C2_inspection_omission") & (a.defence == "D0_none")]),
            "A1_D4b_harm": sel("A1_adaptive_offset_in_window", "D4b_master_data"),
            "A1_FULL_harm": sel("A1_adaptive_offset_in_window", "FULL_D3+D4bc"),
            "A2_FULL_harm": sel("A2_adaptive_slow_in_window", "FULL_D3+D4bc"),
            "benign_nominal_spc_alarm": bn.spc_alarm.mean(),
            "benign_nominal_spc_alarm_no_upset": (bn.spc_alarm & ~bn.spc_alarm_after_upset).mean(),
        })
    ASSUMPTIONS.clear(); ASSUMPTIONS.update(base_vals)
    pd.DataFrame(sweep_u).to_csv(f"{out}/table_upset_sweep.csv", index=False)

    # ---------------- Figures -------------------------------------------- #
    an.fig_detection(det, f"{out}/figures/fig_detection")
    an.fig_harm_heatmap(harm, f"{out}/figures/fig_harm_heatmap")
    an.fig_spc_lag(att, f"{out}/figures/fig_spc_lag")
    an.fig_lag_sweep(sweep, f"{out}/figures/fig_lag_sweep")

    meta = {"n_per_condition": args.n, "n_sensitivity": args.n_sens,
            "profile": args.profile or "original",
            "runtime_s": round(time.time() - t0, 1),
            "evaluation": "worst-case compromised planner (no LLM in loop)",
            "status": "parameter set v3, reviewed by the manufacturing co-author (2026-10-03)"}
    json.dump(meta, open(f"{out}/run_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
