"""Paired worst-case (compromised-planner) experiments.

Every trial draws one order and one plan; the SAME plan and the SAME random
stream (common random numbers) are evaluated under every defence
configuration, so defences are compared on identical inputs (paired design,
McNemar tests). The correct plan for the same order and seed is also executed
as the per-trial harm baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import ASSUMPTIONS as A
from .plans import ATTACK_CLASSES, BENIGN_TYPES, make_order, correct_plan, attack_plan
from .defences import d3_safety_authorisation, d4b_master_data
from .sim import execute

# name -> (use D3, use D4b, use D4c SPC)
CONFIGS = {
    "D0_none": (False, False, False),
    "D3_safety_auth": (True, False, False),
    "D4b_master_data": (False, True, False),
    "D4c_spc": (False, False, True),
    "D4bc_process_aware": (False, True, True),
    "FULL_D3+D4bc": (True, True, True),
}


def _run_config(cfg, plan, order, seed, correct):
    use_d3, use_d4b, use_spc = cfg
    reasons = []
    blocked = False
    if use_d3:
        ok, r = d3_safety_authorisation(plan, order)
        blocked |= not ok
        reasons += r
    if use_d4b:
        ok, r = d4b_master_data(plan, order)
        blocked |= not ok
        reasons += r
    rng = np.random.default_rng(seed)
    mrng = np.random.default_rng(seed ^ 0x5A5A5A)
    if blocked:
        res = execute(correct, order, rng, spc=use_spc, fallback_plan=correct, meas_rng=mrng)
        # Review cost is charged to the defence, not to the attacker.
        res["oee"] = res["oee"] * res["elapsed"] / (res["elapsed"] + A["t_review"])
        res["downtime"] += A["t_review"]
        res["elapsed"] += A["t_review"]
        res["defence_time"] += A["t_review"]
    else:
        res = execute(plan, order, rng, spc=use_spc, fallback_plan=correct, meas_rng=mrng)
    res["blocked_pre"] = blocked
    res["reasons"] = "; ".join(sorted(set(reasons)))
    return res


def harm_success(r: dict, base: dict) -> bool:
    """Production-harm criterion, paired against the correct plan executed on the same
    random stream (base). Defence review time is excluded via oee_ex_defence."""
    nonconf = r["scrap"] + r["rework"] + r["escape"]
    base_nc = base["scrap"] + base["rework"] + base["escape"]
    return bool(r["safety_incident"]
                or r["escape"] - base["escape"] >= A["harm_escape_min"]
                or nonconf - base_nc >= A["harm_nonconf_min"]
                or base["oee_ex_defence"] - r["oee_ex_defence"] >= A["harm_oee_loss_min"])


def run_attacks(n: int, seed: int = 20260929) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for cls, family in ATTACK_CLASSES.items():
        for t in range(n):
            idx += 1
            order = make_order(rng, idx, "nominal")
            correct = correct_plan(order)
            plan, meta = attack_plan(order, cls, rng)
            exec_seed = int(rng.integers(0, 2**31 - 1))
            base = execute(correct, order, np.random.default_rng(exec_seed), spc=False,
                           fallback_plan=correct)
            for name, cfg in CONFIGS.items():
                r = _run_config(cfg, plan, order, exec_seed, correct)
                nonconf = r["scrap"] + r["rework"] + r["escape"]
                base_nc = base["scrap"] + base["rework"] + base["escape"]
                # Attacker harm, paired against the correct plan on the same
                # random stream; defence review time is excluded.
                harm = harm_success(r, base)
                rows.append({
                    "trial": idx, "attack_class": cls, "family": family,
                    "product": order.product, "defence": name,
                    "blocked_pre": r["blocked_pre"],
                    "spc_alarm_part": r["spc_alarm_part"],
                    # SPC counts as detecting the attack only if it alarmed
                    # while the attack plan itself was executing.
                    "spc_detected": (not r["blocked_pre"]) and r["alarm_on_original_plan"],
                    "detected": r["blocked_pre"] or ((not r["blocked_pre"]) and r["alarm_on_original_plan"]),
                    "harm_success": bool(harm),
                    "safety_incident": r["safety_incident"],
                    "scrap": r["scrap"], "rework": r["rework"], "escape": r["escape"],
                    "produced": r["produced"], "good": r["good"],
                    "elapsed_min": r["elapsed"], "oee": r["oee"],
                    "oee_ex_defence": r["oee_ex_defence"],
                    "base_oee": base["oee"], "base_escape": base["escape"],
                    "defence_time": r["defence_time"],
                    "base_nonconf": base_nc,
                    "reasons": r["reasons"], **{f"meta_{k}": v for k, v in meta.items()},
                })
    return pd.DataFrame(rows)


def run_benign(n: int, p_lag: float | None = None, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for btype in BENIGN_TYPES:
        for t in range(n):
            idx += 1
            order = make_order(rng, idx, btype, p_lag=p_lag)
            plan = correct_plan(order)
            exec_seed = int(rng.integers(0, 2**31 - 1))
            lag = bool(order.exceptions) and order.recorded != order.exceptions
            for name, cfg in CONFIGS.items():
                r = _run_config(cfg, plan, order, exec_seed, plan)
                rows.append({
                    "trial": idx, "benign_type": btype, "product": order.product,
                    "master_data_lag": lag, "defence": name,
                    "false_block": r["blocked_pre"],
                    "spc_alarm": r["spc_alarm_part"] is not None,
                    "spc_alarm_after_upset": r["upset_before_alarm"],
                    "scrap": r["scrap"], "rework": r["rework"], "escape": r["escape"],
                    "elapsed_min": r["elapsed"], "oee": r["oee"],
                    "defence_time": r["defence_time"], "reasons": r["reasons"],
                })
    return pd.DataFrame(rows)
