"""Discrete-event execution of an accepted plan (SimPy) with the quality model
and the optional SPC outcome monitor (D4c).

Semantics follow model.py. One call executes one order of `qty` parts on one
machine. If D4c raises an alarm, the cell stops for review, the tool is
changed, and the remaining parts are produced with the correct plan.
"""
from __future__ import annotations

import math
import numpy as np
import simpy

from .model import (ASSUMPTIONS as A, PRODUCTS, SAFETY_LIMITS, Order, ra_um,
                    tool_life_min, finish_time_min, expected_wear_dev)


def _plan_state(plan: dict, order: Order) -> dict:
    prod = PRODUCTS[order.product]
    ops = [s["op"] for s in plan["steps"]]
    st = {"ops": ops, "machine": plan["machine"], "holds": 0.0,
          "speed": prod.speed_nom, "feed": prod.feed_nom, "offset": prod.offset_nom,
          "program_ok": True, "version_ok": True, "fixture": None,
          "sampling": 1}
    for s in plan["steps"]:
        if s["op"] == "set_param":
            st.update(speed=s["speed"], feed=s["feed"], offset=s["offset"])
        elif s["op"] == "select_program":
            st["program_ok"] = s["program"] == prod.programs[order.variant]
            st["version_ok"] = s.get("version", 1) == order.exceptions.get("program_version", 1)
        elif s["op"] == "load_fixture":
            st["fixture"] = s["fixture"]
        elif s["op"] == "hold":
            st["holds"] += s["minutes"]
        elif s["op"] == "inspect_final":
            st["sampling"] = s.get("sampling", 1)
    st["has_fixture"] = "load_fixture" in ops
    st["fixture_ok"] = st["fixture"] == prod.fixture
    st["reordered"] = ("machine_rough" in ops and "machine_finish" in ops
                       and ops.index("machine_finish") < ops.index("machine_rough"))
    st["inproc"] = "inspect_inprocess" in ops
    st["final"] = "inspect_final" in ops
    st["unsafe"] = (st["speed"] > SAFETY_LIMITS["speed_max"]) or not st["has_fixture"]
    # Parameters the SPC expectation may trust: commanded values only if they
    # are authorised by master data (quality window or recorded concession).
    f_hi = max(prod.feed_win[1], order.recorded.get("concession_feed", prod.feed_win[1]))
    speed_ok = prod.speed_win[0] <= st["speed"] <= prod.speed_win[1]
    feed_ok = prod.feed_win[0] <= st["feed"] <= f_hi
    st["spc_speed"] = st["speed"] if speed_ok else prod.speed_nom
    st["spc_feed"] = st["feed"] if feed_ok else prod.feed_nom
    return st


def ideal_cycle(prod) -> float:
    """Ideal (value-adding) cycle: handling + rough + nominal finish pass."""
    return A["t_load_unload"] + A["t_rough"] + finish_time_min(prod, prod.speed_nom, prod.feed_nom)


def execute(plan: dict, order: Order, rng: np.random.Generator, spc: bool,
            fallback_plan: dict | None = None,
            meas_rng: np.random.Generator | None = None) -> dict:
    """`rng` drives the process only; `meas_rng` drives SPC measurement noise,
    so enabling SPC does not perturb the process stream (common random numbers)."""
    meas_rng = np.random.default_rng(0) if meas_rng is None else meas_rng
    prod = PRODUCTS[order.product]
    env = simpy.Environment()
    out = {"produced": 0, "good": 0, "rework": 0, "scrap": 0, "escape": 0,
           "downtime": 0.0, "safety_incident": False, "spc_alarm_part": None,
           "tool_failures": 0, "defence_time": 0.0, "upsets": 0,
           "upset_before_alarm": False, "alarm_on_original_plan": False}
    sigma = math.hypot(A["noise_sigma"], A["meas_sigma"])
    lam, L = A["ewma_lambda"], A["ewma_L"]

    def run():
        st = _plan_state(plan, order)
        original = True
        slow = A["alt_machine_slowdown"] if st["machine"] == "M2" else 1.0
        wear = 0.0
        in_tool = 0
        z = 0.0
        n_meas = 0
        variant_sign = rng.choice([-1.0, 1.0])
        if st["has_fixture"]:
            yield env.timeout(A["t_fixture_setup"])
        for i in range(order.qty):
            # Safety incident: crash on first part, stop, review, replan.
            if st["unsafe"]:
                out["safety_incident"] = True
                out["produced"] += 1
                out["scrap"] += 1
                out["downtime"] += A["t_crash"] + A["t_review"]
                yield env.timeout(A["t_crash"] + A["t_review"])
                st = _plan_state(fallback_plan, order)
                original = False
                slow = A["alt_machine_slowdown"] if st["machine"] == "M2" else 1.0
                wear, in_tool = 0.0, 0
                continue
            # Recipe tool change.
            if in_tool >= prod.tool_change_every:
                yield env.timeout(A["t_tool_change"])
                wear, in_tool = 0.0, 0
            # Cycle time.
            tf = finish_time_min(prod, st["speed"], st["feed"])
            t = A["t_load_unload"] + (A["t_rough"] + tf) * slow + st["holds"]
            if st["inproc"]:
                t += A["t_inspect_inprocess"]
            inspected_final = st["final"] and (i % st["sampling"] == 0)
            if inspected_final:
                t += A["t_inspect_final"]
            yield env.timeout(t)
            # Wear and tool failure.
            wear += tf / tool_life_min(prod, st["speed"])
            in_tool += 1
            out["produced"] += 1
            if wear >= 1.0:
                out["tool_failures"] += 1
                out["scrap"] += 1
                out["downtime"] += A["t_tool_failure"]
                yield env.timeout(A["t_tool_failure"])
                wear, in_tool = 0.0, 0
                continue
            # Dimensional deviation and roughness.
            dev = st["offset"] + A["delta_wear_max"] * wear + rng.normal(0, A["noise_sigma"])
            if not st["program_ok"]:
                dev += variant_sign * prod.variant_offset
            if not st["version_ok"]:
                dev += A["dev_wrong_version"]
            if not st["fixture_ok"]:
                dev += abs(rng.normal(A["fixture_err_mean"], A["fixture_err_sd"]))
            if st["reordered"]:
                dev += A["dev_reorder"]   # gross geometry error (also in breakage mode)
            if rng.random() < A["p_upset"]:
                out["upsets"] += 1
                lo, hi = A["upset_mag_tol"]
                dev += rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi) * prod.tol
            ra = ra_um(st["feed"], prod.nose_radius, wear) * rng.lognormal(0.0, A["ra_noise_sigma"])
            nonconf = abs(dev) > prod.tol or ra > prod.ra_max
            detected = False
            if nonconf and st["inproc"]:
                if A["inproc_detect_mode"] == "gauge":
                    # Automatic diameter gauge: flags the part if the MEASURED deviation is
                    # out of tolerance (roughness-only defects are invisible to it), unless
                    # the gauge itself fails. False rejects of good parts are not modelled.
                    measured_dev = dev + rng.normal(0, A["meas_sigma"])
                    detected = abs(measured_dev) > prod.tol and rng.random() >= A["p_gauge_fail"]
                elif rng.random() < A["p_detect_inprocess"]:
                    detected = True
            if nonconf and not detected and inspected_final and rng.random() < A["p_detect_final"]:
                detected = True
            if st["reordered"] and A["reorder_mode"] == "breakage":
                # Finishing insert takes the full roughing stock: the part is lost and
                # the insert breaks with probability p_reorder_break.
                out["scrap"] += 1
                if rng.random() < A["p_reorder_break"]:
                    out["tool_failures"] += 1
                    out["downtime"] += A["t_tool_failure"]
                    yield env.timeout(A["t_tool_failure"])
                    wear, in_tool = 0.0, 0
            elif not nonconf:
                out["good"] += 1
            elif detected:
                rough_at_size = ra > prod.ra_max and abs(dev) <= prod.tol
                if dev < -prod.tol:
                    out["scrap"] += 1        # undersize: cannot be recovered
                elif rough_at_size and A["rough_at_size_scrap"]:
                    out["scrap"] += 1        # at size but rough: re-cut would go undersize
                elif A["p_rework_success"] < 1.0 and rng.random() >= A["p_rework_success"]:
                    out["scrap"] += 1        # re-cut of a small oversize failed (tool rubs)
                else:
                    out["rework"] += 1       # oversize or roughness: re-cut
            else:
                out["escape"] += 1
            # SPC (D4c) on whatever measurements the plan produces.
            measured = st["inproc"] or inspected_final
            if spc and measured and out["spc_alarm_part"] is None:
                resid = dev + meas_rng.normal(0, A["meas_sigma"]) \
                        - (prod.offset_nom + expected_wear_dev(prod, in_tool,
                                                               st["spc_speed"], st["spc_feed"]))
                n_meas += 1
                z = lam * resid + (1 - lam) * z
                lim = L * sigma * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * n_meas)))
                if abs(z) > lim:
                    out["spc_alarm_part"] = i + 1
                    out["upset_before_alarm"] = out["upsets"] > 0
                    out["alarm_on_original_plan"] = original
                    out["downtime"] += A["t_review"] + A["t_tool_change"]
                    out["defence_time"] += A["t_review"] + A["t_tool_change"]
                    yield env.timeout(A["t_review"] + A["t_tool_change"])
                    st = _plan_state(fallback_plan, order)
                    slow = A["alt_machine_slowdown"] if st["machine"] == "M2" else 1.0
                    wear, in_tool, z, n_meas = 0.0, 0, 0.0, 0
                    original = False

    env.process(run())
    env.run()
    out["elapsed"] = env.now
    ic = ideal_cycle(prod)
    out["oee"] = out["good"] * ic / env.now if env.now > 0 else 0.0
    # OEE excluding time the defence itself consumed: isolates attacker harm.
    t_ex = env.now - out["defence_time"]
    out["oee_ex_defence"] = out["good"] * ic / t_ex if t_ex > 0 else 0.0
    return out
