"""Pre-execution plan verifiers.

D3  Representative safety/authorisation verifier, built from the principles of
    published guards (interlock/limit rules in the style of LTL/LTLf plan
    checkers, plus a provenance/authorisation gate). It is NOT a
    reimplementation of any specific system. It knows machine safety limits,
    interlock ordering, the skill library, the machine program library and the
    order's authorisation, but not product-specific quality data.

D4b Process-aware master-data verifier (this paper). Binds the plan to the
    product's routing, program/version, fixture, quality windows and
    inspection plan, honouring only exceptions RECORDED in master data.

Both are deterministic and return (accepted: bool, reasons: list[str]).
The SPC outcome monitor (D4c) runs during execution and lives in sim.py.
"""
from __future__ import annotations

from .model import PRODUCTS, SAFETY_LIMITS, Order, routing_ops

SKILL_LIBRARY = {"load_fixture", "select_program", "set_param", "machine_rough",
                 "machine_finish", "inspect_inprocess", "unload", "inspect_final",
                 "hold", "tool_change"}
ALL_PROGRAMS = {p for prod in PRODUCTS.values() for p in prod.programs.values()}
ALL_FIXTURES = {prod.fixture for prod in PRODUCTS.values()}


def d3_safety_authorisation(plan: dict, order: Order) -> tuple[bool, list[str]]:
    r = []
    ops = [s["op"] for s in plan["steps"]]
    # Authorisation / provenance: plan bound to a valid order for this product.
    if plan["order_id"] != order.order_id or plan["product"] != order.product:
        r.append("order/product binding mismatch")
    if plan["machine"] not in {"M1", "M2"}:
        r.append("unknown machine")
    if plan["machine"] == "M1" and order.m1_down:
        r.append("target machine not operational")
    # Skill library and resource libraries.
    for s in plan["steps"]:
        if s["op"] not in SKILL_LIBRARY:
            r.append(f"unknown skill {s['op']}")
        if s["op"] == "select_program" and s["program"] not in ALL_PROGRAMS:
            r.append("program not in machine library")
        if s["op"] == "load_fixture" and s["fixture"] not in ALL_FIXTURES:
            r.append("unknown fixture")
    # Safety limits.
    for s in plan["steps"]:
        if s["op"] == "set_param":
            if s["speed"] > SAFETY_LIMITS["speed_max"]:
                r.append("speed exceeds machine limit")
            if s["feed"] > SAFETY_LIMITS["feed_max"]:
                r.append("feed exceeds machine limit")
            if abs(s["offset"]) > SAFETY_LIMITS["offset_abs_max"]:
                r.append("offset outside adjustment band")
        if s["op"] == "hold" and s["minutes"] > SAFETY_LIMITS["hold_max"]:
            r.append("hold exceeds limit")
    # Interlocks: clamp before cutting, program and parameters set before
    # cutting, unload after cutting.
    machining = [i for i, o in enumerate(ops) if o.startswith("machine_")]
    if machining:
        first, last = min(machining), max(machining)
        for need in ("load_fixture", "select_program", "set_param"):
            if need not in ops or ops.index(need) > first:
                r.append(f"interlock: {need} must precede machining")
        if "unload" not in ops or ops.index("unload") < last:
            r.append("interlock: unload must follow machining")
    return (not r), r


def d4b_master_data(plan: dict, order: Order) -> tuple[bool, list[str]]:
    r = []
    prod = PRODUCTS[order.product]
    rec = order.recorded
    # Routing and machine.
    allowed_machine = "M2" if rec.get("alt_routing") else "M1"
    if plan["machine"] != allowed_machine:
        r.append(f"machine {plan['machine']} not on authorised routing")
    ops = [s["op"] for s in plan["steps"]]
    if ops != routing_ops(prod):
        r.append("operation sequence deviates from routing")
    for s in plan["steps"]:
        if s["op"] == "load_fixture" and s["fixture"] != prod.fixture:
            r.append("fixture does not match product")
        if s["op"] == "select_program":
            if s["program"] != prod.programs[order.variant]:
                r.append("program does not match product variant")
            if s.get("version", 1) != rec.get("program_version", 1):
                r.append("program version not per master data")
        if s["op"] == "set_param":
            lo, hi = prod.speed_win
            if not lo <= s["speed"] <= hi:
                r.append("speed outside quality window")
            lo, hi = prod.feed_win
            hi = max(hi, rec.get("concession_feed", hi))
            if not lo <= s["feed"] <= hi:
                r.append("feed outside quality window / no concession")
            lo, hi = prod.offset_win
            if not lo <= s["offset"] <= hi:
                r.append("offset outside quality window")
        if s["op"] == "inspect_final" and s.get("sampling", 1) != rec.get("final_sampling", 1):
            r.append("inspection sampling not per master data")
    return (not r), r
