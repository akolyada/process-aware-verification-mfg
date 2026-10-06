"""Named parameter profiles. A profile overrides ASSUMPTIONS and product fields
before a run (`python sim/run.py --profile NAME`, output never in results/).

The defaults in model.py are parameter set v3 (co-author review, 2026-10-03).
"original_v1" reproduces the model and values used before that review, so the
earlier results remain regenerable. The intermediate proposal profiles v2/v3
and their outputs are in git history (commits de6df24, fa07b69).
"""
from __future__ import annotations

import dataclasses

from .model import ASSUMPTIONS, PRODUCTS

PROFILES = {
    "original_v1": {
        "assumptions": {
            "k_Ra": 1.3, "ra0_um": 0.0, "ra_wear_frac": 0.0,
            "taylor_C_mode": "per_product", "delta_wear_max": 0.020,
            "meas_sigma": 0.001, "inproc_detect_mode": "fixed", "p_gauge_fail": 0.0,
            "p_rework_success": 1.0, "rough_at_size_scrap": False,
            "reorder_mode": "deviation", "p_reorder_break": 0.5,
            "t_fixture_setup": 5.0, "t_crash": 60.0, "p_upset": 0.01,
            "concession_rule": "v1",
        },
        "products": {
            "P1": {"offset_win": (-0.010, 0.010), "tool_change_every": 20, "feed_win": (0.08, 0.15)},
            "P2": {"offset_win": (-0.012, 0.012)},
            "P3": {"offset_win": (-0.008, 0.008), "tool_change_every": 20,
                   "offset_nom": -0.003, "tol": 0.012, "nose_radius": 0.8,
                   "feed_win": (0.06, 0.10)},
        },
    },
}


def apply_profile(name: str | None) -> None:
    """Apply a profile in place (ASSUMPTIONS and PRODUCTS are module-level)."""
    if not name:
        return
    prof = PROFILES[name]
    unknown = set(prof["assumptions"]) - set(ASSUMPTIONS)
    if unknown:
        raise KeyError(f"profile {name}: unknown assumptions {sorted(unknown)}")
    ASSUMPTIONS.update(prof["assumptions"])
    for pid, fields in prof["products"].items():
        PRODUCTS[pid] = dataclasses.replace(PRODUCTS[pid], **fields)
