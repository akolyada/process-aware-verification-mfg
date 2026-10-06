"""Cell master data and process-quality model for a robot-tended CNC turning cell.

Numeric values are literature-typical values for finish turning of steel with
carbide inserts, reviewed by the manufacturing co-author (parameter set v3,
2026-10-03). Every assumption is exported to results/table_assumptions.csv.
Every assumption is collected in `ASSUMPTIONS` so it can be listed in the paper.

Quality relationships used (cite standard sources in the paper):
  * Theoretical surface roughness for a round-nosed tool:
        Ra ~= f^2 / (32 * r_eps)            (f [mm/rev], r_eps [mm], Ra [mm])
    multiplied by an empirical factor k_Ra >= 1 for real-process effects.
  * Taylor tool-life equation:  V * T^n = C   ->  T = (C / V)^(1/n)  [min]
  * Tool-wear-induced diameter growth, linear in consumed tool life:
        delta_wear = delta_wear_max * (consumed_life_fraction)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

# --------------------------------------------------------------------------- #
# Global assumptions (single source of truth; exported to the paper's table)   #
# --------------------------------------------------------------------------- #
ASSUMPTIONS = {
    # Parameter set v3, accepted by the manufacturing co-author (V. Tonkonogyi) on
    # 2026-10-03 as confirmed by the lead author. The
    # previous set is kept as profile "original_v1" in profiles.py.
    "k_Ra": 1.25,                 # empirical factor on the kinematic Ra
    "ra0_um": 0.35,               # additive Ra floor [um] (low-feed correction)
    "ra_wear_frac": 0.30,         # relative Ra growth at end of tool life, linear in wear
    "taylor_n": 0.25,             # carbide on steel
    "taylor_C_mode": "fixed",     # one C for the insert grade and steel
    "taylor_C": 468.0,            # Taylor constant [m/min * min^n]
    "taylor_T_at_nominal": 30.0,  # used only in "per_product" mode (original_v1)
    "delta_wear_max": 0.030,      # diameter growth [mm] at end of tool life (VB 0.15-0.2 mm)
    "noise_sigma": 0.003,         # process dimensional noise, 1 sigma [mm]
    "meas_sigma": 0.002,          # on-machine probe noise, 1 sigma [mm]
    "inproc_detect_mode": "gauge",  # in-process gauge judges the MEASURED diameter
    "p_gauge_fail": 0.015,        # probability the automatic gauge fails to judge a part
    "p_detect_inprocess": 0.90,   # used only in "fixed" detection mode (original_v1)
    "p_detect_final": 0.95,       # prob. manual final inspection flags a nonconformity
    "p_rework_success": 0.8,      # probability a reworkable part is recovered (else scrap)
    "rough_at_size_scrap": True,  # rough part already at size cannot be re-cut -> scrap
    "reorder_mode": "breakage",   # finishing before roughing: part scrapped, insert may break
    "p_reorder_break": 0.25,      # per-part probability the finishing insert breaks
    "stock_allowance_dia_mm": 5.0,     # documentation: bar stock over finished diameter
    "finish_allowance_radial_mm": 0.4, # documentation: radial stock left for the finish pass
    "t_load_unload": 0.6,         # robot load + unload per part [min]
    "t_fixture_setup": 12.0,      # fixture setup per order [min]
    "t_rough": 0.8,               # rough pass per part [min]
    "t_inspect_inprocess": 0.4,   # [min]
    "t_inspect_final": 0.3,       # [min]
    "t_tool_change": 2.0,         # [min]
    "t_review": 15.0,             # human review after a blocked plan or SPC alarm [min]
    "t_crash": 360.0,             # downtime after a safety incident (geometry check) [min]
    "t_tool_failure": 10.0,       # downtime after tool failure [min]
    "alt_machine_slowdown": 1.3,  # machining-time factor on alternate machine M2
    "order_qty": 30,              # parts per order
    "ewma_lambda": 0.2,           # SPC EWMA smoothing
    "ewma_L": 3.0,                # SPC EWMA limit width
    "p_master_data_lag": 0.05,    # prob. a legitimate exception is not yet recorded
    "p_upset": 0.002,             # per-part prob. of a random process upset (no citable source; swept)
    "upset_mag_tol": (2.0, 4.0),  # upset dimensional error as multiple of tolerance
    "ra_window_margin": 0.80,     # recipe rule: feed-window top keeps Ra at tool change <= this x spec
    "ra_concession_margin": 0.85, # concessions keep predicted Ra (at tool change) <= this x spec
    "concession_rule": "at_tool_change",  # "v1": original rule (zero wear, forced above window)
    "ra_noise_sigma": 0.1,        # lognormal sigma of Ra scatter around the predicted value
    "dev_wrong_version": 0.03,    # diameter error from an outdated program version [mm] (not exercised)
    "dev_reorder": 0.30,          # gross geometry error of a reordered part [mm] (seen by gauging/SPC)
    "fixture_err_mean": 0.03,     # wrong-fixture error [mm] (not exercised; runout model is future work)
    "fixture_err_sd": 0.01,
    # Harm criterion (attack vs correct plan on the same random stream).
    "harm_escape_min": 1,         # >= this many extra escaped parts
    "harm_nonconf_min": 2,        # >= this many extra nonconforming parts
    "harm_oee_loss_min": 0.05,    # >= this OEE loss (fraction, i.e. 5 pp)
}
# Outcome rule (logic, not a coefficient): a detected nonconforming part is
# scrap if undersize, rework if oversize or too rough; an undetected one is an
# escape; the part in cut at a tool failure and the part at a crash are scrap.

# Machine (safety) limits: what a safety/interlock verifier knows.
SAFETY_LIMITS = {
    "speed_max": 400.0,   # m/min
    "feed_max": 0.40,     # mm/rev
    "offset_abs_max": 0.05,  # mm, tool-offset adjustment band
    "hold_max": 60.0,     # min, single hold step
}


@dataclass(frozen=True)
class Product:
    pid: str
    diameter: float          # finished diameter [mm]
    length: float            # machined length [mm]
    tol: float               # symmetric diameter tolerance [mm]
    ra_max: float            # surface roughness spec [um]
    nose_radius: float       # tool nose radius [mm]
    speed_nom: float         # m/min
    feed_nom: float          # mm/rev
    offset_nom: float        # mm (recipe centring offset)
    speed_win: tuple         # quality window (min, max)
    feed_win: tuple
    offset_win: tuple
    tool_change_every: int   # parts per tool (recipe)
    fixture: str
    programs: dict           # variant -> program id (base version)
    inprocess_required: bool
    variant_offset: float    # dimensional error from running another variant's program [mm]

    @property
    def taylor_C(self) -> float:
        if ASSUMPTIONS["taylor_C_mode"] == "fixed":
            return ASSUMPTIONS["taylor_C"]
        n = ASSUMPTIONS["taylor_n"]
        return self.speed_nom * ASSUMPTIONS["taylor_T_at_nominal"] ** n


PRODUCTS = {
    "P1": Product("P1", 40.0, 100.0, 0.015, 1.6, 0.8, 200.0, 0.10, -0.004,
                  (180.0, 220.0), (0.08, 0.13), (-0.012, 0.004), 15,
                  "FX-P1", {"A": "NC-P1-A", "B": "NC-P1-B"}, True, 0.10),
    "P2": Product("P2", 80.0, 30.0, 0.020, 3.2, 1.2, 180.0, 0.15, -0.004,
                  (160.0, 200.0), (0.12, 0.22), (-0.014, 0.004), 25,
                  "FX-P2", {"A": "NC-P2-A", "B": "NC-P2-B"}, False, 0.15),
    "P3": Product("P3", 60.0, 60.0, 0.015, 0.8, 1.2, 220.0, 0.08, -0.005,
                  (200.0, 240.0), (0.06, 0.083), (-0.010, 0.003), 12,
                  "FX-P3", {"A": "NC-P3-A", "B": "NC-P3-B"}, True, 0.08),
}

# Routing = ordered list of operations; parameters bound per order at planning.
BASE_ROUTING = ["load_fixture", "select_program", "set_param", "machine_rough",
                "machine_finish", "inspect_inprocess", "unload", "inspect_final"]


def routing_ops(product: Product) -> list[str]:
    ops = list(BASE_ROUTING)
    if not product.inprocess_required:
        ops.remove("inspect_inprocess")
    return ops


@dataclass
class Order:
    """A work order and its (possibly) authorised exceptions.

    `exceptions` is ground truth: what was legitimately approved.
    `recorded` is what the master data contains at planning time
    (may lag reality -> source of false blocks for a master-data verifier).
    """
    order_id: str
    product: str
    variant: str
    qty: int
    m1_down: bool = False
    exceptions: dict = field(default_factory=dict)
    recorded: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Quality physics                                                              #
# --------------------------------------------------------------------------- #
def ra_um(feed: float, nose_radius: float, wear: float = 0.0) -> float:
    """Surface roughness [um]: k_Ra x theoretical value plus an optional floor
    Ra0 (low-feed correction), scaled by optional relative growth with tool wear."""
    return ((ASSUMPTIONS["k_Ra"] * (feed ** 2) / (32.0 * nose_radius) * 1000.0
             + ASSUMPTIONS["ra0_um"]) * (1.0 + ASSUMPTIONS["ra_wear_frac"] * min(wear, 1.0)))


def wear_frac_at_tool_change(product: Product, feed: float) -> float:
    """Consumed tool life at the recipe tool change, at nominal speed and this feed."""
    return min(product.tool_change_every * finish_time_min(product, product.speed_nom, feed)
               / tool_life_min(product, product.speed_nom), 1.0)


def ra_at_tool_change(product: Product, feed: float) -> float:
    """Predicted Ra [um] of the last part before the recipe tool change."""
    return ra_um(feed, product.nose_radius, wear_frac_at_tool_change(product, feed))


def max_feed_for_ra(product: Product, ra_target_um: float) -> float:
    """Largest feed whose predicted Ra at the recipe tool change is <= ra_target_um
    (bisection; Ra increases with feed although wear per part decreases)."""
    lo, hi = 1e-3, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ra_at_tool_change(product, mid) <= ra_target_um else (lo, mid)
    return lo


def rederive_feed_recipe() -> dict:
    """Re-apply the recipe roughness rule after the roughness coefficients change.
    If the feed-window top no longer keeps Ra at the recipe tool change within
    ra_window_margin, the whole feed recipe (window and nominal feed) is scaled down
    by the same factor, as a plant would lower feeds for a rougher process.
    Modifies PRODUCTS in place; returns {pid: scale} for the products changed."""
    import dataclasses
    scales = {}
    for pid, p in list(PRODUCTS.items()):
        top = max_feed_for_ra(p, ASSUMPTIONS["ra_window_margin"] * p.ra_max)
        if top < p.feed_win[1]:
            s = top / p.feed_win[1]
            PRODUCTS[pid] = dataclasses.replace(
                p, feed_nom=p.feed_nom * s, feed_win=(p.feed_win[0] * s, p.feed_win[1] * s))
            scales[pid] = s
    return scales


def feed_for_ra(ra_target_um: float, nose_radius: float) -> float:
    """Inverse of ra_um at zero wear: the feed that gives ra_target_um."""
    kin = max(ra_target_um - ASSUMPTIONS["ra0_um"], 0.0)
    return (kin / 1000.0 / ASSUMPTIONS["k_Ra"] * 32.0 * nose_radius) ** 0.5


def tool_life_min(product: Product, speed: float) -> float:
    n = ASSUMPTIONS["taylor_n"]
    return (product.taylor_C / speed) ** (1.0 / n)


def finish_time_min(product: Product, speed: float, feed: float) -> float:
    """Finish-pass cutting time t = pi*D*L / (1000*V*f)  [min]."""
    return math.pi * product.diameter * product.length / (1000.0 * speed * feed)


def expected_wear_dev(product: Product, part_index_in_tool: int,
                      speed: float | None = None, feed: float | None = None) -> float:
    """Dimensional drift expected from tool wear (used by SPC).

    SPC evaluates the expectation at AUTHORISED parameters (see sim.py), so a
    legitimate concession does not look like a process shift.
    """
    speed = product.speed_nom if speed is None else speed
    feed = product.feed_nom if feed is None else feed
    t = finish_time_min(product, speed, feed)
    frac = part_index_in_tool * t / tool_life_min(product, speed)
    return ASSUMPTIONS["delta_wear_max"] * min(frac, 1.0)
