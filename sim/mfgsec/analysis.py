"""Statistics (Wilson CIs, exact McNemar, Holm), tables and vector figures."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none", "svg.hashsalt": "mfgsec", "pdf.fonttype": 42})
LNCS_WIDTH_IN = 4.8  # LNCS text width ~12.2 cm

SHORT = {"C1_wrong_variant": "C1 Wrong variant",
         "C2_inspection_omission": "C2 Inspection omission",
         "C3_param_outside_quality_window": "C3 Param. outside window",
         "C4_operation_reorder": "C4 Operation reorder",
         "C5_throughput_sabotage": "C5 Throughput sabotage",
         "C6_denial_of_service_holds": "C6 DoS holds",
         "S1_overspeed": "S1 Overspeed",
         "S2_missing_fixture": "S2 Missing fixture",
         "A1_adaptive_offset_in_window": "A1 Adaptive offset",
         "A2_adaptive_slow_in_window": "A2 Adaptive slow"}
DEF_SHORT = {"D0_none": "D0", "D3_safety_auth": "D3", "D4b_master_data": "D4b",
             "D4c_spc": "D4c", "D4bc_process_aware": "D4b+c", "FULL_D3+D4bc": "Full"}


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired booleans. Returns (b, c, p): b = x&~y, c = ~x&y."""
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def rate_table(df: pd.DataFrame, col: str, family_col="family") -> pd.DataFrame:
    rows = []
    for (cls, dfn), g in df.groupby(["attack_class", "defence"]):
        k, n = int(g[col].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"model": "worst-case compromised planner", "defence": dfn,
                     "attack_class": cls, "setting": g[family_col].iloc[0],
                     "n": n, "successes": k, "rate": k / n, "CI_low": lo, "CI_high": hi})
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, col: str, a: str, b: str) -> pd.DataFrame:
    rows = []
    for cls, g in df.groupby("attack_class"):
        ga = g[g.defence == a].sort_values("trial")[col].to_numpy(bool)
        gb = g[g.defence == b].sort_values("trial")[col].to_numpy(bool)
        bb, cc, p = mcnemar_exact(ga, gb)
        rows.append({"attack_class": cls, "comparison": f"{a} vs {b}", "metric": col,
                     "rate_a": ga.mean(), "rate_b": gb.mean(),
                     "discordant_a_only": bb, "discordant_b_only": cc, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].tolist())
    return out


def harm_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["nonconf"] = d.scrap + d.rework + d.escape
    d["excess_nonconf"] = d.nonconf - d.base_nonconf
    d["excess_escape"] = d.escape - d.base_escape
    d["oee_loss_pp"] = 100 * (d.base_oee - d.oee_ex_defence)
    g = d.groupby(["attack_class", "defence"])
    return g.agg(excess_nonconf_mean=("excess_nonconf", "mean"),
                 excess_escape_mean=("excess_escape", "mean"),
                 oee_loss_pp_mean=("oee_loss_pp", "mean"),
                 safety_incident_rate=("safety_incident", "mean")).reset_index()


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def fig_detection(rates: pd.DataFrame, path_stem: str):
    defs = ["D3_safety_auth", "D4b_master_data", "D4bc_process_aware"]
    classes = list(SHORT)
    fig, ax = plt.subplots(figsize=(LNCS_WIDTH_IN, 2.8))
    w = 0.26
    x = np.arange(len(classes))
    hatches = ["", "///", "..."]
    shades = ["0.85", "0.55", "0.2"]
    for j, dfn in enumerate(defs):
        r = rates[rates.defence == dfn].set_index("attack_class").loc[classes]
        err = np.clip(np.vstack([r.rate - r.CI_low, r.CI_high - r.rate]), 0, None)
        ax.bar(x + (j - 1) * w, r.rate, w, yerr=err, capsize=1.5, color=shades[j],
               edgecolor="black", linewidth=0.5, hatch=hatches[j],
               label=DEF_SHORT[dfn], error_kw={"linewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in classes], rotation=40, ha="right")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.08)
    ax.axvline(5.5, color="0.4", linewidth=0.6, linestyle="--")
    ax.axvline(7.5, color="0.4", linewidth=0.6, linestyle="--")
    ax.text(2.5, 1.03, "production-semantic", ha="center", fontsize=7)
    ax.text(6.5, 1.03, "safety", ha="center", fontsize=7)
    ax.text(8.5, 1.03, "adaptive", ha="center", fontsize=7)
    ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300,
                    metadata={"svg": {"Date": None}, "pdf": {"CreationDate": None}}.get(ext))
    plt.close(fig)


def fig_harm_heatmap(rates: pd.DataFrame, path_stem: str):
    defs = list(DEF_SHORT)
    classes = list(SHORT)
    m = rates.pivot(index="attack_class", columns="defence", values="rate").loc[classes, defs]
    fig, ax = plt.subplots(figsize=(LNCS_WIDTH_IN, 3.0))
    # pcolormesh (not imshow) keeps every cell a vector shape in SVG/PDF.
    im = ax.pcolormesh(np.arange(len(defs) + 1) - 0.5, np.arange(len(classes) + 1) - 0.5,
                       m.to_numpy(), cmap="Greys", vmin=0, vmax=1,
                       edgecolors="white", linewidth=0.5, rasterized=False)
    ax.set_xlim(-0.5, len(defs) - 0.5)
    ax.set_ylim(len(classes) - 0.5, -0.5)
    ax.set_xticks(range(len(defs)))
    ax.set_xticklabels([DEF_SHORT[d] for d in defs], rotation=30, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([SHORT[c] for c in classes])
    for i in range(len(classes)):
        for j in range(len(defs)):
            v = m.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.solids.set_rasterized(False)
    cb.set_label("Attack success (production harm)")
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300,
                    metadata={"svg": {"Date": None}, "pdf": {"CreationDate": None}}.get(ext))
    plt.close(fig)


def fig_spc_lag(df: pd.DataFrame, path_stem: str):
    d = df[(df.attack_class == "A1_adaptive_offset_in_window") & (df.defence == "D4c_spc")].copy()
    d["bin"] = pd.cut(d["meta_offset_fraction_of_window"], bins=np.linspace(0.3, 1.0, 8))
    g = d.groupby("bin", observed=True).agg(
        lag=("spc_alarm_part", "median"), q1=("spc_alarm_part", lambda s: s.quantile(0.25)),
        q3=("spc_alarm_part", lambda s: s.quantile(0.75)),
        harm=("harm_success", "mean"))
    mids = [iv.mid for iv in g.index]
    fig, ax = plt.subplots(figsize=(LNCS_WIDTH_IN * 0.62, 2.1))
    ax.errorbar(mids, g.lag, yerr=np.clip([g.lag - g.q1, g.q3 - g.lag], 0, None), fmt="o-", color="black",
                markersize=3, linewidth=0.8, capsize=2)
    ax.set_xlabel("Adversarial offset\n(fraction of upper window limit)")
    ax.set_ylabel("SPC alarm at part no.\n(median, IQR)")
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300,
                    metadata={"svg": {"Date": None}, "pdf": {"CreationDate": None}}.get(ext))
    plt.close(fig)


def fig_lag_sweep(sweep: pd.DataFrame, path_stem: str):
    fig, ax = plt.subplots(figsize=(LNCS_WIDTH_IN * 0.62, 2.1))
    ax.errorbar(100 * sweep.p_lag, 100 * sweep.rate,
                yerr=np.clip([100 * (sweep.rate - sweep.CI_low), 100 * (sweep.CI_high - sweep.rate)], 0, None),
                fmt="s-", color="black", markersize=3, linewidth=0.8, capsize=2)
    ax.set_xlabel("Master-data lag probability (%)")
    ax.set_ylabel("False-block rate on\nlegitimate deviations (%)")
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300,
                    metadata={"svg": {"Date": None}, "pdf": {"CreationDate": None}}.get(ext))
    plt.close(fig)
