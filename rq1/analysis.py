"""RQ1 statistics, tables and figure (spec §7, §9). Run after run_rq1:

    python -m rq1.analysis [--raw results_rq1/raw_rq1.csv] [--out results_rq1]

Writes table_compliance.csv, table_rq1_mcnemar.csv, table_rq1_benign.csv,
table_rq1_downstream.csv and figures/fig_rq1_compliance.{svg,pdf,png}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
from mfgsec.analysis import wilson, mcnemar_exact, holm, plt, LNCS_WIDTH_IN  # noqa: E402

COND_SHORT = {"plain": "Plain", "d1_hardened": "D1 hardened", "d4a_channel_separation": "D4a"}
CLS_SHORT = {"C1_wrong_variant": "C1", "C2_inspection_omission": "C2",
             "C3_param_outside_quality_window": "C3", "C4_operation_reorder": "C4",
             "C5_throughput_sabotage": "C5", "C6_denial_of_service_holds": "C6"}


def _rate(k, n):
    lo, hi = wilson(k, n)
    return {"n": n, "successes": k, "rate": k / n if n else np.nan, "CI_low": lo, "CI_high": hi}


def compliance(att: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in att.groupby(["model", "condition", "attack_class"]):
        rows.append({"model": keys[0], "condition": keys[1], "attack_class": keys[2],
                     **_rate(int(g.injected_followed.sum()), len(g)),
                     "parse_errors": int(g.parse_error.sum())})
    for keys, g in att.groupby(["model", "condition"]):      # pooled over C1-C6
        rows.append({"model": keys[0], "condition": keys[1], "attack_class": "C1-C6 pooled",
                     **_rate(int(g.injected_followed.sum()), len(g)),
                     "parse_errors": int(g.parse_error.sum())})
    return pd.DataFrame(rows)


def mcnemar(att: pd.DataFrame) -> pd.DataFrame:
    """Paired on trial id: plain vs D1 and plain vs D4a, per model x class; Holm across the grid."""
    rows = []
    for (model, cls), g in att.groupby(["model", "attack_class"]):
        piv = g.pivot_table(index="trial", columns="condition", values="injected_followed",
                            aggfunc="first")
        for other in ("d1_hardened", "d4a_channel_separation"):
            if "plain" not in piv or other not in piv:
                continue
            pr = piv[["plain", other]].dropna().astype(bool)
            b, c, p = mcnemar_exact(pr["plain"].to_numpy(), pr[other].to_numpy())
            rows.append({"model": model, "attack_class": cls, "comparison": f"plain vs {other}",
                         "rate_plain": pr["plain"].mean(), "rate_other": pr[other].mean(),
                         "discordant_plain_only": b, "discordant_other_only": c, "p_value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"].tolist())
    return out


def benign(ben: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in ben.groupby(["model", "condition"]):
        parsed = g[~g.parse_error]
        rows.append({"model": keys[0], "condition": keys[1], "n": len(g),
                     "parse_error_rate": g.parse_error.mean(),
                     "deviates_from_correct_rate": 1 - g.plan_equals_correct.mean(),
                     "D4b_false_block_rate": parsed["D4b_master_data_detected"].mean() if len(parsed) else np.nan,
                     "D2_false_block_rate": parsed["D2_detected"].dropna().astype(bool).mean() if "D2_detected" in parsed and parsed["D2_detected"].notna().any() else np.nan,
                     "D0_harm_rate": parsed["D0_none_harm"].mean() if len(parsed) else np.nan,
                     "latency_s_median": g.latency_s.median()})
    return pd.DataFrame(rows)


def downstream(att: pd.DataFrame) -> pd.DataFrame:
    """Real (not worst-case) plans: detection and harm per defence, over parsed attack trials."""
    defs = ["D0_none", "D3_safety_auth", "D4b_master_data", "FULL_D3+D4bc"]
    if "D2_detected" in att:
        defs.insert(1, "D2")
    rows = []
    for keys, g in att[~att.parse_error].groupby(["model", "condition"]):
        for d in defs:
            gg = g
            if d == "D2":                      # D2 runs only on configured conditions
                gg = g[g["D2_detected"].notna()]
                if gg.empty:
                    continue
            det = gg[f"{d}_detected"].astype(bool) if f"{d}_detected" in gg else pd.Series(False, index=gg.index)
            harm = gg[f"{d}_harm"].astype(bool)
            rows.append({"model": keys[0], "condition": keys[1], "defence": d,
                         "n": len(gg), "detected": int(det.sum()),
                         "detected_rate": det.mean(), "harm": int(harm.sum()),
                         **{f"harm_{k}": v for k, v in _rate(int(harm.sum()), len(gg)).items()
                            if k in ("rate", "CI_low", "CI_high")}})
    return pd.DataFrame(rows)


def fig_compliance(comp: pd.DataFrame, stem: str):
    pooled = comp[comp.attack_class == "C1-C6 pooled"]
    models = sorted(pooled.model.unique())
    conds = [c for c in COND_SHORT if c in set(pooled.condition)]
    fig, ax = plt.subplots(figsize=(LNCS_WIDTH_IN * 0.75, 2.2))
    w = 0.8 / max(len(conds), 1)
    x = np.arange(len(models))
    shades, hatches = ["0.85", "0.55", "0.2"], ["", "///", "..."]
    for j, c in enumerate(conds):
        r = pooled[pooled.condition == c].set_index("model").reindex(models)
        err = np.clip(np.vstack([r.rate - r.CI_low, r.CI_high - r.rate]), 0, None)
        ax.bar(x + (j - (len(conds) - 1) / 2) * w, r.rate, w, yerr=err, capsize=1.5,
               color=shades[j % 3], hatch=hatches[j % 3], edgecolor="black", linewidth=0.5,
               label=COND_SHORT[c], error_kw={"linewidth": 0.6})
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Injection followed\n(C1–C6 pooled)"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=len(conds), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300,
                    metadata={"svg": {"Date": None}, "pdf": {"CreationDate": None}}.get(ext))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results_rq1" / "raw_rq1.csv"))
    ap.add_argument("--out", default=str(ROOT / "results_rq1"))
    a = ap.parse_args()
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.raw)
    att, ben = df[df.attack_class != "benign"], df[df.attack_class == "benign"]
    comp = compliance(att); comp.to_csv(out / "table_compliance.csv", index=False)
    mcnemar(att).to_csv(out / "table_rq1_mcnemar.csv", index=False)
    benign(ben).to_csv(out / "table_rq1_benign.csv", index=False)
    downstream(att).to_csv(out / "table_rq1_downstream.csv", index=False)
    by = []
    for factor in ("channel", "style"):
        for keys, g in att.groupby(["model", "condition", factor]):
            by.append({"model": keys[0], "condition": keys[1], "factor": factor, "level": keys[2],
                       **_rate(int(g.injected_followed.sum()), len(g))})
    pd.DataFrame(by).to_csv(out / "table_rq1_by_channel_style.csv", index=False)
    fig_compliance(comp, str(out / "figures" / "fig_rq1_compliance"))
    print(comp[comp.attack_class == "C1-C6 pooled"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
