"""Cell architecture diagram (vector). Writes paper/figures/fig_architecture.{svg,pdf,png};
the PNG is only Word's fallback image, never inserted on its own.

    python tools/fig_architecture.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 7, "font.family": "serif", "svg.fonttype": "none", "svg.hashsalt": "mfgsec",
                     "pdf.fonttype": 42})
OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(4.8, 2.75))
ax.set_xlim(0, 100)
ax.set_ylim(0, 57)
ax.axis("off")


def box(x, y, w, h, text, fill="white", dashed=False, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=0.7, edgecolor="black", facecolor=fill,
                                linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(p, q, text=None, off=(0, 1.6), dashed=False, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, linewidth=0.7,
                                 color="black", linestyle="--" if dashed else "-",
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha=ha,
                va="bottom", fontsize=5.5, style="italic")


# Trusted sources (top left) and untrusted channels (bottom left).
box(1, 40, 19, 13, "MES / ERP\nmaster data\n(trust anchor)", fill="0.92", bold=True)
box(1, 23, 19, 11, "Structured\nwork-order\nfields", fill="0.92")
box(1, 3, 19, 14, "Untrusted free text\n(notes, chat,\nOCR labels,\nsupplier docs)",
    dashed=True)

# Prompt assembly with D4a, planner.
box(31, 18, 15, 20, "Prompt\nassembly\n\nD4a: typed\nextraction of\nuntrusted text")
box(51, 23, 11, 10, "LLM task\nplanner", fill="0.85", bold=True)

# Verifiers.
box(67, 40, 14, 13, "D3 safety/\nauthorisation\nverifier")
box(85, 40, 14, 13, "D4b\nmaster-data\nplan verifier")
# Executor and cell.
box(67, 20, 32, 11, "Skill executor: robot + CNC lathe\nwith in-process and final gauging")
box(67, 2, 32, 10, "D4c SPC outcome monitor\n(EWMA on measured diameter)")

arrow((20, 46.5), (31, 34))
ax.text(26.5, 43.5, "master-data\nsummary", ha="left", va="bottom", fontsize=5.5, style="italic")
arrow((20, 28.5), (31, 28.5))
arrow((20, 10), (31, 22))
ax.text(26.5, 11.5, "untrusted\nslots", ha="left", va="top", fontsize=5.5, style="italic")
arrow((46, 28), (51, 28))
arrow((62, 30), (67, 43))
ax.text(63.2, 39, "plan", ha="right", va="bottom", fontsize=5.5, style="italic")
arrow((81, 46.5), (85, 46.5))
arrow((92, 40), (92, 31), "accept", off=(1.0, -1.2), ha="left")
arrow((83, 20), (83, 12), "measurements", off=(1.0, -1.0), ha="left")
arrow((67, 7), (60, 7), dashed=True)
ax.text(59, 7, "alarm: stop,\nreview,\ntool change", ha="right", va="center", fontsize=5.5,
        style="italic")
# Master data also feeds D4b and the SPC expectation.
ax.plot([10.5, 10.5, 92, 92], [53.5, 55.5, 55.5, 53.5], color="black", linewidth=0.6,
        linestyle=":")
ax.text(51, 56, "routing, recipe, windows, inspection plan, recorded exceptions",
        ha="center", va="bottom", fontsize=5.5, style="italic")

fig.tight_layout(pad=0.2)
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"fig_architecture.{ext}", dpi=300,
                    metadata={"svg": {"Date": None}, "pdf": {"CreationDate": None}}.get(ext))
print("wrote", OUT / "fig_architecture.svg")
