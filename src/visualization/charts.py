"""Charts of the results, drawn from the files in reports/.

    python -m src.visualization.charts
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import FIGURES, REPORTS
from src.preparation.conditions import NAMES
from src.visualization import style
from src.visualization.style import GREY, GREY_DARK, NEUTRAL, SHADES

RULES = [("per field", "Per-field calibration", SHADES["base"]),
         ("conformal, pooled plants", "Conformal, pooled plants", "#8E8E93"),
         ("conformal, detected weeds", "Conformal, detected weeds", "#6E6E73"),
         ("standard cutoff", "Standard cutoff", "#4A4A4F")]
PROMISES = [0.10, 0.15, 0.20]


def weeds_lost():
    d = pd.read_csv(REPORTS / "weeds_lost.csv").set_index("condition")
    shown = ["clean", "blur_31", "bright_180", "dark_25"]
    segments = [("kept_by_cutoff", "Kept by the standard cutoff", GREY),
                ("dropped_by_cutoff", "Seen, then dropped by the cutoff", SHADES["base"]),
                ("not_seen", "Never seen as weed", GREY_DARK)]
    figure, axis = plt.subplots(figsize=(8, 2.7))
    axis.set_xlim(0, 100)
    axis.set_ylim(-0.6, len(shown) - 0.4)
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.set_yticks(range(len(shown)), [NAMES[c].replace(", 31 px", "").replace(", 25%", "") for c in shown][::-1])
    axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("Share of weeds")
    figure.canvas.draw()
    for i, c in enumerate(shown):
        y, left = len(shown) - 1 - i, 0.0
        for key, _, color in segments:
            v = 100 * d.loc[c, key]
            style.rounded_barh(axis, y, left + 0.4, v - 0.8, color, height=0.62, radius=0.1)
            ink = NEUTRAL["background"] if color == SHADES["base"] else NEUTRAL["ink"]
            axis.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=ink, zorder=3)
            left += v
    axis.set_title("Where weeds are lost")
    style.legend(figure, [("swatch", c, l) for _, l, c in segments], anchor=(0.92, 0.98))
    style.save(figure, FIGURES, "weeds-lost")


def promise(measure, title, label, name, condition="clean"):
    d = pd.read_csv(REPORTS / "promise.csv")
    d = d[d.condition == condition]
    figure, axis = plt.subplots(figsize=(8, 3.0))
    groups, width = np.arange(len(PROMISES)), 0.18
    axis.set_xlim(-0.55, len(PROMISES) - 0.45)
    top = 105 if measure == "fields_kept" else 52
    axis.set_ylim(0, top)
    ticks = [0, 25, 50, 75, 100] if measure == "fields_kept" else [0, 10, 20, 30, 40, 50]
    axis.set_yticks(ticks, [f"{t}%" for t in ticks])
    axis.set_ylabel(label)
    axis.set_xticks(groups, [f"Promise of {int(p * 100)}%" for p in PROMISES])
    figure.canvas.draw()
    for j, (rule, _, color) in enumerate(RULES):
        x = groups + (j - 1.5) * (width + 0.03)
        values = [100 * d[(d.promise == p) & (d.rule == rule)][measure].iloc[0] for p in PROMISES]
        style.rounded_bars(axis, x, values, [color] * len(values), width=width, radius=0.05)
        for xi, v in zip(x, values):
            axis.text(xi, v + top / 50, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5,
                      color=NEUTRAL["ink"] if rule == "per field" else NEUTRAL["secondary"])
    items = [("swatch", c, l) for _, l, c in RULES]
    if measure == "fields_kept":
        axis.axhline(90, color=SHADES["strong"], linestyle=(0, (2, 3)), linewidth=1, zorder=3)
        items.append(("dash", SHADES["strong"], "Target, 9 in 10 fields"))
    axis.set_title(title)
    style.legend(figure, items, anchor=(0.97, 0.98))
    style.save(figure, FIGURES, name)


def threshold_choice():
    d = pd.read_csv(REPORTS / "field_thresholds.csv")
    chosen = d.chosen.iloc[0]
    safe = d.safe.to_numpy()
    # Quarter-decade bins with the chosen threshold as one edge, and one bar for fields at zero
    edges = chosen * 10 ** (np.arange(-4, 9) / 4)
    counts = np.histogram(safe[safe > 0], edges)[0]
    centres = np.sqrt(edges[:-1] * edges[1:])
    zero = int((safe == 0).sum())
    figure, axis = plt.subplots(figsize=(8, 2.8))
    axis.set_xlim(-2.2, len(centres) - 0.4)
    axis.set_ylim(0, counts.max() * 1.18)
    positions = np.r_[-1.6, np.arange(len(centres))]
    values = np.r_[zero, counts]
    colors = [GREY] + [SHADES["base"] if e >= chosen else GREY for e in edges[:-1]]
    figure.canvas.draw()
    style.rounded_bars(axis, positions, values, colors, width=0.72, radius=0.12)
    for x, v in zip(positions, values):
        if v:
            axis.text(x, v + counts.max() * 0.02, str(v), ha="center", va="bottom", fontsize=7.5,
                      color=NEUTRAL["secondary"])
    cut = list(edges).index(chosen) - 0.5
    axis.axvline(cut, color=SHADES["strong"], linewidth=1.4, linestyle=(0, (3, 2.5)), zorder=3)
    decades = [0.001, 0.01, 0.1, 1]
    ticks = [-1.6] + [float(np.interp(np.log10(d), np.log10(edges), np.arange(len(edges)) - 0.5)) for d in decades]
    names = ["Zero"] + [f"{d:g}" for d in decades]
    axis.set_xticks(ticks, names)
    axis.set_ylabel("Calibration fields")
    axis.set_xlabel("Largest threshold that keeps the field within a 10% promise, clean images")
    axis.set_title("Choosing the spray threshold")
    style.legend(figure, [("swatch", SHADES["base"], f"Within the promise ({int((safe >= chosen).sum())} fields)"),
                          ("swatch", GREY, f"Needs a lower threshold ({int((safe < chosen).sum())})"),
                          ("dash", SHADES["strong"], f"Chosen threshold, {chosen:.3f}")], anchor=(0.97, 0.98))
    style.save(figure, FIGURES, "threshold-choice")


def envelope():
    d = pd.read_csv(REPORTS / "promise.csv")
    thresholds = pd.read_csv(REPORTS / "thresholds.csv")
    d = d[d.rule == "per field"].merge(thresholds[["promise", "condition", "achievable"]], on=["promise", "condition"])
    conditions = list(NAMES)
    promises = [0.05, 0.10, 0.15, 0.20]
    figure, axis = plt.subplots(figsize=(8, 3.6))
    axis.set_xlim(-0.5, len(promises) - 0.5)
    axis.set_ylim(len(conditions) - 0.5, -0.5)
    axis.grid(False)
    axis.set_xticks(range(len(promises)), [f"Promise of {int(p * 100)}%" for p in promises])
    axis.set_yticks(range(len(conditions)), [NAMES[c] for c in conditions])
    axis.xaxis.tick_top()
    for spine in axis.spines.values():
        spine.set_visible(False)
    from matplotlib.patches import FancyBboxPatch
    for i, c in enumerate(conditions):
        for j, p in enumerate(promises):
            r = d[(d.condition == c) & (d.promise == p)].iloc[0]
            color = SHADES["soft"] if r.achievable else NEUTRAL["surface"]
            axis.add_patch(FancyBboxPatch((j - 0.46, i - 0.42), 0.92, 0.84, boxstyle="round,pad=0,rounding_size=0.08",
                                          facecolor=color, edgecolor="none", mutation_aspect=1))
            if r.achievable:
                axis.text(j, i - 0.1, f"{100 * r.fields_kept:.0f}% of fields", ha="center", va="center",
                          fontsize=8, color=NEUTRAL["ink"], weight="medium")
                axis.text(j, i + 0.2, f"{100 * r.crops_sprayed:.0f}% of crops sprayed", ha="center", va="center",
                          fontsize=6.8, color=SHADES["strong"])
            else:
                axis.text(j, i, "Not achievable", ha="center", va="center", fontsize=7.5, color=NEUTRAL["tertiary"])
    axis.set_title("Where a promise can be kept", pad=22)
    style.save(figure, FIGURES, "envelope")


def main():
    style.apply()
    weeds_lost()
    promise("fields_kept", "Fields that kept the promise", "Share of field visits", "fields-kept")
    promise("crops_sprayed", "Crops sprayed to keep it", "Share of crop plants sprayed", "crops-sprayed")
    threshold_choice()
    envelope()


if __name__ == "__main__":
    main()
