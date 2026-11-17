"""One visual language for every figure: dark, flat, legible, one accent.

The figures are drawn for a dark page, in the manner of Apple's dark
appearance: near-black background, soft white text, quiet hairlines, and the
same panels and corners as the author's other repositories.

The accent is olive. Its shades are named by emphasis, not by lightness, so
"strong" is the shade that stands out most against the background. Olive
always marks the model, and grey the rules it is compared with.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ACCENT = "#96AB42"

SHADES = {
    "strong": "#D9E4AE",
    "base": ACCENT,
    "soft": "#6F8131",
    "muted": "#4E5B24",
    "faint": "#323A19",
}

NEUTRAL = {
    "background": "#161618",
    "surface": "#242427",
    "ink": "#F5F5F7",
    "secondary": "#A1A1A6",
    "tertiary": "#6E6E73",
    "hairline": "#333336",
}

GREY, GREY_DARK = "#8E8E93", "#55555A"


def apply() -> None:
    plt.rcParams.update({
        "font.family": ["Inter", "SF Pro Text", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 9,
        "text.color": NEUTRAL["ink"],
        "axes.titlesize": 10,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelcolor": NEUTRAL["secondary"],
        "axes.labelsize": 8.5,
        "axes.edgecolor": NEUTRAL["hairline"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": NEUTRAL["hairline"],
        "grid.linewidth": 0.6,
        "xtick.color": NEUTRAL["secondary"],
        "ytick.color": NEUTRAL["secondary"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 0.9,
        "legend.frameon": False,
        "figure.dpi": 200,
        "figure.facecolor": NEUTRAL["background"],
        "axes.facecolor": NEUTRAL["background"],
    })


def rounded_bars(axis, positions, heights, colors, width: float = 0.62, radius: float = 0.12) -> None:
    """Bars with evenly rounded corners, drawn after the axis limits are final.

    Data units differ on x and y, so the corner is corrected by the axis aspect
    to stay a true circle, and shrinks on short bars so it never overflows.
    """
    box = axis.get_window_extent()
    (x0, x1), (y0, y1) = axis.get_xlim(), axis.get_ylim()
    aspect = ((y1 - y0) / box.height) / ((x1 - x0) / box.width)
    for x, height, color in zip(positions, heights, colors):
        corner = min(radius, abs(height) / aspect / 2, width / 2)
        axis.add_patch(FancyBboxPatch(
            (x - width / 2 + corner, min(0, height) + corner * aspect),
            width - 2 * corner, abs(height) - 2 * corner * aspect,
            boxstyle=f"round,pad={corner}", mutation_aspect=aspect,
            facecolor=color, edgecolor="none", zorder=2,
        ))


def rounded_barh(axis, y, left, width, color, height: float = 0.62, radius: float = 0.12) -> None:
    """One horizontal bar with evenly rounded corners, drawn after the axis limits are final.

    The radius is given in units of the y axis, and corrected by the axis
    aspect so the corner stays a true circle. It shrinks on short bars.
    """
    if width <= 0:
        return
    box = axis.get_window_extent()
    (x0, x1), (y0, y1) = axis.get_xlim(), axis.get_ylim()
    aspect = ((y1 - y0) / box.height) / ((x1 - x0) / box.width)      # y units per x unit, on screen
    corner = min(radius / aspect, width / 2, height / aspect / 2)     # in x units
    axis.add_patch(FancyBboxPatch(
        (left + corner, y - height / 2 + corner * aspect), width - 2 * corner, height - 2 * corner * aspect,
        boxstyle=f"round,pad={corner}", mutation_aspect=aspect, facecolor=color, edgecolor="none", zorder=2,
    ))


class _RoundedSwatch:
    """Draws legend keys as the same rounded squares used in the charts."""

    def __init__(self, colors):
        self.colors = colors if isinstance(colors, (list, tuple)) else [colors]

    def legend_artist(self, legend, original, fontsize, handlebox):
        size = handlebox.height
        gap = size * 0.25
        handlebox.width = len(self.colors) * size + (len(self.colors) - 1) * gap
        for index, color in enumerate(self.colors):
            handlebox.add_artist(FancyBboxPatch(
                (handlebox.xdescent + index * (size + gap), handlebox.ydescent), size, size,
                boxstyle=f"round,pad=0,rounding_size={size * 0.28}",
                facecolor=color, edgecolor="none", transform=handlebox.get_transform(),
            ))


def legend(figure, items: list[tuple[str, object, str]], anchor=(1.0, 1.0)) -> None:
    """A single top-right legend. Each item is (kind, colour or colours, label).

    Kinds: "swatch" for a rounded square (or a row of them), "dot", "dash".
    """
    from matplotlib.lines import Line2D

    handles, labels, handler_map = [], [], {}
    for kind, colors, label in items:
        if kind == "swatch":
            handle = object.__new__(type(f"Swatch{len(handles)}", (), {}))
            handler_map[type(handle)] = _RoundedSwatch(colors)
        elif kind == "dot":
            handle = Line2D([], [], marker="o", linestyle="none", markersize=4,
                            markerfacecolor=colors, markeredgewidth=0, alpha=0.7)
        else:
            handle = Line2D([], [], color=colors, linestyle=(0, (2, 3)), linewidth=1)
        handles.append(handle)
        labels.append(label)

    figure.legend(handles, labels, handler_map=handler_map, loc="lower right",
                  bbox_to_anchor=anchor, ncol=len(items), fontsize=8,
                  handlelength=1.6, handleheight=0.9, columnspacing=1.6, handletextpad=0.6,
                  labelcolor=NEUTRAL["secondary"])


PADDING = 0.3  # inches of background around the content
ROUNDING = 0.018  # corner radius as a share of panel width, as in the UAV figures


def save(figure, directory: Path, name: str, panel: bool = True) -> None:
    """Save as SVG and PNG on a rounded panel, transparent beyond its corners.

    The panel wraps the figure's content with even padding. Diagrams that
    draw their own rounded frame pass panel=False.
    """
    figure.canvas.draw()
    content = figure.get_tightbbox(figure.canvas.get_renderer())
    if panel:
        content = content.padded(PADDING)
        # Placed in figure fractions, which follow the crop when saving; the
        # aspect keeps the corner a true circle on a non-square figure.
        width, height = figure.get_size_inches()
        figure.add_artist(FancyBboxPatch(
            (content.x0 / width, content.y0 / height), content.width / width, content.height / height,
            boxstyle=f"round,pad=0,rounding_size={ROUNDING * content.width / width}", mutation_aspect=width / height,
            transform=figure.transFigure, facecolor=NEUTRAL["background"], edgecolor="none", zorder=-10,
        ))
    figure.patch.set_alpha(0)
    for axis in figure.axes:
        axis.patch.set_alpha(0)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(directory / f"{name}.{extension}", bbox_inches=content, pad_inches=0, transparent=True)
    plt.close(figure)
