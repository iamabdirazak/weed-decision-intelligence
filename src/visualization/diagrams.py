"""Documentation diagrams: the model as a file, and its architecture.

The layout: a soft frame with a title, a symmetric row of raised
cards, each with an icon, a name and a short description, and small arrows
between them.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.path import Path as Shape
from PIL import Image

from src.paths import ROOT
from src.visualization import style

DOCS = ROOT / "docs"
STRONG, BASE, SOFT, MUTED, FAINT = (style.SHADES[k] for k in ("strong", "base", "soft", "muted", "faint"))
FRAME, CARD, ARROW = style.NEUTRAL["background"], style.NEUTRAL["surface"], style.NEUTRAL["tertiary"]

CARD_W, CARD_H, GAP = 2.6, 3.4, 0.41  # portrait cards, like the UAV figure
MARGIN = 0.4  # around the row of cards, equal to CARD_Y
CARD_Y = 0.4
CARD_RADIUS, FRAME_RADIUS = 0.27, 0.38  # in proportion to the UAV figure's cards and frame
HEIGHT = CARD_Y + CARD_H + 1.0  # room above the cards for the title
ICON_Y = CARD_Y + 2.25
NAME_Y, SUBTITLE_Y = CARD_Y + 1.05, CARD_Y + 0.65
MIDDLE = CARD_Y + CARD_H / 2


# Drawing primitives ---------------------------------------------------------

def rounded(axis, x, y, w, h, color, radius=0.12, z=3):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                  facecolor=color, edgecolor="none", zorder=z))


# Icons, each centred on (cx, cy) ------------------------------------------

def leaf(axis, x, y, size, color, angle=0.0, z=5):
    """A small plant: two leaves from one point."""
    for side in (-1, 1):
        a = np.radians(angle + side * 35)
        t = np.linspace(0, 1, 30)
        spine = np.c_[np.sin(a) * t, np.cos(a) * t] * size
        normal = np.array([np.cos(a), -np.sin(a)]) * size * 0.28 * np.sin(np.pi * t)[:, None]
        outline = np.r_[spine + normal, (spine - normal)[::-1]]
        axis.add_patch(Polygon(outline + (x, y), facecolor=color, edgecolor="none", zorder=z))


S = 2.6 / 262  # the first figure's icons were drawn on 262-unit cards


def single_leaf(axis, cx, cy, length, color, z=6):
    """One leaf on a diagonal, with its midrib."""
    t = np.linspace(0, 1, 60)
    half = length * 0.3 * np.sin(np.pi * t) ** 0.9
    along = (t - 0.5) * length
    outline = np.r_[np.c_[along, half], np.c_[along, -half][::-1]]
    turn = np.radians(45)
    rotate = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    axis.add_patch(Polygon(outline @ rotate.T + (cx, cy), facecolor=color, edgecolor="none", zorder=z))
    rib = np.c_[np.linspace(-0.62, 0.3, 2) * length, [0, 0]] @ rotate.T + (cx, cy)
    axis.plot(rib[:, 0], rib[:, 1], color=CARD, linewidth=1.6, solid_capstyle="round", zorder=z + 1)


def at(cx, cy, x, y):
    """A point from the first figure's icon space (y down, centred on 192) into the card."""
    return cx + x * S, cy - (y - 192) * S


def box(axis, cx, cy, x, y, w, h, r, color, alpha=1.0, z=4):
    left, top = at(cx, cy, x, y)
    axis.add_patch(FancyBboxPatch((left, top - h * S), w * S, h * S, boxstyle=f"round,pad=0,rounding_size={r * S}",
                                  facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


def arc(axis, cx, cy, x, y, r, start, end, color, width, alpha=1.0, z=4):
    """An arc in degrees, counter-clockwise from the right, in the icon space."""
    t = np.radians(np.linspace(start, end, 80))
    px, py = at(cx, cy, x + r * np.cos(t), y - r * np.sin(t))
    axis.plot(px, py, color=color, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=z)


def icon_image(axis, cx, cy):
    """Camera frames arriving one after another."""
    cx -= 14 * S
    for x, y, alpha in ((-62, 172, 0.3), (-48, 158, 0.55), (-34, 144, 1.0)):
        box(axis, cx, cy, x, y, 96, 68, 12, BASE, alpha, z=3 + alpha)
    axis.add_patch(Polygon([at(cx, cy, 4, 168), at(cx, cy, 24, 178), at(cx, cy, 4, 188)],
                           facecolor=style.NEUTRAL["ink"], edgecolor="none", zorder=5))


def icon_detect(axis, cx, cy):
    """A viewfinder closing on a leaf."""
    half, reach, r = 0.42, 0.16, 0.09
    for sx in (-1, 1):
        for sy in (-1, 1):
            t = np.radians(np.linspace(0, 90, 20))
            corner = np.c_[half - r + r * np.cos(t), half - r + r * np.sin(t)]
            path = np.r_[[[half, half - reach]], corner, [[half - reach, half]]]
            axis.plot(cx + sx * path[:, 0], cy + sy * path[:, 1], color=BASE, linewidth=2.6,
                      solid_capstyle="round", zorder=5)
    single_leaf(axis, cx, cy, 0.46, STRONG)


def icon_score(axis, cx, cy):
    """Weed shares of a few plants, one bar each, with the weed standing out."""
    heights, width, gap = (0.34, 0.58, 0.9, 0.46), 0.2, 0.1
    left = cx - (len(heights) * width + (len(heights) - 1) * gap) / 2
    base = cy - 0.45
    for index, height in enumerate(heights):
        rounded(axis, left + index * (width + gap), base, width, height, STRONG if index == 2 else SOFT, 0.07, z=4)
    axis.plot([left - 0.08, left + 4 * width + 3 * gap + 0.08], [base - 0.07, base - 0.07], color=MUTED,
              linewidth=1.6, solid_capstyle="round", zorder=4)


def icon_conditions(axis, cx, cy):
    """Speed and light: a sun with motion lines trailing behind it."""
    sx, sy, r = cx + 0.22, cy + 0.05, 0.2
    axis.add_patch(Circle((sx, sy), r, facecolor=BASE, edgecolor="none", zorder=5))
    for angle in np.radians(np.arange(0, 360, 45)):
        axis.plot([sx + 0.3 * np.cos(angle), sx + 0.4 * np.cos(angle)],
                  [sy + 0.3 * np.sin(angle), sy + 0.4 * np.sin(angle)], color=BASE, linewidth=2.2,
                  solid_capstyle="round", zorder=5)
    for dy, length, color in ((0.2, 0.24, MUTED), (0.0, 0.34, SOFT), (-0.2, 0.24, MUTED)):
        axis.plot([cx - 0.7, cx - 0.7 + length], [sy + dy, sy + dy], color=color, linewidth=2.6,
                  solid_capstyle="round", zorder=4)


def icon_calibrate(axis, cx, cy):
    """A range of scores with the threshold set inside it."""
    box(axis, cx, cy, -66, 180, 44, 30, 8, BASE, 0.3)
    box(axis, cx, cy, -22, 180, 44, 30, 0, BASE, 0.6)
    box(axis, cx, cy, 22, 180, 44, 30, 8, BASE)
    for x in (-23.5, 20.5):
        box(axis, cx, cy, x, 152, 3, 86, 1.5, STRONG, z=6)


def icon_decide(axis, cx, cy):
    """A nozzle releasing a fan of spray."""
    rounded(axis, cx - 0.24, cy + 0.3, 0.48, 0.2, SOFT, 0.07, z=5)
    rounded(axis, cx - 0.09, cy + 0.18, 0.18, 0.14, BASE, 0.04, z=5)
    for row, (y, count, spread) in enumerate(((cy + 0.02, 3, 0.16), (cy - 0.18, 5, 0.34), (cy - 0.4, 7, 0.52))):
        for x in np.linspace(-spread, spread, count):
            axis.add_patch(Circle((cx + x, y), 0.045 - row * 0.004, facecolor=STRONG if row < 2 else BASE,
                                  edgecolor="none", zorder=5))


def diagram(title: str, cards: list[tuple], name: str) -> None:
    style.apply()
    width = len(cards) * CARD_W + (len(cards) - 1) * GAP + 2 * MARGIN
    figure, axis = plt.subplots(figsize=(width / 1.45, HEIGHT / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")

    rounded(axis, 0, 0, width, HEIGHT, FRAME, FRAME_RADIUS, z=1)
    axis.text(width / 2, HEIGHT - 0.5, title, ha="center", va="center", fontsize=13,
              fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)

    for index, (name_text, subtitle, draw) in enumerate(cards):
        x = MARGIN + index * (CARD_W + GAP)
        centre = x + CARD_W / 2
        rounded(axis, x, CARD_Y, CARD_W, CARD_H, CARD, CARD_RADIUS, z=2)
        draw(axis, centre, ICON_Y)
        axis.text(centre, NAME_Y, name_text, ha="center", va="center", fontsize=10,
                  fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)
        axis.text(centre, SUBTITLE_Y, subtitle, ha="center", va="center", fontsize=7.4,
                  color=style.NEUTRAL["secondary"], zorder=6)
        if index:
            mid = x - GAP / 2
            axis.add_patch(Polygon([[mid - 0.07, MIDDLE - 0.12], [mid - 0.07, MIDDLE + 0.12], [mid + 0.09, MIDDLE]],
                                   facecolor=ARROW, edgecolor="none", zorder=3))

    style.save(figure, DOCS, name, panel=False)


def architecture() -> None:
    diagram("Model Architecture", [
        ("Input", "Camera frames", icon_image),
        ("Detection", "Faint plants kept", icon_detect),
        ("Scoring", "Weed share per plant", icon_score),
        ("Conditions", "Blur, low light, glare", icon_conditions),
        ("Calibration", "Threshold per condition", icon_calibrate),
        ("Output", "Spray or skip", icon_decide),
    ], "architecture")


# The model as a file -------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch


def fill(axis, patch, colors, angle=90, z=3):
    """Fill a patch with a linear gradient; angle 90 runs bottom to top."""
    axis.add_patch(patch)
    patch.set_facecolor("none")
    patch.set_edgecolor("none")
    x0, y0, x1, y1 = patch.get_path().transformed(patch.get_patch_transform()).get_extents().bounds
    x1, y1 = x0 + x1, y0 + y1
    t = np.radians(angle)
    gx, gy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    ramp = gx * np.cos(t) + gy * np.sin(t)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    image = axis.imshow(ramp, extent=(x0, x1, y0, y1), origin="lower", zorder=z,
                        cmap=LinearSegmentedColormap.from_list("", colors), interpolation="bicubic")
    image.set_clip_path(patch)


def corner(points, centre, radius, start, end):
    angles = np.linspace(np.radians(start), np.radians(end), 12)
    points += [(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)) for a in angles]


def page(x, y, w, h, fold, radius):
    """A document outline with rounded corners and a folded top-right corner."""
    points = []
    corner(points, (x + radius, y + radius), radius, 180, 270)
    corner(points, (x + w - radius, y + radius), radius, 270, 360)
    points += [(x + w, y + h - fold), (x + w - fold, y + h)]
    corner(points, (x + radius, y + h - radius), radius, 90, 180)
    return PathPatch(Shape(points + [points[0]], closed=True))


def squircle(x, y, size, radius):
    return FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={radius}")


def onnx_mark() -> np.ndarray:
    """The ONNX mark from docs/onnx.png, without its wordmark, recoloured in the accent.

    Lines take the brightest shade. Each facet keeps its original shading, mapped
    from the deepest shade for the darkest facet to the base shade for the lightest.
    """
    source = np.asarray(Image.open(DOCS / "onnx.png").convert("RGBA")).astype(float) / 255
    mark = source[:575]                                                        # above the wordmark
    rows, columns = np.where(mark[..., 3] > 0.05)
    mark = mark[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    lightness, alpha = mark[..., :3].mean(axis=2), mark[..., 3]
    line = lightness < 0.5
    facets = LinearSegmentedColormap.from_list("", [MUTED, SOFT, BASE])
    shade = np.clip((lightness - 0.6) / 0.38, 0, 1)                           # facet greys span about 0.6 to 0.98
    image = np.empty_like(mark)
    image[..., :3] = np.where(line[..., None], matplotlib.colors.to_rgb(STRONG), facets(shade)[..., :3])
    image[..., 3] = np.where(line, alpha, alpha * 0.9)
    return image


def model_file() -> None:
    """The released model shown as what it is: a single ONNX file."""
    style.apply()
    width, height = 6 * CARD_W + 5 * GAP + 2 * MARGIN, 7.2
    figure, axis = plt.subplots(figsize=(width / 1.45, height / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    rounded(axis, 0, 0, width, height, FRAME, FRAME_RADIUS, z=1)

    w, h, fold = 2.55, 3.4, 0.62                                              # document proportions, 3:4
    x, y = width / 2 - w / 2, 2.5
    cx = width / 2

    for spread, alpha in ((0.32, 0.04), (0.2, 0.06), (0.1, 0.09), (0.03, 0.14)):   # soft lift
        shadow = page(x - spread, y - spread - 0.12, w + 2 * spread, h + 2 * spread, fold + spread, 0.2 + spread)
        shadow.set(facecolor="black", edgecolor="none", alpha=alpha, zorder=2)
        axis.add_patch(shadow)

    sheet = page(x, y, w, h, fold, 0.2)
    fill(axis, sheet, ["#1D1D20", "#2A2A2E", "#38383D"], z=3)                # lit from above
    rim = page(x, y, w, h, fold, 0.2)
    rim.set(facecolor="none", edgecolor="#4A4A50", linewidth=1, zorder=4)
    axis.add_patch(rim)
    flap = PathPatch(Shape([(x + w - fold, y + h), (x + w - fold, y + h - fold + 0.1),
                            (x + w - fold + 0.1, y + h - fold), (x + w, y + h - fold),
                            (x + w - fold, y + h)], closed=True))
    fill(axis, flap, ["#3E3E44", "#5A5A61"], angle=45, z=4)

    mark = onnx_mark()
    size = 1.3                                                                 # about half the page width
    aspect = mark.shape[0] / mark.shape[1]
    label = 0.3                                                                # cap height of the label
    block = size * aspect + 0.36 + label                                       # mark, gap, label
    top = y + h * 0.47 + block / 2                                             # centred, just below the middle
    axis.imshow(mark, extent=(cx - size / 2, cx + size / 2, top - size * aspect, top),
                zorder=6, interpolation="lanczos")
    axis.text(cx, top - block + label / 2, "ONNX", ha="center", va="center", fontsize=15,
              fontweight="medium", color="#8E8E93", alpha=0.85, zorder=6)    # quieter than the mark

    axis.text(width / 2, 1.7, "Weed Decision Model", ha="center", va="center", fontsize=14,
              fontweight="semibold", color=style.NEUTRAL["ink"], zorder=6)
    axis.text(width / 2, 1.25, "detector.onnx  \u00b7  2.6 million parameters  \u00b7  10.2 MB", ha="center", va="center", fontsize=9,
              color=style.NEUTRAL["secondary"], zorder=6)

    style.save(figure, DOCS, "model", panel=False)


if __name__ == "__main__":
    model_file()
    architecture()
