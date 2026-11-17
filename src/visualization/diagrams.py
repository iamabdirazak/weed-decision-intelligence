"""Documentation diagrams: the model as a file, and its architecture.

The layout: a soft frame with a title, a symmetric row of raised
cards, each with an icon, a name and a short description, and small arrows
between them.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
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


def icon_image(axis, cx, cy):
    """A field photograph: soil with rows of plants."""
    rounded(axis, cx - 0.7, cy - 0.5, 1.4, 1.0, FAINT, 0.1)
    for index, (x, y, color) in enumerate(((-0.4, 0.12, SOFT), (0.0, 0.12, SOFT), (0.4, 0.12, SOFT),
                                           (-0.2, -0.32, BASE), (0.25, -0.32, SOFT))):
        leaf(axis, cx + x, cy + y - 0.08, 0.26, color)


def icon_detect(axis, cx, cy):
    """Plants found with boxes, including a faint one that is kept."""
    for x, y, size, color, faint in ((-0.36, 0.2, 0.32, SOFT, False), (0.36, 0.2, 0.32, BASE, False),
                                     (0.0, -0.36, 0.3, BASE, True)):
        leaf(axis, cx + x, cy + y - size * 0.45, size, MUTED if faint else color)
        box = FancyBboxPatch((cx + x - size * 0.62, cy + y - size * 0.55), size * 1.24, size * 1.15,
                             boxstyle="round,pad=0,rounding_size=0.05", facecolor="none",
                             edgecolor=MUTED if faint else STRONG, linewidth=1.4,
                             linestyle=(0, (2, 2)) if faint else "-", zorder=6)
        axis.add_patch(box)


def icon_score(axis, cx, cy):
    """Crop and weed confidence become one share between 0 and 1."""
    rounded(axis, cx - 0.62, cy + 0.18, 0.5, 0.16, SOFT, 0.06)
    rounded(axis, cx - 0.62, cy - 0.06, 0.82, 0.16, BASE, 0.06)
    rounded(axis, cx - 0.62, cy - 0.46, 1.24, 0.1, FAINT, 0.05)
    share = 0.82 / (0.82 + 0.5)
    rounded(axis, cx - 0.62, cy - 0.46, 1.24 * share, 0.1, BASE, 0.05, z=4)
    axis.add_patch(Circle((cx - 0.62 + 1.24 * share, cy - 0.41), 0.1, facecolor=STRONG, edgecolor="none", zorder=5))


def icon_conditions(axis, cx, cy):
    """The same plant, clean, blurred and in low light."""
    for index, (shade, blur) in enumerate(((SOFT, 0), (SOFT, 1), (MUTED, 0))):
        x = cx - 0.66 + index * 0.46
        rounded(axis, x, cy - 0.32, 0.4, 0.64, FAINT if index < 2 else "#22261A", 0.07)
        for offset in ((0.1, 0.05, 0.0) if blur else (0.0,)):
            leaf(axis, x + 0.2, cy - 0.2 + offset, 0.24, MUTED if offset else shade, z=5 if offset == 0 else 4)


def icon_calibrate(axis, cx, cy):
    """One safe threshold per field, and the threshold that keeps most of them within the promise."""
    rng = np.random.default_rng(3)
    xs = np.r_[rng.uniform(-0.62, -0.2, 3), rng.uniform(-0.05, 0.62, 9)]
    for index, x in enumerate(np.sort(xs)):
        axis.add_patch(Circle((cx + x, cy - 0.28 + (index % 3) * 0.2), 0.065,
                              facecolor=MUTED if x < -0.12 else BASE, edgecolor="none", zorder=5))
    axis.plot([cx - 0.12, cx - 0.12], [cy - 0.5, cy + 0.5], color=STRONG, linewidth=1.6,
              linestyle=(0, (2.5, 2)), zorder=6)


def icon_decide(axis, cx, cy):
    """A nozzle spraying the weed and skipping the crop."""
    rounded(axis, cx - 0.2, cy + 0.32, 0.4, 0.22, SOFT, 0.06)
    rng = np.random.default_rng(1)
    for _ in range(14):
        t = rng.uniform(0.1, 1)
        x = rng.uniform(-1, 1) * 0.28 * t
        axis.add_patch(Circle((cx + x, cy + 0.28 - t * 0.55), 0.03, facecolor=STRONG, edgecolor="none", zorder=5))
    leaf(axis, cx, cy - 0.52, 0.3, BASE)
    leaf(axis, cx - 0.58, cy - 0.52, 0.26, MUTED)
    leaf(axis, cx + 0.58, cy - 0.52, 0.26, MUTED)


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
        ("Input", "Field image", icon_image),
        ("Detection", "Faint plants kept", icon_detect),
        ("Weed score", "Weed share per plant", icon_score),
        ("Conditions", "Blur, low light, glare", icon_conditions),
        ("Calibration", "Threshold per condition", icon_calibrate),
        ("Decision", "Spray or skip", icon_decide),
    ], "architecture")


# The model as a file -------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as Shape
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
