"""Render virtual nozzle-tip positions on the CT-derived nasal surface."""
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "figures").mkdir(exist_ok=True)
BUILD = ROOT / ".build"
for key, name in {
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache",
    "XDG_CONFIG_HOME": "config",
    "MPLCONFIGDIR": "matplotlib",
    "PYVISTA_USERDATA_PATH": "pyvista",
}.items():
    directory = BUILD / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import numpy as np
import pyvista as pv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch


def main():
    anatomy = (
        ROOT
        / "simulation/anatomical"
    )
    config = json.loads((ROOT / "spray_depth_angle_grid_0_15mm/scan_config.json").read_text())
    roi_path = anatomy / "assets/nc04/olfactory_roi.json"
    geometry = roi_path.parent / json.loads(roi_path.read_text())["geometry"]
    entry = np.asarray(config["entrance_native_mm"], dtype=float)
    pitch = 37.5
    p, y = np.deg2rad([pitch, config["yaw_deg"]])
    axis = np.array([np.cos(p) * np.cos(y), np.sin(p), np.cos(p) * np.sin(y)])
    tips = {depth: entry + depth * axis for depth in (10, 15)}
    mesh = pv.read(Path(os.environ["NASOTWIN_NC04_GEOMETRY"]) if os.environ.get("NASOTWIN_NC04_GEOMETRY") else geometry)
    centres = mesh.cell_centers().points
    selected = (
        (centres[:, 0] > -6) & (centres[:, 0] < 98)
        & (centres[:, 1] > -3) & (centres[:, 1] < 61)
        & (np.abs(centres[:, 2]) < 25)
    )
    surface = mesh.extract_cells(np.flatnonzero(selected)).extract_surface().triangulate()
    far = surface.clip(normal=(0, 0, 1), origin=(0, 0, -4.5), invert=True).decimate(.85)
    near = surface.clip(normal=(0, 0, 1), origin=(0, 0, -4.5), invert=False).decimate(.90)

    def background(bounds, height=1000):
        xmin, xmax, ymin, ymax = bounds
        width = round(height * (xmax - xmin) / (ymax - ymin))
        centre = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        plot = pv.Plotter(off_screen=True, window_size=(width, height), lighting="none")
        plot.set_background("white")
        plot.add_mesh(far, color="#9BB1BD", opacity=.64, ambient=.64,
                      diffuse=.36, smooth_shading=True)
        plot.add_mesh(near, color="#C6D5DD", opacity=.08, ambient=.8)
        plot.enable_parallel_projection()
        plot.camera.position = (centre[0], centre[1], 200)
        plot.camera.focal_point = (centre[0], centre[1], -4)
        plot.camera.up = (0, 1, 0)
        plot.camera.parallel_scale = (ymax - ymin) / 2
        plot.reset_camera_clipping_range()
        pixels = plot.screenshot(return_img=True)
        plot.close()
        return pixels

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    dark, muted = "#243D4B", "#617582"
    colors = {10: "#008F95", 15: "#CF8128"}
    overview = (-5, 99, -9, 63)
    closeup = (1, 29, -3, 25)
    fig = plt.figure(figsize=(15, 7.3), facecolor="white")
    full = fig.add_axes([.035, .19, .595, .65])
    zoom = fig.add_axes([.69, .205, .275, .60])

    for ax, bounds in [(full, overview), (zoom, closeup)]:
        ax.imshow(background(bounds), extent=bounds, origin="upper", zorder=0)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal")
        ax.axis("off")
        line = np.stack([entry - 3 * axis, entry + 19 * axis])
        ax.plot(line[:, 0], line[:, 1], color="#647D89", lw=1.4,
                ls=(0, (4, 3)), zorder=3)
        ax.scatter(*entry[:2], s=48, facecolors="white", edgecolors=dark,
                   linewidths=1.5, zorder=6)
        for depth, point in tips.items():
            ax.scatter(*point[:2], s=100 if ax is zoom else 75,
                       c=colors[depth], edgecolors="white", linewidths=1.7, zorder=6)

    def label(ax, text, point, position, color, align="left", size=12):
        ax.annotate(text, xy=point[:2], xytext=position, ha=align, va="center",
                    fontsize=size, color=color, fontweight="medium",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.93),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.1,
                                    shrinkA=5, shrinkB=7),
                    zorder=8)

    label(full, "10 mm", tips[10], (2, 24), colors[10])
    label(full, "15 mm", tips[15], (34, 10), colors[15])
    label(full, "0 mm reference", entry, (2, -5), muted, size=10)
    label(zoom, "10 mm", tips[10], (9, 15.5), colors[10], size=14)
    label(zoom, "15 mm", tips[15], (22, 20), colors[15], align="center", size=14)
    label(zoom, "0 mm\nvirtual entry", entry, (3, 1), dark, size=11)

    # Identical world-coordinate bounds define the highlighted area and inset.
    full.add_patch(Rectangle((closeup[0], closeup[2]),
                             closeup[1] - closeup[0], closeup[3] - closeup[2],
                             fill=False, edgecolor="#93A8B4", lw=1.1,
                             ls=(0, (4, 3)), zorder=2))
    for y0, y1 in [(closeup[2], 0), (closeup[3], 1)]:
        connector = ConnectionPatch(
            xyA=(closeup[1], y0), coordsA=full.transData,
            xyB=(0, y1), coordsB=zoom.transAxes,
            color="#D7E0E5", lw=.9, zorder=1,
        )
        fig.add_artist(connector)
    for a, b in [((3, -1), (8, -1))]:
        zoom.plot([a[0], b[0]], [a[1], b[1]], color=dark, lw=2.2)
        zoom.text(5.5, -2, "5 mm", ha="center", va="top", color=muted, fontsize=9)
    zoom.add_patch(Rectangle((0, 0), 1, 1, transform=zoom.transAxes,
                             fill=False, edgecolor="#D8E2E7", lw=1.2, clip_on=False))
    full.annotate("", xy=(91, -5), xytext=(80, -5),
                  arrowprops=dict(arrowstyle="->", color=muted, lw=1))
    full.text(85.5, -7, "Posterior", ha="center", va="top", fontsize=9, color=muted)
    full.annotate("", xy=(94, 6), xytext=(94, -5),
                  arrowprops=dict(arrowstyle="->", color=muted, lw=1))
    full.text(96, .5, "Superior", rotation=90, ha="center", va="center",
              fontsize=9, color=muted)

    fig.text(.045, .935, "Where do 10 mm and 15 mm place the spray tip?",
             fontsize=21, fontweight="medium", color=dark)
    fig.text(.045, .888, "CT-based geometry  ·  Pitch 37.5°  ·  Yaw 10°  ·  Side view",
             fontsize=11, color=muted)
    fig.text(.045, .835, "a", fontsize=16, fontweight="bold", color=dark)
    fig.text(.067, .837, "Nasal cavity overview", fontsize=12, color=dark)
    fig.text(.69, .835, "b", fontsize=16, fontweight="bold", color=dark)
    fig.text(.712, .837, "Entry close-up", fontsize=12, color=dark)
    fig.text(.69, .158, "The two tips are 5 mm apart along the axis.",
             fontsize=10, color=muted)
    fig.text(.045, .095, "Depth is measured along the nozzle axis from the current virtual entry reference.",
             fontsize=11, color=dark)
    fig.text(.045, .057, "Markers: virtual tip positions. Model: point source with path constraints. Nostril-plane calibration status: pending.",
             fontsize=9.5, color=muted)
    output = ROOT / "figures/nc04_insertion_10_15mm_positions_EN"
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print("Saved:", output.with_suffix(".png"), flush=True)
    print("Virtual tip coordinates (mm):",
          {d: np.round(pt, 4).tolist() for d, pt in tips.items()}, flush=True)


if __name__ == "__main__":
    main()
