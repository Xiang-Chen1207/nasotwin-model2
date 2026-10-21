"""Render anterior and lateral views of the CT-derived nasal geometry."""
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "animations").mkdir(exist_ok=True)
RUNTIME = ROOT / ".build" / f"structure-render-{os.getuid()}"
for variable, name in {
    "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
    "MPLCONFIGDIR": "mpl", "PYVISTA_USERDATA_PATH": "pyvista",
}.items():
    directory = RUNTIME / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    anatomy = ROOT / "simulation/anatomical"
    config_path = anatomy / "assets/nc04/olfactory_roi.json"
    config = json.loads(config_path.read_text())
    mesh = pv.read(Path(os.environ["NASOTWIN_NC04_GEOMETRY"]) if os.environ.get("NASOTWIN_NC04_GEOMETRY") else config_path.parent / config["geometry"])
    centres = mesh.cell_centers().points
    display = (
        (centres[:, 0] > -6) & (centres[:, 0] < 98)
        & (centres[:, 1] > -3) & (centres[:, 1] < 61)
        & (np.abs(centres[:, 2]) < 25)
    )
    bounds = np.asarray(config["bounds_mm"]["standard"]).reshape(3, 2)
    roi = np.all((centres >= bounds[:, 0]) & (centres <= bounds[:, 1]), axis=1)

    def surface(mask, reduction):
        return (
            mesh.extract_cells(np.flatnonzero(mask))
            .extract_surface().triangulate().decimate(reduction)
        )

    walls = surface(display & ~roi, .90)
    target = surface(display & roi, .75)
    plot = pv.Plotter(off_screen=True, window_size=(1240, 590), lighting="light kit")
    plot.set_background("white")
    plot.add_mesh(walls, color="#A4BAC7", opacity=.25, smooth_shading=True,
                  ambient=.48, diffuse=.52, specular=.06)
    plot.add_mesh(target, color="#8957AE", opacity=.85, smooth_shading=True,
                  ambient=.55, diffuse=.45)
    plot.enable_parallel_projection()
    centre = np.array([46., 29., 0.])
    plot.camera.focal_point = centre
    plot.camera.up = (0, 1, 0)
    plot.camera.parallel_scale = 36
    from matplotlib.font_manager import findfont
    font_path = findfont("DejaVu Sans")
    title_font = ImageFont.truetype(str(font_path), 27)
    text_font = ImageFont.truetype(str(font_path), 19)
    small_font = ImageFont.truetype(str(font_path), 17)
    frames, stills = [], {}
    phases = np.linspace(0, 1, 60)
    for index, phase in enumerate(phases):
        smooth = phase * phase * (3 - 2 * phase)
        theta = smooth * np.pi / 2
        # X points posteriorly: the anterior camera starts on the negative-X side.
        plot.camera.position = centre + 210 * np.array([-np.cos(theta), 0., np.sin(theta)])
        plot.reset_camera_clipping_range()
        plot.render()
        pixels = plot.screenshot(return_img=True)
        if index in (0, 30, 59):
            stills[index] = pixels.copy()
        canvas = Image.new("RGB", (1280, 720), "white")
        canvas.paste(Image.fromarray(pixels).convert("RGB"), (20, 76))
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 20), "CT-derived nasal geometry", font=title_font, fill="#243D4B")
        view = "Anterior view" if index == 0 else (
            "Lateral view" if index == 59 else "Anterior to lateral")
        draw.text((1250, 28), view, anchor="ra", font=text_font, fill="#617582")
        for x, color, text in [
            (30, "#A4BAC7", "Nasal walls"),
            (250, "#8957AE", "Candidate target region"),
        ]:
            draw.ellipse((x, 680, x + 12, 692), fill=color)
            draw.text((x + 21, 673), text, font=small_font, fill="#526575")
        frames.append(canvas)
        if index in (0, 20, 40, 59):
            print(f"Rendered frame {index + 1}/60", flush=True)
    plot.close()
    durations = [50] * len(frames)
    durations[0] += 500
    durations[-1] += 500
    animation = ROOT / "animations/nc04_structure_front_to_side_EN.gif"
    frames[0].save(animation, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 4))
    for ax, index, label in zip(
        axes, (0, 30, 59), ("A  Anterior", "B  Oblique", "C  Lateral")
    ):
        pixels = stills[index]
        foreground = np.any(pixels[:, :, :3] < 244, axis=2)
        yy, xx = np.nonzero(foreground)
        crop = pixels[max(0, yy.min()-12):yy.max()+13, max(0, xx.min()-12):xx.max()+13]
        ax.imshow(crop)
        ax.axis("off")
        ax.set_title(label, loc="left", color="#243D4B", fontsize=12, pad=12)
    fig.subplots_adjust(left=.02, right=.99, top=.86, bottom=.03, wspace=.10)
    poster = ROOT / "figures/nc04_structure_views_EN"
    for extension in ("png", "pdf"):
        fig.savefig(poster.with_suffix("." + extension), dpi=200, facecolor="white")
    plt.close(fig)
    caption = (
        "**CT-derived nasal geometry viewed from anterior to lateral.** "
        "Gray-blue surfaces show the NC04 nasal geometry; purple highlights the fixed "
        "candidate olfactory region used for deposition analysis. Transparent walls expose "
        "internal structure. The ROI is a candidate envelope defined by anatomical "
        "coordinates and the recorded geometric boundary rules. The four-second animation rotates the "
        "camera smoothly through 90 degrees, with brief holds at the anterior and lateral "
        "views. Playback time describes the camera rotation through the geometry.\n"
    )
    animation.with_suffix(".md").write_text(caption)
    print("GIF COMPLETE", animation, "duration_ms", sum(durations), flush=True)
    print("PDF poster:", poster.with_suffix(".pdf"), flush=True)


if __name__ == "__main__":
    main()
