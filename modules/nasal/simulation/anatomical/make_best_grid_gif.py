"""Simulate and render a three-second replay of the highest-ranked spray setting."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, help="Path to the separately supplied NC04 STL.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()
    if args.geometry:
        os.environ['NASOTWIN_NC04_GEOMETRY'] = str(args.geometry.resolve())
    grid = args.config.parent
    # Store rendering caches in a user-specific project directory.
    runtime = grid.parent / ".build" / f"gif-runtime-{os.getuid()}"
    for variable, subdir in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl",
        "PYVISTA_USERDATA_PATH": "pyvista",
    }.items():
        path = runtime / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)

    import numpy as np
    import pyvista as pv
    from PIL import Image, ImageDraw, ImageFont
    import nc04_angles as solver

    cfg = json.loads(args.config.read_text())
    with (grid / "deposition_grid.csv").open() as stream:
        best = max((r for r in csv.DictReader(stream) if r["status"] == "positive"),
                   key=lambda r: float(r["mean_pct"]))
    with (grid / "seed_results.csv").open() as stream:
        candidates = [r for r in csv.DictReader(stream) if r["case"] == best["case"]]
    representative = min(candidates,
                         key=lambda r: abs(float(r["standard_volume_pct"]) - float(best["mean_pct"])))
    seed = int(representative["seed"])
    depth, pitch = float(best["depth_mm"]), float(best["pitch_deg"])
    axis = solver.axis_native(pitch, cfg["yaw_deg"])
    entry = np.array(cfg["entrance_native_mm"])
    tip = entry + depth * axis
    output = grid / "best_animation"
    output.mkdir(exist_ok=True)
    name = f"depth{depth:g}_pitch{pitch:g}_seed{seed}_dense_replay"
    overrides = {k: cfg[k] for k in
                 ["cone_full_angle_deg", "rr_characteristic_diameter_um", "speed_m_s"]}
    times = np.r_[np.linspace(0, .01, 201), np.linspace(.012, .3, 145)]
    if not args.render_only:
        solver.initialize_worker()
        result = solver.simulate(
            (pitch, cfg["yaw_deg"], cfg["parcels"], seed,
             cfg["flight_time_s"], cfg["maximum_time_step_s"]),
            nozzle_tip=tip, case_name=name, output_dir=output,
            sample_count=cfg["parcels"], frame_times=times, spray_overrides=overrides)
        if not result["valid"]:
            raise RuntimeError("The selected replay injection is outside the fluid domain.")
        if args.simulate_only:
            print("TRAJECTORY COMPLETE", output / f"{name}.npz", flush=True)
            return
    # Dense output partitions the integration steps. Replay metrics therefore
    # come from this replay, while the displayed grid mean remains unchanged.
    data = np.load(output / f"{name}.npz")
    positions = data["positions_native_mm"]
    paths = data["sample_paths_native_mm"]
    hit_time = data["hit_time_s"]
    deposited = (data["status"] == 1) & np.isfinite(hit_time)
    target_hits = data["roi_hit_mask"][1] & deposited
    replay_target_pct = 100 * float(target_hits.mean())
    ids = np.random.default_rng(2026).choice(cfg["parcels"], 1200, replace=False)
    roi_cfg = json.loads(solver.CONFIG.read_text())
    mesh = pv.read(solver.geometry_path(roi_cfg))
    centres = mesh.cell_centers().points
    mask = ((centres[:, 0] > -6) & (centres[:, 0] < 98)
            & (centres[:, 1] > -3) & (centres[:, 1] < 61)
            & (np.abs(centres[:, 2]) < 25))
    display = mesh.extract_cells(np.flatnonzero(mask)).extract_surface().triangulate()
    far = display.clip(normal=(0, 0, 1), origin=(0, 0, -4.5), invert=True).decimate(.88)
    near = display.clip(normal=(0, 0, 1), origin=(0, 0, -4.5), invert=False).decimate(.9)
    roi = solver.roi_masks(mesh, roi_cfg)["standard"] & (centres[:, 2] <= 0)
    target = mesh.extract_cells(np.flatnonzero(roi)).extract_surface().triangulate().decimate(.75)

    def scene(size, centre, scale):
        plot = pv.Plotter(off_screen=True, window_size=size, lighting="none")
        plot.set_background("white")
        plot.add_mesh(far, color="#94AEBB", opacity=.58, ambient=.65,
                      diffuse=.35, smooth_shading=True)
        plot.add_mesh(near, color="#D4E0E6", opacity=.06, ambient=.8)
        plot.add_mesh(target, color="#9271B6", opacity=.4, ambient=.7)
        plot.add_mesh(pv.Line(entry - 4 * axis, tip),
                      color="#35586C", line_width=3)
        plot.add_mesh(pv.Sphere(radius=.65, center=tip), color="#D89A48")
        plot.enable_parallel_projection()
        plot.camera.position = (centre[0], centre[1], 200)
        plot.camera.focal_point = (centre[0], centre[1], -4)
        plot.camera.up = (0, 1, 0)
        plot.camera.parallel_scale = scale
        plot.reset_camera_clipping_range()
        pixels = plot.screenshot(return_img=True)
        plot.close()
        return Image.fromarray(pixels).convert("RGB")

    full_size, zoom_size = (850, 545), (390, 390)
    full_centre = np.array([47., 28.])
    zoom_centre = np.median(positions[target_hits, :2], axis=0)
    full_base = scene(full_size, full_centre, 37)
    zoom_base = scene(zoom_size, zoom_centre, 13)
    from matplotlib.font_manager import findfont
    font_path = findfont("DejaVu Sans")
    font = ImageFont.truetype(str(font_path), 23)
    small = ImageFont.truetype(str(font_path), 17)
    tiny = ImageFont.truetype(str(font_path), 14)
    bold = ImageFont.truetype(str(font_path), 27)
    colors = {"flight": "#008E9C", "target": "#7A439F", "wall": "#A1AFB9"}

    def projection(points, size, centre, scale):
        factor = size[1] / (2 * scale)
        return np.c_[size[0] / 2 + (points[:, 0] - centre[0]) * factor,
                     size[1] / 2 - (points[:, 1] - centre[1]) * factor]

    def paint_particles(base, size, centre, scale, frame, tm, radius):
        panel = base.copy()
        draw = ImageDraw.Draw(panel)
        points = projection(paths[frame, ids], size, centre, scale)
        for (x, y), pid in zip(points, ids):
            if not (0 <= x < size[0] and 0 <= y < size[1]):
                continue
            settled = deposited[pid] and hit_time[pid] <= tm
            color = colors["target"] if settled and target_hits[pid] else (
                colors["wall"] if settled else colors["flight"])
            r = radius if not settled else .85 * radius
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        return panel

    requests = np.r_[np.linspace(0, .004, 41),
                     np.linspace(.0045, .01, 12), np.linspace(.012, .3, 7)]
    frame_indices = np.clip(np.searchsorted(times, requests), 0, len(times)-1)
    frames = []
    for index in frame_indices:
        tm = float(times[index])
        canvas = Image.new("RGB", (1280, 720), "white")
        full = paint_particles(full_base, full_size, full_centre, 37, index, tm, 1.6)
        zoom = paint_particles(zoom_base, zoom_size, zoom_centre, 13, index, tm, 2.1)
        canvas.paste(full, (5, 100))
        canvas.paste(zoom, (870, 180))
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 18),
                  f"Nasal spray deposition | best within {cfg['depth_min_mm']}-{cfg['depth_max_mm']} mm",
                  font=bold, fill="#25394A")
        draw.text((30, 61), f"Insertion {depth:g} mm   |   Pitch {pitch:g}°   |   Full cone 10°",
                  font=font, fill="#526575")
        draw.text((880, 109), "Target-region close-up", font=font, fill="#25394A")
        draw.text((880, 148), "Candidate olfactory ROI", font=small, fill=colors["target"])
        draw.rectangle((869, 179, 1261, 571), outline="#D9E2E7", width=2)
        draw.text((880, 585), f"Grid mean: {float(best['mean_pct']):.2f}% (3 seeds)",
                  font=small, fill="#25394A")
        fraction = 100 * np.count_nonzero(target_hits & (hit_time <= tm)) / cfg["parcels"]
        draw.text((880, 614), f"Replay target deposition: {fraction:.2f}%",
                  font=small, fill=colors["target"])
        draw.rectangle((0, 648, 1280, 720), fill="white")
        for x, text, color in [(30, "Airborne", colors["flight"]),
                               (180, "Target deposited", colors["target"]),
                               (410, "Other wall", colors["wall"])]:
            draw.ellipse((x, 663, x+10, 673), fill=color)
            draw.text((x+18, 655), text, font=small, fill="#526575")
        draw.text((810, 655), f"Simulation time: {tm*1000:6.2f} ms",
                  font=font, fill="#25394A")
        draw.text((30, 691), f"CT-based model  |  seed {seed}  |  1,200 / 10,000 parcels shown"
                  "  |  slowed, variable-speed replay", font=tiny, fill="#647781")
        frames.append(canvas)
    durations = [40] * len(frames)
    durations[0] += 200
    durations[-1] += 400
    depth_tag = f"{depth:g}".replace(".", "p")
    pitch_tag = f"{pitch:g}".replace(".", "p")
    destination = grid.parent / f"animations/nc04_best_{depth_tag}mm_{pitch_tag}deg_3s_EN.gif"
    frames[0].save(destination, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    print("GIF COMPLETE", destination, "duration_ms", sum(durations),
          "replay_target_pct", replay_target_pct, flush=True)


if __name__ == "__main__":
    main()
