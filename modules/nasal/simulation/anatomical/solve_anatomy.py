"""Advance droplets through an anatomical velocity field with wall capture.

The archived Calmet mesh supplies the frozen t = 0.14639 s nodal velocity.
Point droplets follow drag and gravity; the first surface intersection records
deposition, and the distal outlet records departures. The NC04 still-air solver
reuses this module's triangle collision, drag, and exponential step routines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyvista as pv
from scipy.special import ndtr, ndtri
import vtk

from prepare_flow import DEFAULT_BOUNDS, FrozenFlow, SNAPSHOT_TIME_S, load_frozen_flow, prepare_grid


NOZZLE_TIP_M = (0.09833178, 0.03628044, 0.01593775)
SPRAY_AXIS = (-0.22242829, 0.58986068, 0.77626247)
AIR_DENSITY = 1.225
AIR_VISCOSITY = 1.81e-5
DROPLET_DENSITY = 1000.0
GRAVITY = np.array([0.0, 0.0, -9.81])


class SurfaceCollision:
    """Conservative voxel broad phase and exact VTK triangle intersections.

    Triangle bounding boxes mark potentially intersected grid cells. Each
    trajectory segment tests its swept bounding box, and VTK resolves the final
    triangle intersection. Segments outside the grid receive a direct query.
    """

    def __init__(self, surface, grid, outlet_z, outlet_xy_bounds, outlet_tolerance, require_outlet=True):
        self.surface = surface.triangulate()
        self.locator = vtk.vtkCellTreeLocator()
        self.locator.SetDataSet(self.surface)
        self.locator.BuildLocator()
        self.origin = np.asarray(grid.origin)
        self.spacing = np.asarray(grid.spacing)
        self.dimensions = np.asarray(grid.dimensions) - 1
        self.occupied = np.zeros(tuple(self.dimensions), dtype=bool)
        triangles = np.asarray(self.surface.faces).reshape(-1, 4)[:, 1:]
        vertices = np.asarray(self.surface.points)[triangles]
        centres = vertices.mean(axis=1)
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        norm = np.linalg.norm(normals, axis=1)
        xy = np.asarray(outlet_xy_bounds).reshape(2, 2)
        self.outlet_cells = (
            (np.max(np.abs(vertices[:, :, 2] - outlet_z), axis=1) <= outlet_tolerance)
            & (np.abs(normals[:, 2]) >= 0.9999 * norm)
            & (norm > 0.0)
            & np.all((centres[:, :2] >= xy[:, 0]) & (centres[:, :2] <= xy[:, 1]), axis=1)
        )
        if require_outlet and not np.any(self.outlet_cells):
            raise RuntimeError("No original surface triangles match the measured distal outlet.")
        low = np.floor((vertices.min(axis=1) - self.origin) / self.spacing).astype(int)
        high = np.floor((vertices.max(axis=1) - self.origin) / self.spacing).astype(int)
        intersects = np.all((high >= 0) & (low < self.dimensions), axis=1)
        low = np.maximum(low[intersects], 0)
        high = np.minimum(high[intersects], self.dimensions - 1)
        for lower, upper in zip(low, high):
            self.occupied[
                lower[0] : upper[0] + 1,
                lower[1] : upper[1] + 1,
                lower[2] : upper[2] + 1,
            ] = True
        self.cell = vtk.vtkGenericCell()
        self.exact_queries = 0

    def candidates(self, start, end):
        lower = np.floor((np.minimum(start, end) - self.origin) / self.spacing).astype(int)
        upper = np.floor((np.maximum(start, end) - self.origin) / self.spacing).astype(int)
        outside = np.any((lower < 0) | (upper >= self.dimensions), axis=1)
        long_segment = np.any(upper - lower > 3, axis=1)
        candidates = outside | long_segment
        ids = np.flatnonzero(~candidates)
        if not len(ids):
            return candidates
        extents = upper[ids] - lower[ids]
        max_extent = np.max(extents, axis=0)
        for dx in range(max_extent[0] + 1):
            for dy in range(max_extent[1] + 1):
                for dz in range(max_extent[2] + 1):
                    offset = np.array([dx, dy, dz])
                    use = np.all(offset <= extents, axis=1)
                    selected = ids[use]
                    index = lower[selected] + offset
                    candidates[selected] |= self.occupied[tuple(index.T)]
        return candidates

    def intersect(self, start, end):
        fractions = np.full(len(start), np.inf)
        points = end.copy()
        cell_ids = np.full(len(start), -1, dtype=np.int64)
        hit_t, sub_id, cell_id = vtk.mutable(0.0), vtk.mutable(0), vtk.mutable(0)
        hit_point, pcoords = [0.0] * 3, [0.0] * 3
        selected = np.flatnonzero(self.candidates(start, end))
        for i in selected:
            hit = self.locator.IntersectWithLine(
                start[i], end[i], 1e-10, hit_t, hit_point, pcoords, sub_id, cell_id, self.cell
            )
            if hit:
                fractions[i] = float(hit_t)
                points[i] = hit_point
                cell_ids[i] = int(cell_id)
        self.exact_queries += len(selected)
        return fractions, points, cell_ids


def drag_rate(velocity, air_velocity, diameter):
    """Compute the spherical Schiller--Naumann drag relaxation rate.

    beta = 18 mu/(rho_p d^2) [Cd Re/24], with the Stokes limit handled exactly.
    """
    reynolds = AIR_DENSITY * diameter * np.linalg.norm(velocity - air_velocity, axis=1) / AIR_VISCOSITY
    correction = np.where(reynolds < 1000.0, 1.0 + 0.15 * reynolds**0.687, 0.44 * reynolds / 24.0)
    return 18.0 * AIR_VISCOSITY * correction / (DROPLET_DENSITY * diameter**2)


def frozen_step(position, velocity, air_velocity, beta, dt):
    beta = beta[:, None]
    decay = np.exp(-beta * dt)
    response = -np.expm1(-beta * dt) / beta
    relative = velocity - air_velocity
    next_position = position + air_velocity * dt + relative * response + GRAVITY * (dt - response) / beta
    next_velocity = air_velocity + relative * decay + GRAVITY * response
    return next_position, next_velocity


def make_injection(args):
    rng = np.random.default_rng(args.seed)
    count = args.parcels
    sigma = np.log(args.sigma_g)
    mu = np.log(args.dv50_um * 1e-6)
    # Equal-volume parcels sample the volume distribution at its specified median.
    quantile = (np.arange(count) + rng.random(count)) / count
    if sigma == 0.0:
        diameter = np.full(count, args.dv50_um * 1e-6)
        if not args.diameter_min_um <= args.dv50_um <= args.diameter_max_um:
            raise ValueError("The monodisperse diameter is outside the specified cutoff.")
    else:
        lower = ndtr((np.log(args.diameter_min_um * 1e-6) - mu) / sigma)
        upper = ndtr((np.log(args.diameter_max_um * 1e-6) - mu) / sigma)
        diameter = np.exp(mu + sigma * ndtri(lower + quantile * (upper - lower)))
    rng.shuffle(diameter)
    volume_weights = np.full(count, 1.0 / count)
    count_weights = diameter**-3
    count_weights /= count_weights.sum()

    axis = np.asarray(args.spray_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(axis, reference)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    centre = np.asarray(args.nozzle_tip_m) + args.breakup_distance_m * axis
    radius = args.release_radius_m * np.sqrt(rng.random(count))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    position = centre + (radius * np.cos(azimuth))[:, None] * tangent + (radius * np.sin(azimuth))[:, None] * bitangent
    cosine = rng.uniform(np.cos(np.deg2rad(args.half_angle_deg)), 1.0, count)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    direction = cosine[:, None] * axis + (sine * np.cos(azimuth))[:, None] * tangent + (sine * np.sin(azimuth))[:, None] * bitangent
    velocity = args.speed_m_s * direction
    release = args.injection_duration_s * (np.arange(count) + 0.5) / count
    return position, velocity, diameter, volume_weights, count_weights, release, axis


def solve(args):
    vtk.vtkSMPTools.Initialize(4)
    start_clock = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading the archived CT-derived volume mesh and frozen velocity...", flush=True)
    volumes = load_frozen_flow(args.assets)
    grid_path = args.output / "velocity_grid.vti"
    requested_bounds = np.asarray(args.bounds_m).reshape(3, 2)
    requested_dimensions = np.ceil((requested_bounds[:, 1] - requested_bounds[:, 0]) / args.spacing_m).astype(int) + 1
    grid = pv.read(grid_path) if grid_path.exists() else None
    if grid is None or not (
        np.array_equal(grid.dimensions, requested_dimensions)
        and np.allclose(grid.origin, requested_bounds[:, 0], rtol=0.0, atol=1e-12)
        and np.allclose(grid.spacing, args.spacing_m, rtol=0.0, atol=1e-12)
    ):
        print("Cell-interpolating the frozen CFD field onto the masked lookup grid...", flush=True)
        grid = prepare_grid(volumes, args.bounds_m, args.spacing_m, grid_path)
    print("Building original-mesh interpolation and triangle-collision locators...", flush=True)
    flow = FrozenFlow(volumes, grid)
    surface = pv.read(args.assets / "published_surface.vtp")
    collision = SurfaceCollision(surface, grid, args.exit_z_m, args.outlet_xy_bounds_m, args.outlet_tolerance_m)
    initial_position, initial_velocity, diameter, volume_weights, count_weights, release_times, axis = make_injection(args)
    _, injection_valid = flow.exact(initial_position)
    if not np.all(injection_valid):
        bad = initial_position[~injection_valid]
        raise RuntimeError(
            f"{len(bad)}/{args.parcels} specified injection positions are outside the archived fluid mesh; "
            f"first invalid position {bad[0].tolist()}. No position was relocated."
        )

    count = args.parcels
    status = np.full(count, -1, dtype=np.int8)
    position = np.full((count, 3), np.nan)
    velocity = np.zeros((count, 3))
    termination_time = np.full(count, np.nan)
    termination_cell = np.full(count, -1, dtype=np.int64)
    times = np.linspace(0.0, args.t_end_s, args.frames)
    positions = np.full((args.frames, count, 3), np.nan, dtype=np.float32)
    history = np.full((args.frames, count), -1, dtype=np.int8)
    current_time = 0.0
    steps = 0
    minimum_dt = np.inf
    maximum_dt = 0.0
    print("Integrating droplets in the frozen field...", flush=True)
    for frame in range(1, args.frames):
        target_time = times[frame]
        while current_time < target_time - 1e-14:
            released = (status == -1) & (release_times <= current_time + 1e-13)
            status[released] = 0
            position[released] = initial_position[released]
            velocity[released] = initial_velocity[released]
            future = release_times[status == -1]
            next_release = float(future.min()) if len(future) else np.inf
            active = np.flatnonzero(status == 0)
            if not len(active):
                current_time = min(target_time, next_release)
                continue
            air, valid = flow.evaluate(position[active])
            if not np.all(valid):
                particle = active[np.flatnonzero(~valid)[0]]
                raise RuntimeError(
                    f"Particle {particle} has no source fluid cell at t={current_time:.9g}, "
                    f"position={position[particle].tolist()}; integration stops instead of inventing flow."
                )
            beta_start = drag_rate(velocity[active], air, diameter[active])
            # Exponential integration is stable for stiff drag. The additional
            # half-relaxation-time cap improves resolution of nonlinear drag.
            dt = min(args.dt_s, 0.5 / float(beta_start.max()), target_time - current_time, next_release - current_time)
            midpoint, midpoint_velocity = frozen_step(position[active], velocity[active], air, beta_start, 0.5 * dt)
            midpoint_air, midpoint_valid = flow.evaluate(midpoint)
            # Retain the last valid air velocity for a wall-crossing segment.
            # The original-mesh intersection determines its first-hit endpoint.
            midpoint_air[~midpoint_valid] = air[~midpoint_valid]
            beta_mid = drag_rate(midpoint_velocity, midpoint_air, diameter[active])
            proposed, proposed_velocity = frozen_step(position[active], velocity[active], midpoint_air, beta_mid, dt)
            wall_fraction, wall_point, wall_cell = collision.intersect(position[active], proposed)
            terminated = np.isfinite(wall_fraction)
            escaped = terminated & collision.outlet_cells[np.maximum(wall_cell, 0)]
            deposited = terminated & ~escaped
            final_position = proposed.copy()
            final_position[terminated] = wall_point[terminated]
            position[active] = final_position
            velocity[active] = proposed_velocity
            status[active[deposited]] = 1
            status[active[escaped]] = 2
            termination_time[active[terminated]] = current_time + dt * wall_fraction[terminated]
            termination_cell[active[terminated]] = wall_cell[terminated]
            # Require a resolved wall or exit intersection for an invalid midpoint.
            if np.any(~midpoint_valid & ~terminated):
                raise RuntimeError("A midpoint left the source fluid mesh without a resolved surface/exit intersection.")
            current_time += dt
            steps += 1
            minimum_dt = min(minimum_dt, dt)
            maximum_dt = max(maximum_dt, dt)
        positions[frame] = position
        history[frame] = status
        if frame % 10 == 0 or frame == args.frames - 1:
            counts = {str(code): int(np.count_nonzero(status == code)) for code in (-1, 0, 1, 2)}
            print(f"t={target_time:.4f} s; particle states={counts}", flush=True)

    fractions_volume = {name: float(volume_weights[status == code].sum()) for code, name in [(-1, "unreleased"), (0, "airborne"), (1, "deposited"), (2, "outlet_exit")]}
    fractions_number = {name: float(count_weights[status == code].sum()) for code, name in [(-1, "unreleased"), (0, "airborne"), (1, "deposited"), (2, "outlet_exit")]}
    assert np.isclose(sum(fractions_volume.values()), 1.0, atol=1e-12)
    assert np.isclose(sum(fractions_number.values()), 1.0, atol=1e-12)
    np.savez_compressed(
        args.output / "trajectories.npz",
        positions_m=positions,
        status=history,
        times_s=times,
        diameter_m=diameter,
        release_times_s=release_times,
        nozzle_tip_m=np.asarray(args.nozzle_tip_m),
        spray_axis=axis,
        parcel_volume_weights=volume_weights,
        parcel_count_weights=count_weights,
        injection_positions_m=initial_position,
        termination_times_s=termination_time,
        termination_surface_cell_id=termination_cell,
        source_snapshot_time_s=np.array(SNAPSHOT_TIME_S),
    )
    scalar_info = {
        "model": "Lagrangian droplets tracked in the archived frozen anatomical CFD velocity snapshot",
        "source_snapshot_time_s": SNAPSHOT_TIME_S,
        "source_volume_parts": len(volumes),
        "source_volume_cells": sum(volume.n_cells for volume in volumes),
        "source_volume_points": sum(volume.n_points for volume in volumes),
        "source_surface_triangles": collision.surface.n_cells,
        "physical_outlet_triangles": int(collision.outlet_cells.sum()),
        "coordinate_units": "m",
        "axes": "x left-right, y posterior, z upward",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "diameter_distribution": "equal-volume parcels; stratified truncated volume-lognormal; requested Dv50 is pretruncation",
        "realized_diameter_min_um": float(diameter.min() * 1e6),
        "realized_diameter_max_um": float(diameter.max() * 1e6),
        "realized_volume_median_um": float(np.median(diameter) * 1e6),
        "count_weighting": "droplet-number fractions use normalized 1/d^3 weighting of equal-volume parcels",
        "payload_weighting": "equal parcel liquid volume; CFU conversion uses uniform bacterial concentration and equal post-spray viability as configured mapping conditions",
        "airflow_interpolation": "VTK source-cell interpolation to masked Cartesian grid; trilinear only for eight valid corners; original-cell fallback otherwise",
        "grid_spacing_m": args.spacing_m,
        "grid_lookup_queries": flow.grid_queries,
        "original_cell_queries": flow.exact_queries,
        "exact_surface_collision_queries": collision.exact_queries,
        "drag": "spherical Schiller-Naumann; Cd=24/Re*(1+0.15*Re^0.687) below Re=1000; Cd=0.44 above",
        "air_density_kg_m3": AIR_DENSITY,
        "air_dynamic_viscosity_pa_s": AIR_VISCOSITY,
        "droplet_density_kg_m3": DROPLET_DENSITY,
        "gravity_m_s2": GRAVITY.tolist(),
        "integration": "exponential midpoint; dt <= requested maximum and half minimum instantaneous drag relaxation time",
        "integration_steps": steps,
        "minimum_step_s": minimum_dt,
        "maximum_step_s": maximum_dt,
        "boundary_policy": "published surface triangles absorb point droplets, except the measured planar distal cap at z=-0.01614967 m within the specified posterior x/y bounds; exact cap intersections count as outlet exits",
        "status_codes": {"-1": "unreleased", "0": "airborne", "1": "deposited", "2": "outlet_exit"},
        "state_parcel_counts": {str(code): int(np.count_nonzero(status == code)) for code in (-1, 0, 1, 2)},
        "volume_fractions": fractions_volume,
        "droplet_number_fractions": fractions_number,
        "elapsed_wall_s": time.monotonic() - start_clock,
        "omissions": ["transient airflow", "two-way particle coupling", "turbulent dispersion beyond frozen resolved flow", "evaporation", "finite-radius wall interception", "rebound", "mucus transport", "species-dependent viability", "buoyancy"],
        "initial_position_policy": "measured nozzle tip plus 3 mm breakup offset by default; every initial position checked in the original fluid mesh; no silent relocation",
    }
    (args.output / "simulation_info.json").write_text(json.dumps(scalar_info, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved trajectories.npz and simulation_info.json in {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--assets", type=Path, default=base / "assets")
    parser.add_argument("--output", type=Path, default=base / "outputs")
    parser.add_argument("--parcels", type=int, default=4000)
    parser.add_argument("--dv50-um", type=float, default=30.0)
    parser.add_argument("--sigma-g", type=float, default=2.08)
    parser.add_argument("--diameter-min-um", type=float, default=2.0)
    parser.add_argument("--diameter-max-um", type=float, default=150.0)
    parser.add_argument("--speed-m-s", type=float, default=18.0)
    parser.add_argument("--half-angle-deg", type=float, default=40.0)
    parser.add_argument("--injection-duration-s", type=float, default=0.08)
    parser.add_argument("--t-end-s", type=float, default=0.3)
    parser.add_argument("--frames", type=int, default=401)
    parser.add_argument("--dt-s", type=float, default=2e-5)
    parser.add_argument("--spacing-m", type=float, default=0.0004)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nozzle-tip-m", nargs=3, type=float, default=NOZZLE_TIP_M)
    parser.add_argument("--spray-axis", nargs=3, type=float, default=SPRAY_AXIS)
    parser.add_argument("--breakup-distance-m", type=float, default=0.003)
    parser.add_argument("--release-radius-m", type=float, default=0.00015)
    parser.add_argument("--bounds-m", nargs=6, type=float, default=DEFAULT_BOUNDS)
    parser.add_argument("--exit-z-m", type=float, default=-0.01614967)
    parser.add_argument("--outlet-xy-bounds-m", nargs=4, type=float, default=(0.089, 0.104, 0.130, 0.139))
    parser.add_argument("--outlet-tolerance-m", type=float, default=2e-5)
    args = parser.parse_args()
    if not (args.parcels > 0 and args.frames >= 2 and 0.0 < args.injection_duration_s <= args.t_end_s):
        parser.error("Require parcels > 0, frames >= 2, and 0 < injection-duration-s <= t-end-s.")
    if not (args.dv50_um > 0 and args.sigma_g >= 1.0 and 0 < args.diameter_min_um < args.diameter_max_um):
        parser.error("Require positive diameter inputs, sigma-g >= 1, and ordered cutoff bounds.")
    if not (args.dt_s > 0 and args.spacing_m > 0 and args.speed_m_s >= 0 and 0 <= args.half_angle_deg <= 90):
        parser.error("Require positive dt/spacing, nonnegative speed, and half-angle within [0, 90].")
    return args


if __name__ == "__main__":
    solve(parse_args())
