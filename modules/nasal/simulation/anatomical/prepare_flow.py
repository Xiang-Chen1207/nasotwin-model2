"""Resample the archived Calmet velocity snapshot for droplet transport.

Read the EnSight volume mesh and nodal velocity at t = 0.14639 s, then build a
masked Cartesian lookup grid. Source-cell interpolation provides velocity values
near walls and outside fully valid grid cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


SNAPSHOT_TIME_S = 0.14639
VELOCITY_NAME = "velocity_m_s"
DEFAULT_BOUNDS = (0.075, 0.12, 0.025, 0.14, -0.02, 0.06)
# Linear and higher-order 3-D VTK cell types used for volume interpolation.
VOLUME_TYPES = (10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 29, 31, 32, 33, 42)


def _leaves(block):
    if isinstance(block, pv.MultiBlock):
        for child in block:
            if child is not None:
                yield from _leaves(child)
    elif block is not None and block.n_cells:
        yield block


def load_frozen_flow(assets: Path):
    """Load the EnSight mesh and instantaneous nodal velocity.

    A static case file selects the geometry and velocity variables. Coordinates
    and nodal velocities retain the source SI units.
    """
    assets = Path(assets)
    case_path = assets / "velocity_snapshot.case"
    case_path.write_text(
        "FORMAT\ntype: ensight gold\n\nGEOMETRY\n"
        "model: fensap.ensi.geo\n\nVARIABLE\n"
        "vector per node: velocity_m_s fensap.ensi.VELOC-000001\n",
        encoding="utf-8",
    )
    reader = pv.EnSightReader(str(case_path))
    source = reader.read()
    volumes = []
    for block in _leaves(source):
        if VELOCITY_NAME not in block.point_data:
            continue
        unstructured = block if isinstance(block, pv.UnstructuredGrid) else block.cast_to_unstructured_grid()
        is_volume = np.isin(unstructured.celltypes, VOLUME_TYPES)
        if not np.any(is_volume):
            continue
        volumes.append(unstructured if np.all(is_volume) else unstructured.extract_cells(is_volume))
    if not volumes:
        raise RuntimeError("The archived case contains no 3-D cells with the requested nodal velocity.")
    return volumes


def prepare_grid(volumes, bounds, spacing_m, output_path: Path):
    """Interpolate source-cell velocities onto a masked Cartesian lookup grid.

    The valid_fluid mask records the domain supported by source volume cells.
    """
    bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
    dimensions = np.ceil((bounds[:, 1] - bounds[:, 0]) / spacing_m).astype(int) + 1
    grid = pv.ImageData(
        dimensions=tuple(dimensions),
        spacing=(spacing_m,) * 3,
        origin=tuple(bounds[:, 0]),
    )
    source = volumes[0] if len(volumes) == 1 else pv.MultiBlock(volumes)
    probe = vtk.vtkCompositeDataProbeFilter()
    probe.SetInputData(grid)
    probe.SetSourceData(source)
    probe.SetCellLocatorPrototype(vtk.vtkCellTreeLocator())
    probe.SetValidPointMaskArrayName("valid_fluid")
    probe.ComputeToleranceOff()
    probe.SetTolerance(1e-10)
    probe.Update()
    sampled = pv.wrap(probe.GetOutput()).copy(deep=False)
    if VELOCITY_NAME not in sampled.point_data:
        raise RuntimeError("VTK did not produce the archived velocity during grid resampling.")
    sampled.field_data["source_snapshot_time_s"] = np.array([SNAPSHOT_TIME_S])
    sampled.save(str(output_path))
    return sampled


class FrozenFlow:
    """Fast valid-cell trilinear lookup, with exact source-cell fallback.

    All eight grid corners must be valid fluid points to use the resampled
    field. Near walls, thin passages, or outside the acceleration grid, cached
    VTK cell locators recover original mesh shape-function interpolation.
    The returned validity mask identifies points supported by source cells.
    """

    def __init__(self, volumes, grid):
        self.volumes = volumes
        self.dimensions = np.asarray(grid.dimensions, dtype=int)
        self.origin = np.asarray(grid.origin, dtype=float)
        self.spacing = np.asarray(grid.spacing, dtype=float)
        shape = tuple(self.dimensions)
        self.velocity = np.asarray(grid.point_data[VELOCITY_NAME]).reshape(shape + (3,), order="F")
        self.valid = np.asarray(grid.point_data["valid_fluid"]).reshape(shape, order="F").astype(bool)
        self.bounds = np.asarray([volume.bounds for volume in volumes]).reshape(-1, 3, 2)
        self.locators = []
        self.source_velocity = []
        for volume in volumes:
            locator = vtk.vtkCellTreeLocator()
            locator.SetDataSet(volume)
            locator.BuildLocator()
            self.locators.append(locator)
            self.source_velocity.append(vtk_to_numpy(volume.GetPointData().GetArray(VELOCITY_NAME)))
        # Reuse source-cell locators across VTK batch probes to accelerate
        # interpolation near thin airway walls.
        self.probes = []
        for volume, locator in zip(self.volumes, self.locators):
            strategy = vtk.vtkCellLocatorStrategy()
            strategy.SetCellLocator(locator)
            strategy.Initialize(volume)
            probe = vtk.vtkProbeFilter()
            probe.SetSourceData(volume)
            probe.SetFindCellStrategy(strategy)
            probe.ComputeToleranceOff()
            probe.SetTolerance(1e-10)
            probe.SetValidPointMaskArrayName("valid_fluid")
            self.probes.append(probe)
        self.grid_queries = 0
        self.exact_queries = 0

    def exact(self, positions):
        result = np.zeros((len(positions), 3), dtype=float)
        valid = np.zeros(len(positions), dtype=bool)
        if len(positions):
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(np.ascontiguousarray(positions,dtype=np.float64),deep=True))
            queries = vtk.vtkPolyData()
            queries.SetPoints(points)
            for probe in self.probes:
                probe.SetInputData(queries)
                probe.Update()
                attributes = probe.GetOutput().GetPointData()
                resolved = vtk_to_numpy(attributes.GetArray("valid_fluid")).astype(bool)
                values = vtk_to_numpy(attributes.GetArray(VELOCITY_NAME))
                take = resolved & ~valid
                result[take] = values[take]
                valid |= resolved
        self.exact_queries += len(positions)
        return result, valid

    def evaluate(self, positions):
        count = len(positions)
        result = np.zeros((count, 3), dtype=float)
        valid = np.zeros(count, dtype=bool)
        coordinates = (positions - self.origin) / self.spacing
        lower = np.floor(coordinates).astype(np.int64)
        fractions = coordinates - lower
        in_grid = np.all((lower >= 0) & (lower < self.dimensions - 1), axis=1)
        candidates = np.flatnonzero(in_grid)
        if candidates.size:
            corner_valid = np.ones(len(candidates), dtype=bool)
            interpolated = np.zeros((len(candidates), 3))
            base = lower[candidates]
            frac = fractions[candidates]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        offset = np.array([dx, dy, dz])
                        corner = base + offset
                        index = tuple(corner.T)
                        corner_valid &= self.valid[index]
                        weight = np.prod(np.where(offset, frac, 1.0 - frac), axis=1)
                        interpolated += weight[:, None] * self.velocity[index]
            good = candidates[corner_valid]
            result[good] = interpolated[corner_valid]
            valid[good] = True
            self.grid_queries += len(good)
        fallback = np.flatnonzero(~valid)
        if fallback.size:
            values, resolved = self.exact(positions[fallback])
            result[fallback] = values
            valid[fallback] = resolved
        return result, valid
