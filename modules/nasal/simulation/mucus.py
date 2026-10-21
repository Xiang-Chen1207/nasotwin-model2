"""FiPy finite-volume mucus transport with reversible surface attachment.

The fluid state stores CFU in nz*nx cells and the surface state stores CFU in
nx axial cells. Inputs use SI units; cumulative exposure is reported in CFU*h.
FiPy assembles anisotropic diffusion and upwind advection. SciPy's sparse matrix
exponential advances the coupled linear system, including clearance, biological
change, and exposure integral states. A regular mesh converts FiPy concentration
coefficients to the cell-count representation."""
import os
os.environ.setdefault("FIPY_SOLVERS", "scipy")

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from fipy import (Grid2D, CellVariable, FaceVariable, TransientTerm,
                  DiffusionTerm, UpwindConvectionTerm, LinearLUSolver)


def transport_operator(params):
    nx, nz = int(params["nx"]), int(params["nz"])
    dx, dz = params["length_m"] / nx, params["thickness_m"] / nz
    mesh = Grid2D(nx=nx, ny=nz, dx=dx, dy=dz)
    c = CellVariable(mesh=mesh, value=0.0, hasOld=True)
    velocity = FaceVariable(mesh=mesh, rank=1, value=(params["velocity_m_s"], 0.0))
    # Apply closed diffusive boundaries and append the convective outlet below.
    velocity.setValue(0.0, where=mesh.exteriorFaces)
    tensor = (((params["diffusion_parallel_m2_s"], 0.0),
               (0.0, params["diffusion_perpendicular_m2_s"])),)
    equation = (TransientTerm(var=c) == DiffusionTerm(coeff=tensor, var=c)
                - UpwindConvectionTerm(coeff=velocity, var=c))
    equation.cacheMatrix()
    equation.sweep(var=c, dt=1.0, solver=LinearLUSolver())
    volumes = np.asarray(mesh.cellVolumes)
    # FiPy's cached system is M/dt - K. Recover M^-1 K at dt=1 second.
    operator = sparse.eye(nx*nz, format="csr") - sparse.diags(1/volumes) @ equation.matrix.matrix
    right = np.arange(nz)*nx + nx-1
    outflow = np.zeros(nx*nz)
    outflow[right] = params["velocity_m_s"] / dx
    operator = operator - sparse.diags(outflow)
    return operator.tocsr(), outflow, dx, dz


def coupled_operator(transport, outflow, params, biology, da, r_free, r_attached):
    nx, nz = int(params["nx"]), int(params["nz"])
    n = nx*nz
    dz = params["thickness_m"] / nz
    kappa = da * params["thickness_m"] / biology["attachment_reference_s"]
    kdet = biology["detachment_s_inv"]
    direct = biology["direct_attached_clearance_s_inv"]
    # Robin flux uses the wall concentration, obtained by eliminating the
    # bottom half-cell diffusion resistance. Both directions share this flux.
    beta = 1.0 / (1.0 + kappa*dz/(2*params["diffusion_perpendicular_m2_s"]))
    capture, release = beta*kappa/dz, beta*kdet
    bottom = np.arange(nx)
    sink = np.zeros(n)
    sink[bottom] = capture
    to_surface = sparse.csr_matrix((np.full(nx, capture), (bottom, bottom)), shape=(nx, n))
    to_mobile = sparse.csr_matrix((np.full(nx, release), (bottom, bottom)), shape=(n, nx))
    free = transport + sparse.diags(np.full(n, r_free)-sink)
    attached = sparse.eye(nx, format="csr")*(r_attached-release-direct)
    living = sparse.bmat([[free, to_mobile], [to_surface, attached]], format="csr")
    # Independent integral states: outflow (including direct removal), net
    # biological change, and area under the culturable-count curve in hours.
    observation_rows = sparse.csr_matrix(np.vstack([
        np.r_[outflow, np.full(nx, direct)],
        np.r_[np.full(n, r_free), np.full(nx, r_attached)],
        np.ones(n+nx)/3600.0
    ]))
    full = sparse.bmat([[living, sparse.csr_matrix((n+nx, 3))],
                        [observation_rows, sparse.csr_matrix((3, 3))]], format="csr")
    return full, kappa


def run_mucus(config, spray):
    p, biology = config["mucus"], config["biology"]
    nx, nz = int(p["nx"]), int(p["nz"])
    n = nx*nz
    times = np.linspace(0.0, p["duration_s"], int(p["output_frames"]))
    transport, outflow, dx, dz = transport_operator(p)
    # Bin each deposited parcel's volume/CFU weight into the top mucus layer.
    # The initial attached population is zero; deposition is an initial condition.
    xdep = np.asarray(spray["deposition_x_m"])
    weights = np.asarray(spray["deposition_weight"])
    valid = np.isfinite(xdep) & (weights > 0)
    dep, _ = np.histogram(xdep[valid], bins=np.linspace(0,p["length_m"],nx+1), weights=weights[valid])
    initial = np.zeros(n+nx+3)
    initial[(nz-1)*nx:nz*nx] = dep
    case_totals, case_attached, case_final_clear, case_exposure = [], [], [], []
    baseline = None
    max_error = 0.0
    minimum = np.inf
    for da in biology["attachment_Da"]:
        cached = {}
        per_strain = []
        for strain in config["strains"]:
            rates = (strain["net_growth_free_h_inv"]/3600,
                     strain["net_growth_attached_h_inv"]/3600)
            if rates not in cached:
                matrix, kappa = coupled_operator(transport,outflow,p,biology,da,*rates)
                cached[rates] = expm_multiply(matrix, initial, start=0,
                    stop=p["duration_s"], num=len(times), endpoint=True,
                    traceA=float(matrix.diagonal().sum()))
            per_strain.append(cached[rates]*strain["emitted_cfu"]*strain["post_spray_culturability"])
        result = np.stack(per_strain, axis=1)  # time, strain, state
        mobile = result[:,:,:n].reshape(len(times),4,nz,nx)
        attached = result[:,:,n:n+nx]
        cleared, net_bio, exposure = result[:,:,-3], result[:,:,-2], result[:,:,-1]
        total = mobile.sum(axis=(2,3)) + attached.sum(axis=2)
        initial_counts = total[0]
        error = np.max(np.abs(total+cleared-net_bio-initial_counts) / np.maximum(initial_counts,1))
        max_error = max(max_error,float(error))
        minimum = min(minimum,float(np.min(result[:,:,:n+nx])))
        case_totals.append(total)
        case_attached.append(attached.sum(axis=2))
        case_final_clear.append(cleared[-1])
        case_exposure.append(exposure[-1])
        print(f"Da={da:g}: retained60min={total[-1].sum():.6g} CFU, "
              f"attached60min={attached[-1].sum():.6g}, balance={error:.3g}",flush=True)
        if np.isclose(da,biology["baseline_Da"]):
            baseline = dict(mobile_cfu=mobile, attached_cfu=attached,
                cleared_cfu=cleared, net_bio_cfu=net_bio, exposure_cfu_h=exposure,
                initial_deposited_cfu=initial_counts)
    if baseline is None:
        raise ValueError("baseline_Da must be included in attachment_Da")
    baseline.update(mucus_times_s=times, x_m=(np.arange(nx)+0.5)*dx,
        z_m=(np.arange(nz)+0.5)*dz, mucus_thickness_m=p["thickness_m"],
        channel_width_m=p["width_m"], case_names=np.array([f"Da={x:g}" for x in biology["attachment_Da"]]),
        case_total_cfu=np.asarray(case_totals),case_attached_cfu=np.asarray(case_attached),
        case_final_cleared_cfu=np.asarray(case_final_clear),case_exposure_cfu_h=np.asarray(case_exposure),
        max_relative_balance_error=max_error,minimum_cfu=minimum)
    return baseline
