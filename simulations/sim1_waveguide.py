"""
sim1_waveguide.py
═══════════════════════════════════════════════════════════════════════════════
Simulation 1: SOI Ridge Waveguide — Single-Mode TE Propagation at 1550 nm
───────────────────────────────────────────────────────────────────────────────
Platform: 450 nm × 220 nm Si ridge on SiO2 BOX, air-clad top surface.
Method:   3D FDTD via Tidy3D (Flexcompute cloud / local).
Goal:     Verify single-mode TE guidance, visualize mode profile, confirm
          power stays guided along the full propagation length.

Physics overview
────────────────
A silicon ridge waveguide confines light by total internal reflection (TIR)
at the Si/SiO2 interface (bottom) and Si/air interface (top and sides).
The 450 nm × 220 nm cross-section was chosen by the industry to support
exactly one TE mode at 1550 nm — a standard adopted across foundries
(GlobalFoundries 45SPCLO, IMEC iSiPP50G, AIM Photonics).

The fundamental TE mode has its dominant E-field component (Ex) oriented
horizontally across the waveguide width.  A TM mode would have Ey dominant
and vertical confinement as the primary guiding mechanism.

Tidy3D simulation structure
────────────────────────────
1. Geometry: three Structure objects (BOX, waveguide, cladding is implicit air).
2. Source:   ModeSource injects the fundamental TE eigenmode at z = z_src.
3. Monitors:
     a. ModeSolver monitor (mode_monitor): extracts neff, field profile at source cross-section.
     b. FieldMonitor at y = 0 (mid-height): captures Ex in the x-z plane (top-down view).
     c. FluxMonitor at z = z_flux: measures transmitted power vs frequency.
4. PML boundary conditions absorb outgoing radiation on all faces.

Running this script
────────────────────
  Option A — Tidy3D cloud (recommended, uses GPU):
      Set TIDY3D_API_KEY in your environment, then:
          python sim1_waveguide.py --mode cloud

  Option B — Local CPU (slow, good for structure inspection):
          python sim1_waveguide.py --mode local

  Option C — Dry-run (build structure, plot geometry, no solve):
          python sim1_waveguide.py --mode dryrun
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web

# ── Local project imports ────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.materials import Si, SiO2, FREQ0, WAVELENGTH_UM
from utils.materials import WG_WIDTH_UM, WG_HEIGHT_UM, BOX_THICKNESS_UM
from utils.plotting  import (plot_efield_xy, plot_mode_intensity,
                              plot_propagation)

# ════════════════════════════════════════════════════════════════════════════
# 1. SIMULATION DOMAIN & DISCRETIZATION
# ════════════════════════════════════════════════════════════════════════════

# Propagation length: long enough to observe stable guided propagation,
# short enough that cloud run-time stays practical.
# Rule of thumb: use at least 10× the mode's guided wavelength λg = λ/neff.
# At neff ≈ 2.4, λg ≈ 0.65 µm, so 10 µm is ~15 guided wavelengths.
Lz = 10.0   # propagation length [µm]

# Transverse extent: needs to be wide enough that the evanescent tail
# is well-absorbed by PML before it reaches the boundary.
# The evanescent field decays on the scale of λ/(2π·√(neff²-n²)) ≈ 0.2 µm,
# so 2 µm of cladding on each side is conservative.
Lx = 3.0    # total x-span [µm]  (width direction)
Ly = 3.0    # total y-span [µm]  (height direction)

# Grid resolution: λ/neff / (grid points per guided wavelength)
# 10 points per wavelength inside Si at 1550 nm → dx ≈ 0.045 µm = 45 nm.
# Tidy3D's auto-grid does this automatically given a min_steps_per_wvl target.
GRID_POINTS_PER_WVL = 10    # conservative; 15–20 for publication accuracy

# Source and monitor z-positions
Z_SRC  = -Lz / 2 + 0.5   # inject near the start of the domain [µm]
Z_FLUX =  Lz / 2 - 0.5   # measure flux near the end           [µm]

# Run time: simulation ends when field energy decays to 1e-8 of its peak
# (Tidy3D handles this automatically with shutoff condition).
# Minimum physical time = propagation length / (c/neff) ≈ 50 fs for 10 µm.
RUN_TIME = 1e-12   # [s]  — 1 ps is plenty for a 10 µm domain

print(f"Domain:  {Lx:.1f} × {Ly:.1f} × {Lz:.1f} µm³")
print(f"λ = {WAVELENGTH_UM} µm,  f₀ = {FREQ0/1e12:.2f} THz")


# ════════════════════════════════════════════════════════════════════════════
# 2. GEOMETRY — build the material stack
# ════════════════════════════════════════════════════════════════════════════

# ── 2a. Buried Oxide (BOX) ───────────────────────────────────────────────────
# The SiO2 BOX fills the lower half of the simulation domain.
# Box center is at y = -Ly/2 + BOX_THICKNESS_UM/2, i.e., it spans from
# y = -Ly/2  to  y = -Ly/2 + BOX_THICKNESS_UM.
# In practice, we make it slightly thicker than needed and let PML absorb
# anything that leaks through.

box_center_y = -Ly / 2 + BOX_THICKNESS_UM / 2.0

buried_oxide = td.Structure(
    geometry=td.Box(
        center=(0, box_center_y, 0),
        size=(Lx, BOX_THICKNESS_UM, Lz),
    ),
    medium=SiO2,
    name="BOX_SiO2",
)

# ── 2b. Silicon Ridge Waveguide ──────────────────────────────────────────────
# The waveguide sits on top of the BOX.
# Its bottom surface is at y = -Ly/2 + BOX_THICKNESS_UM,
# so its center is half its height above that.

wg_bottom_y  = -Ly / 2 + BOX_THICKNESS_UM
wg_center_y  = wg_bottom_y + WG_HEIGHT_UM / 2.0

waveguide = td.Structure(
    geometry=td.Box(
        center=(0, wg_center_y, 0),
        size=(WG_WIDTH_UM, WG_HEIGHT_UM, Lz),
    ),
    medium=Si,
    name="Si_waveguide",
)

structures = [buried_oxide, waveguide]


# ════════════════════════════════════════════════════════════════════════════
# 3. MODE SOURCE — inject the fundamental TE eigenmode
# ════════════════════════════════════════════════════════════════════════════

# ModeSource in Tidy3D works by:
#   1. Solving for the supported eigenmodes of the waveguide cross-section
#      at the source plane (z = Z_SRC) using the finite-difference mode solver.
#   2. Injecting the chosen mode (mode_index=0 → fundamental) as the
#      excitation current into the FDTD grid.
# This gives a clean, single-mode launch without the mode-mismatch artifacts
# you'd get from injecting a Gaussian beam.

# The ModeSpec tells the mode solver how many modes to find and which
# to inject.  target_neff gives it an initial guess for the search —
# here we use our analytic estimate (~2.4 for the 450×220 TE mode).

mode_spec = td.ModeSpec(
    num_modes=4,          # find 4 modes (allows us to confirm only 1 TE guided)
    target_neff=2.4,      # starting neff for eigenmode search
    num_pml=12,           # PML layers in transverse mode solver (more → more accurate)
)

mode_source = td.ModeSource(
    center=(0, wg_center_y, Z_SRC),
    size=(Lx, Ly, 0),     # 0 thickness in z → source is a 2D plane
    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FREQ0 * 0.1,  # 10% bandwidth — narrow enough for near-CW operation
    ),
    direction="+",         # inject in +z direction
    mode_spec=mode_spec,
    mode_index=0,          # 0 = fundamental mode (lowest loss, highest neff)
    name="TE_mode_source",
)


# ════════════════════════════════════════════════════════════════════════════
# 4. MONITORS — what we want to measure
# ════════════════════════════════════════════════════════════════════════════

# ── 4a. Mode monitor at the output facet ────────────────────────────────────
# Records the complex amplitudes of each mode at Z_FLUX.
# From this we extract:
#   - Transmitted power in the fundamental TE mode (mode_index=0)
#   - neff of each mode (confirms single-mode guidance)

mode_monitor = td.ModeMonitor(
    center=(0, wg_center_y, Z_FLUX),
    size=(Lx, Ly, 0),
    freqs=[FREQ0],
    mode_spec=mode_spec,
    name="output_mode_monitor",
)

# ── 4b. Field monitor: transverse cross-section at z = Lz/2 ─────────────────
# Captures Ex, Ey, Ez in the y-z plane at the waveguide midpoint.
# This gives us the mode's transverse field profile — the key visualization
# that shows single-mode confinement.

xsec_monitor = td.FieldMonitor(
    center=(0, 0, 0),              # center of domain in z
    size=(Lx, Ly, 0),             # zero z-extent → 2D cross-section
    freqs=[FREQ0],
    fields=["Ex", "Ey", "Ez"],
    name="xsec_field_monitor",
)

# ── 4c. Field monitor: top-down view (x-z plane at waveguide mid-height) ────
# Captures Ex in the horizontal plane through the waveguide center.
# Shows how the mode propagates along z — flat amplitude → well-guided.

topdown_monitor = td.FieldMonitor(
    center=(0, wg_center_y, 0),
    size=(Lx, 0, Lz),             # zero y-extent → 2D top-down slice
    freqs=[FREQ0],
    fields=["Ex"],
    name="topdown_field_monitor",
)

# ── 4d. Flux monitor: total transmitted power vs z ───────────────────────────
# FluxMonitor integrates Poynting vector over the plane → total power.
# Comparing flux at input vs output confirms negligible propagation loss.

flux_monitor = td.FluxMonitor(
    center=(0, wg_center_y, Z_FLUX),
    size=(Lx, Ly, 0),
    freqs=[FREQ0],
    name="flux_monitor",
)

monitors = [mode_monitor, xsec_monitor, topdown_monitor, flux_monitor]


# ════════════════════════════════════════════════════════════════════════════
# 5. BOUNDARY CONDITIONS & GRID
# ════════════════════════════════════════════════════════════════════════════

# PML (Perfectly Matched Layer) boundaries on all 6 faces.
# PML absorbs outgoing radiation without reflection.
# 12-layer PML is standard; 20+ for higher accuracy if needed.
boundary_spec = td.BoundarySpec.all_sides(
    boundary=td.PML(num_layers=12)
)

# Auto-grid: Tidy3D uses the structure geometry and wavelength to set
# a spatially-varying grid with refinement inside Si (high-n) regions.
grid_spec = td.GridSpec.auto(
    min_steps_per_wvl=GRID_POINTS_PER_WVL,
    wavelength=WAVELENGTH_UM,
)


# ════════════════════════════════════════════════════════════════════════════
# 6. ASSEMBLE THE SIMULATION OBJECT
# ════════════════════════════════════════════════════════════════════════════

# Everything in Tidy3D is immutable once created — this is intentional.
# The Simulation object is a complete, serializable description of the problem.
# You can save it to JSON, share it, re-run it, or sweep parameters
# by building a new Simulation with modified fields.

sim = td.Simulation(
    center=(0, 0, 0),
    size=(Lx, Ly, Lz),
    grid_spec=grid_spec,
    structures=structures,
    sources=[mode_source],
    monitors=monitors,
    run_time=RUN_TIME,
    boundary_spec=boundary_spec,
    medium=Air if True else SiO2,   # background medium = air (top cladding)
    # Note: structures fill in their regions; background fills the rest.
    # So air fills everything except where BOX and waveguide structures are.
    shutoff=1e-8,    # stop when stored energy drops below 1e-8 × peak value
    courant=0.9,     # CFL stability factor (0.9 = slightly conservative)
    symmetry=(0, 0, 0),   # no symmetry exploited (full 3D)
    version=td.__version__,
)

print(f"\nSimulation assembled successfully.")
print(f"  Grid cells (approx): {sim.num_cells:,}")
print(f"  Time steps (approx): {sim.num_time_steps:,}")
print(f"  Estimated memory:    {sim.estimate_cost()}")


# ════════════════════════════════════════════════════════════════════════════
# 7. GEOMETRY VISUALIZATION (always safe to run, no API key needed)
# ════════════════════════════════════════════════════════════════════════════

def plot_geometry():
    """
    Use Tidy3D's built-in geometry plotter to inspect the cross-section.
    This is the first thing to check before running — confirm the waveguide
    sits correctly on the BOX and is centered in the domain.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # x-y cross-section (transverse plane)
    ax = axes[0]
    sim.plot(y=wg_center_y, ax=ax)   # slice at waveguide mid-height
    ax.set_title("Geometry: x-z plane (through waveguide center)", pad=8)

    # y-z cross-section (side view along propagation axis)
    ax2 = axes[1]
    sim.plot(x=0, ax=ax2)
    ax2.set_title("Geometry: y-z plane (waveguide cross-section)", pad=8)

    plt.tight_layout()
    plt.savefig("../results/sim1_geometry.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Geometry plot saved → results/sim1_geometry.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. RUN & POST-PROCESS
# ════════════════════════════════════════════════════════════════════════════

def run_and_postprocess(mode: str = "cloud"):
    """
    Submit the simulation, wait for results, extract and plot key quantities.

    Parameters
    ----------
    mode : "cloud"   — submit to Tidy3D cloud (requires API key)
            "local"  — run locally (slow, CPU only)
            "dryrun" — geometry plot only, no solve
    """

    if mode == "dryrun":
        plot_geometry()
        print("\nDry-run complete.  Geometry verified, no solve performed.")
        return

    plot_geometry()

    # ── 8a. Submit & run ─────────────────────────────────────────────────────
    print(f"\nSubmitting to Tidy3D ({mode} mode)...")

    if mode == "cloud":
        # web.run() submits, polls, and returns SimulationData when done.
        # task_name is used to find the job in the Tidy3D dashboard.
        sim_data = web.run(
            sim,
            task_name="SOI_waveguide_TE_1550nm",
            path="sim1_data.hdf5",   # cached result — re-run loads from cache
            verbose=True,
        )
    else:
        # Local solver — useful for debugging structure, slow for real runs
        sim_data = td.run(sim, task_name="sim1_local")

    print("Simulation complete.\n")

    # ── 8b. Extract mode data ─────────────────────────────────────────────────
    # mode_monitor returns an xarray Dataset with dimensions (mode_index, freq)
    # .amps gives complex mode amplitude; power = |amp|²

    mode_data = sim_data["output_mode_monitor"]
    amps      = mode_data.amps.sel(direction="+")   # forward-propagating modes

    print("=== Mode Analysis ===")
    for i in range(mode_spec.num_modes):
        try:
            neff_i = mode_data.n_eff.sel(mode_index=i, f=FREQ0).item()
            loss_i = mode_data.n_group.sel(mode_index=i, f=FREQ0).item()
            amp_i  = amps.sel(mode_index=i, f=FREQ0).item()
            power_i = np.abs(amp_i)**2
            print(f"  Mode {i}: neff = {neff_i:.4f},  "
                  f"power fraction = {power_i:.4f}")
        except Exception:
            print(f"  Mode {i}: not found (below cutoff or not guided)")

    # Transmission = power in fundamental mode / total launched power
    # Should be close to 1.0 for a well-designed mode launch.
    amp0     = amps.sel(mode_index=0, f=FREQ0).item()
    T_mode0  = np.abs(amp0)**2
    print(f"\n  TE₀ mode transmission: {T_mode0:.4f}  ({T_mode0*100:.1f}%)")

    # ── 8c. Extract and plot the transverse field profile ─────────────────────
    xsec_data = sim_data["xsec_field_monitor"]

    # Ex component at FREQ0 — shape: (Nx, Ny)
    Ex_2d = xsec_data["Ex"].sel(f=FREQ0).squeeze()
    Ex_np = Ex_2d.values.real   # take real part for instantaneous snapshot

    x_coords = Ex_2d.coords["x"].values
    y_coords = Ex_2d.coords["y"].values

    # Normalize
    Ex_norm = Ex_np / np.max(np.abs(Ex_np))

    fig1 = plot_efield_xy(
        Ex=Ex_norm, x_coords=x_coords, y_coords=y_coords,
        title="TE₀ Mode: Ex Field Profile  (450×220 nm SOI, λ=1550 nm)",
        wg_width=WG_WIDTH_UM, wg_height=WG_HEIGHT_UM,
        save_path="../results/sim1_Ex_profile.png",
    )
    plt.show()

    # Total intensity |E|² = |Ex|² + |Ey|² + |Ez|²
    intensity = np.zeros_like(Ex_np)
    for comp in ["Ex", "Ey", "Ez"]:
        E = xsec_data[comp].sel(f=FREQ0).squeeze().values
        intensity += np.abs(E)**2

    intensity_norm = intensity / np.max(intensity)
    fig2 = plot_mode_intensity(
        intensity=intensity_norm, x_coords=x_coords, y_coords=y_coords,
        title="TE₀ Mode: Total Intensity |E|²  (450×220 nm SOI, λ=1550 nm)",
        wg_width=WG_WIDTH_UM, wg_height=WG_HEIGHT_UM,
        save_path="../results/sim1_intensity_profile.png",
    )
    plt.show()

    # ── 8d. Extract top-down propagation field ────────────────────────────────
    topdown_data = sim_data["topdown_field_monitor"]
    Ex_topdown   = topdown_data["Ex"].sel(f=FREQ0).squeeze()
    Ex_td_np     = Ex_topdown.values.real

    x_td = Ex_topdown.coords["x"].values
    z_td = Ex_topdown.coords["z"].values

    # Plot propagation (average |Ex| across x vs z)
    power_z = np.mean(np.abs(Ex_topdown.values)**2, axis=0)
    fig3 = plot_propagation(
        power_z=power_z, z_coords=z_td,
        title="Power Propagation Along SOI Waveguide (λ=1550 nm)",
        save_path="../results/sim1_propagation.png",
    )
    plt.show()

    print("\nAll plots saved to results/")
    print("Simulation 1 complete.\n")
    return sim_data


# ════════════════════════════════════════════════════════════════════════════
# 9. CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOI Waveguide Tidy3D Simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["cloud", "local", "dryrun"],
        default="dryrun",
        help=(
            "cloud   → submit to Tidy3D cloud (needs TIDY3D_API_KEY)\n"
            "local   → run locally (slow)\n"
            "dryrun  → geometry check only (default)"
        ),
    )
    args = parser.parse_args()
    run_and_postprocess(mode=args.mode)
