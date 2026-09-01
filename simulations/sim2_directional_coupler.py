"""
sim2_directional_coupler.py
═══════════════════════════════════════════════════════════════════════════════
Simulation 2: SOI Directional Coupler — Evanescent Coupling Gap Sweep
───────────────────────────────────────────────────────────────────────────────
Platform:  450 nm × 220 nm Si ridge on SiO2 BOX, air cladding.
Method:    3D FDTD via Tidy3D — one simulation per gap value.
Sweep:     Gap = 100, 125, 150, 175, 200 nm (5 points; adjust as needed).
Goal:      Extract coupling length Lc(gap) and fit the exponential dependence.

Physics overview — how a directional coupler works
────────────────────────────────────────────────────
When two waveguides are placed close together, their evanescent fields overlap.
This creates two *supermodes* of the coupled system:

  Even supermode  (symmetric):  both waveguides in phase      → neff,even
  Odd  supermode  (anti-symm):  waveguides out of phase by π  → neff,odd

Because neff,even ≠ neff,odd, the two supermodes accumulate a relative phase
difference as they propagate.  After a length:

    Lc = λ / (2 · |neff,even − neff,odd|)       [coupling length]

the phase difference reaches π, and ALL power has transferred from waveguide A
to waveguide B.  After 2·Lc, it transfers back — a perfect 3 dB splitter is
achieved at Lc/2.

As the gap increases:
  - The evanescent overlap decreases exponentially (∝ exp(−κ·gap))
  - Δneff = |neff,even − neff,odd| decreases exponentially
  - Therefore Lc increases exponentially with gap

This exponential relationship is what the sweep extracts and fits.

Reference: Chrostowski & Hochberg "Silicon Photonics Design" (2015), Ch. 5.

Simulation strategy
────────────────────
For each gap value, we run a simulation long enough to observe at least one
complete power transfer cycle (propagation length > Lc).  We then:
  a. Monitor power in waveguide A and B along z.
  b. Find the z at which power in B is maximized → that z = Lc.
  Alternatively (more robust): use the supermode neff difference approach —
  run ModeSolvers for even/odd supermodes and compute Lc analytically.

We implement BOTH approaches:
  Method 1: FDTD field monitoring (visual, pedagogical)
  Method 2: Supermode neff extraction (faster, more accurate for portfolio)
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins.mode import ModeSolver  # built-in eigenmode solver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.materials import Si, SiO2, FREQ0, WAVELENGTH_UM
from utils.materials import WG_WIDTH_UM, WG_HEIGHT_UM, BOX_THICKNESS_UM
from utils.plotting  import (plot_coupling_length_vs_gap,
                              plot_coupler_field_top)

# Import Air (background medium) — recreate here since materials.py uses td.Medium
Air = td.Medium(permittivity=1.0)

# ════════════════════════════════════════════════════════════════════════════
# 1. SWEEP PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

# Gap values to sweep [nm] — converted to µm for Tidy3D
GAP_VALUES_NM   = np.array([100, 125, 150, 175, 200])
GAP_VALUES_UM   = GAP_VALUES_NM * 1e-3   # nm → µm

# Center-to-center distance between the two waveguides for each gap.
# c2c = WG_WIDTH + gap
CENTER_TO_CENTER_UM = WG_WIDTH_UM + GAP_VALUES_UM   # shape: (N_gaps,)

# Propagation length for the FDTD simulations.
# Must be > expected Lc for the smallest gap.  For gap=100 nm on this platform,
# Lc ≈ 5–20 µm depending on neff contrast.  Use 30 µm to be safe.
Lz_COUPLER = 30.0    # [µm]

# Transverse domain — must span both waveguides plus evanescent decay regions
# Total width needed: gap + 2 × WG_WIDTH + 2 × cladding pad
CLAD_PAD = 2.0       # [µm] cladding on each outer edge
Lx_COUPLER = 2 * WG_WIDTH_UM + max(GAP_VALUES_UM) + 2 * CLAD_PAD
Ly_COUPLER = 3.0     # [µm]

print("=== Directional Coupler Gap Sweep ===")
print(f"Gaps:      {GAP_VALUES_NM} nm")
print(f"Domain:    {Lx_COUPLER:.2f} × {Ly_COUPLER:.1f} × {Lz_COUPLER:.1f} µm³")
print(f"c2c span:  {CENTER_TO_CENTER_UM*1e3} nm\n")


# ════════════════════════════════════════════════════════════════════════════
# 2. GEOMETRY BUILDER — parameterized by gap
# ════════════════════════════════════════════════════════════════════════════

def build_coupler_structures(gap_um: float) -> list:
    """
    Build the list of Tidy3D Structure objects for a directional coupler
    with a given evanescent coupling gap.

    Layout (top-down x-view):
        |← WG_WIDTH →|← gap →|← WG_WIDTH →|
    WG A centered at x = -(gap/2 + WG_WIDTH/2)
    WG B centered at x = +(gap/2 + WG_WIDTH/2)

    Parameters
    ----------
    gap_um : coupling gap between waveguide sidewalls [µm]

    Returns
    -------
    list of td.Structure
    """
    # y-position of waveguide center (same as Sim 1)
    wg_bottom_y = -Ly_COUPLER / 2 + BOX_THICKNESS_UM
    wg_center_y = wg_bottom_y + WG_HEIGHT_UM / 2.0

    # x-centers of waveguide A and B
    x_wgA = -(gap_um / 2.0 + WG_WIDTH_UM / 2.0)   # left waveguide
    x_wgB = +(gap_um / 2.0 + WG_WIDTH_UM / 2.0)   # right waveguide

    # Buried oxide
    buried_oxide = td.Structure(
        geometry=td.Box(
            center=(0, -Ly_COUPLER / 2 + BOX_THICKNESS_UM / 2, 0),
            size=(Lx_COUPLER, BOX_THICKNESS_UM, Lz_COUPLER),
        ),
        medium=SiO2,
        name="BOX_SiO2",
    )

    # Waveguide A (input waveguide — we inject here)
    wg_A = td.Structure(
        geometry=td.Box(
            center=(x_wgA, wg_center_y, 0),
            size=(WG_WIDTH_UM, WG_HEIGHT_UM, Lz_COUPLER),
        ),
        medium=Si,
        name="WG_A",
    )

    # Waveguide B (coupled waveguide — power transfers here)
    wg_B = td.Structure(
        geometry=td.Box(
            center=(x_wgB, wg_center_y, 0),
            size=(WG_WIDTH_UM, WG_HEIGHT_UM, Lz_COUPLER),
        ),
        medium=Si,
        name="WG_B",
    )

    return [buried_oxide, wg_A, wg_B], wg_center_y, x_wgA, x_wgB


# ════════════════════════════════════════════════════════════════════════════
# 3. METHOD 2 (PREFERRED): SUPERMODE ANALYSIS VIA ModeSolver
# ════════════════════════════════════════════════════════════════════════════

def compute_Lc_supermode(gap_um: float) -> tuple[float, float, float]:
    """
    Compute coupling length analytically from the even/odd supermode neff split.

    This is the most accurate approach because:
      - It avoids fitting a noisy FDTD power profile
      - It's fast (eigenmode solve, not a full 3D FDTD)
      - It directly gives Δneff = neff,even − neff,odd

    The coupling length formula:
        Lc = λ₀ / (2 × |neff,even − neff,odd|)

    Parameters
    ----------
    gap_um : coupling gap [µm]

    Returns
    -------
    Lc_um         : coupling length [µm]
    neff_even     : effective index of even supermode
    neff_odd      : effective index of odd supermode
    """
    structures, wg_center_y, x_wgA, x_wgB = build_coupler_structures(gap_um)

    # Build a minimal simulation just for the mode solve.
    # The simulation doesn't need to run — the ModeSolver uses only the
    # geometry and medium to compute eigenmodes at the cross-section plane.

    # Dummy source (required by Simulation but won't affect mode solve)
    dummy_source = td.ModeSource(
        center=(x_wgA, wg_center_y, -Lz_COUPLER / 2 + 0.5),
        size=(Lx_COUPLER, Ly_COUPLER, 0),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FREQ0 * 0.1),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=2, target_neff=2.4),
        mode_index=0,
        name="dummy_source",
    )

    sim_for_mode = td.Simulation(
        center=(0, 0, 0),
        size=(Lx_COUPLER, Ly_COUPLER, Lz_COUPLER),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=WAVELENGTH_UM),
        structures=structures,
        sources=[dummy_source],
        monitors=[],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML(num_layers=12)),
        medium=Air,
        version=td.__version__,
    )

    # ModeSolver: solves for eigenmodes at the cross-section z = 0.
    # We ask for 2 supermodes:
    #   mode_index=0 → even supermode  (symmetric, neff,even > neff,odd)
    #   mode_index=1 → odd  supermode  (anti-symmetric)

    mode_spec_super = td.ModeSpec(
        num_modes=4,           # find 4; we'll identify even/odd by symmetry
        target_neff=2.4,
        num_pml=(12, 12),
        precision="double",    # higher precision for small Δneff
    )

    mode_plane = td.ModeMonitor(
        center=(0, wg_center_y, 0),
        size=(Lx_COUPLER, Ly_COUPLER, 0),
        freqs=[FREQ0],
        mode_spec=mode_spec_super,
        name="supermode_monitor",
    )

    ms = ModeSolver(
        simulation=sim_for_mode,
        plane=mode_plane,
        mode_spec=mode_spec_super,
        freqs=[FREQ0],
    )

    mode_data = ms.solve()

    # Extract neff for each supermode
    neff_all = mode_data.n_eff.sel(f=FREQ0).values.real   # shape: (num_modes,)

    # Even supermode has higher neff (more confined, symmetric field)
    # Odd supermode has lower neff (anti-symmetric, broader)
    # Sort descending to get even first
    neff_sorted = np.sort(neff_all)[::-1]

    # The two supermodes of interest:
    neff_even = neff_sorted[0]   # highest neff → even (symmetric)
    neff_odd  = neff_sorted[1]   # next highest  → odd  (anti-symmetric)

    # Coupling length: half the beat length
    delta_neff = abs(neff_even - neff_odd)
    Lc_um = WAVELENGTH_UM / (2.0 * delta_neff)

    print(f"  gap={gap_um*1e3:.0f} nm: "
          f"neff_even={neff_even:.5f}, neff_odd={neff_odd:.5f}, "
          f"Δneff={delta_neff:.5f}, Lc={Lc_um:.2f} µm")

    return Lc_um, neff_even, neff_odd


# ════════════════════════════════════════════════════════════════════════════
# 4. METHOD 1: FDTD POWER-TRANSFER MEASUREMENT (pedagogical)
# ════════════════════════════════════════════════════════════════════════════

def build_coupler_simulation(gap_um: float) -> tuple:
    """
    Build a full 3D FDTD simulation of the directional coupler for a given gap.

    The simulation:
      - Injects the fundamental TE mode into waveguide A at z_src
      - Monitors power in waveguide A and B as a function of z
        using a series of FluxMonitor cross-sections
      - Captures top-down field for visualization

    Parameters
    ----------
    gap_um : coupling gap [µm]

    Returns
    -------
    sim : td.Simulation object ready to submit
    meta: dict with geometric metadata (positions, etc.)
    """
    structures, wg_center_y, x_wgA, x_wgB = build_coupler_structures(gap_um)

    Z_SRC  = -Lz_COUPLER / 2 + 1.0
    Z_END  =  Lz_COUPLER / 2 - 1.0

    # ── Source: inject TE₀ into WG A ─────────────────────────────────────────
    mode_spec_src = td.ModeSpec(num_modes=2, target_neff=2.4, num_pml=12)

    source = td.ModeSource(
        center=(x_wgA, wg_center_y, Z_SRC),
        size=(WG_WIDTH_UM * 3, Ly_COUPLER, 0),  # span wide enough to capture WG A
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FREQ0 * 0.1),
        direction="+",
        mode_spec=mode_spec_src,
        mode_index=0,
        name="WGA_source",
    )

    # ── Monitors ──────────────────────────────────────────────────────────────

    # Power in WG A at output
    flux_A_out = td.FluxMonitor(
        center=(x_wgA, wg_center_y, Z_END),
        size=(WG_WIDTH_UM * 2, Ly_COUPLER, 0),
        freqs=[FREQ0],
        name="flux_WGA_out",
    )

    # Power in WG B at output
    flux_B_out = td.FluxMonitor(
        center=(x_wgB, wg_center_y, Z_END),
        size=(WG_WIDTH_UM * 2, Ly_COUPLER, 0),
        freqs=[FREQ0],
        name="flux_WGB_out",
    )

    # Array of monitors along z — coarser spacing, records power at each z slice
    # This lets us trace P_A(z) and P_B(z) to find where max transfer occurs.
    n_z_monitors = 20
    z_positions  = np.linspace(Z_SRC + 1.0, Z_END, n_z_monitors)

    z_monitors_A = [
        td.FluxMonitor(
            center=(x_wgA, wg_center_y, z),
            size=(WG_WIDTH_UM * 2, Ly_COUPLER, 0),
            freqs=[FREQ0],
            name=f"PA_z{i:02d}",
        )
        for i, z in enumerate(z_positions)
    ]

    z_monitors_B = [
        td.FluxMonitor(
            center=(x_wgB, wg_center_y, z),
            size=(WG_WIDTH_UM * 2, Ly_COUPLER, 0),
            freqs=[FREQ0],
            name=f"PB_z{i:02d}",
        )
        for i, z in enumerate(z_positions)
    ]

    # Top-down field monitor for visualization
    topdown = td.FieldMonitor(
        center=(0, wg_center_y, 0),
        size=(Lx_COUPLER, 0, Lz_COUPLER),
        freqs=[FREQ0],
        fields=["Ex"],
        name="topdown",
    )

    all_monitors = (
        [flux_A_out, flux_B_out, topdown]
        + z_monitors_A
        + z_monitors_B
    )

    # ── Assemble simulation ───────────────────────────────────────────────────
    sim = td.Simulation(
        center=(0, 0, 0),
        size=(Lx_COUPLER, Ly_COUPLER, Lz_COUPLER),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=WAVELENGTH_UM),
        structures=structures,
        sources=[source],
        monitors=all_monitors,
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML(num_layers=12)),
        medium=Air,
        shutoff=1e-8,
        version=td.__version__,
    )

    meta = {
        "wg_center_y": wg_center_y,
        "x_wgA": x_wgA,
        "x_wgB": x_wgB,
        "z_positions": z_positions,
        "n_z_monitors": n_z_monitors,
    }

    return sim, meta


def extract_Lc_from_fdtd(sim_data, meta: dict, gap_um: float) -> float:
    """
    Extract coupling length from FDTD simulation data by finding the z
    at which power in waveguide B is maximized.

    P_B(z) follows: P_B = sin²(π·z / (2·Lc))
    So Lc = z_max / 1, where z_max is the first maximum.

    Parameters
    ----------
    sim_data : Tidy3D SimulationData object
    meta     : geometry metadata dict from build_coupler_simulation()
    gap_um   : coupling gap [µm]  (for labeling)

    Returns
    -------
    Lc_um : coupling length [µm]
    """
    z_positions   = meta["z_positions"]
    n_z           = meta["n_z_monitors"]

    PA_arr = np.zeros(n_z)
    PB_arr = np.zeros(n_z)

    for i in range(n_z):
        PA_arr[i] = float(
            sim_data[f"PA_z{i:02d}"].flux.sel(f=FREQ0).item()
        )
        PB_arr[i] = float(
            sim_data[f"PB_z{i:02d}"].flux.sel(f=FREQ0).item()
        )

    # Normalize by initial power in A (avoid negative flux from PML leakage)
    P_total = PA_arr[0] + PB_arr[0]
    PA_norm = PA_arr / P_total
    PB_norm = PB_arr / P_total

    # Find z where P_B is first maximized
    peak_idx = np.argmax(PB_norm)
    Lc_um    = z_positions[peak_idx]

    print(f"  FDTD: gap={gap_um*1e3:.0f} nm → "
          f"max P_B = {PB_norm[peak_idx]:.3f} at z = {Lc_um:.2f} µm")

    # Plot P_A and P_B vs z for this gap
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(z_positions, PA_norm, "b-o", label="P_A (input WG)", ms=5, lw=1.8)
    ax.plot(z_positions, PB_norm, "r-o", label="P_B (coupled WG)", ms=5, lw=1.8)
    ax.axvline(Lc_um, color="k", ls="--", lw=1.2, label=f"Lc ≈ {Lc_um:.1f} µm")
    ax.set_xlabel("z  [µm]")
    ax.set_ylabel("Normalized power")
    ax.set_title(f"Power Transfer: gap = {gap_um*1e3:.0f} nm  |  Lc ≈ {Lc_um:.1f} µm")
    ax.legend(fontsize=9)
    ax.grid(True, color="#e0e0e0")
    plt.tight_layout()
    plt.savefig(f"../results/sim2_transfer_gap{gap_um*1e3:.0f}nm.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    return Lc_um


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN SWEEP ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════

def run_sweep(mode: str = "supermode", run_mode: str = "cloud"):
    """
    Run the coupling length sweep over all gap values.

    Parameters
    ----------
    mode     : "supermode" — fast, analytic from neff difference (recommended)
               "fdtd"      — full 3D FDTD power-transfer measurement
    run_mode : "cloud" | "local" | "dryrun"

    Returns
    -------
    gaps_nm : array of gap values [nm]
    Lc_arr  : array of coupling lengths [µm]
    """
    Lc_results    = np.zeros(len(GAP_VALUES_NM))
    neff_even_arr = np.zeros(len(GAP_VALUES_NM))
    neff_odd_arr  = np.zeros(len(GAP_VALUES_NM))

    print(f"\nRunning sweep ({mode} method, {run_mode} mode)...\n")

    if mode == "supermode":
        # ── Supermode method: fast, one ModeSolver per gap ────────────────────
        for i, (gap_nm, gap_um) in enumerate(zip(GAP_VALUES_NM, GAP_VALUES_UM)):
            print(f"[{i+1}/{len(GAP_VALUES_NM)}] gap = {gap_nm} nm")

            if run_mode == "dryrun":
                # Synthetic data for dryrun (exponential model with real params)
                Lc_results[i]    = 3.5 * np.exp((gap_nm - 100) / 60.0)
                neff_even_arr[i] = 2.42 - 0.005 * i
                neff_odd_arr[i]  = 2.38 + 0.003 * i
                print(f"  [dryrun] Lc ≈ {Lc_results[i]:.2f} µm  (synthetic)")
            else:
                Lc, ne, no = compute_Lc_supermode(gap_um)
                Lc_results[i]    = Lc
                neff_even_arr[i] = ne
                neff_odd_arr[i]  = no

    elif mode == "fdtd":
        # ── FDTD method: full simulation per gap (slower, more pedagogical) ───
        for i, (gap_nm, gap_um) in enumerate(zip(GAP_VALUES_NM, GAP_VALUES_UM)):
            print(f"\n[{i+1}/{len(GAP_VALUES_NM)}] FDTD: gap = {gap_nm} nm")

            sim, meta = build_coupler_simulation(gap_um)

            if run_mode == "dryrun":
                Lc_results[i] = 3.5 * np.exp((gap_nm - 100) / 60.0)
                print(f"  [dryrun] Lc ≈ {Lc_results[i]:.2f} µm  (synthetic)")
                continue

            if run_mode == "cloud":
                sim_data = web.run(
                    sim,
                    task_name=f"SOI_coupler_gap{gap_nm}nm",
                    path=f"sim2_gap{gap_nm}nm.hdf5",
                    verbose=False,
                )
            else:
                sim_data = td.run(sim, task_name=f"sim2_gap{gap_nm}nm")

            Lc_results[i] = extract_Lc_from_fdtd(sim_data, meta, gap_um)

            # Visualize top-down field for first gap only (most compact coupling)
            if i == 0:
                td_data = sim_data["topdown"]
                Ex_td   = td_data["Ex"].sel(f=FREQ0).squeeze()
                Ex_np   = Ex_td.values.real
                x_td    = Ex_td.coords["x"].values
                z_td    = Ex_td.coords["z"].values
                plot_coupler_field_top(
                    field_xz=Ex_np, x_coords=x_td, z_coords=z_td,
                    gap_nm=gap_nm, Lc_um=Lc_results[i],
                    save_path=f"../results/sim2_topdown_gap{gap_nm}nm.png",
                )
                plt.show()

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'supermode' or 'fdtd'.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("COUPLING LENGTH SWEEP RESULTS")
    print("="*55)
    print(f"{'Gap [nm]':>10} {'Lc [µm]':>12} {'neff_even':>12} {'neff_odd':>12}")
    print("-"*55)
    for i, gap_nm in enumerate(GAP_VALUES_NM):
        print(f"{gap_nm:>10} {Lc_results[i]:>12.3f} "
              f"{neff_even_arr[i]:>12.5f} {neff_odd_arr[i]:>12.5f}")
    print("="*55)

    # ── Exponential fit ───────────────────────────────────────────────────────
    log_Lc = np.log(Lc_results)
    coeffs  = np.polyfit(GAP_VALUES_NM, log_Lc, deg=1)
    A_fit   = np.exp(coeffs[1])
    B_fit   = coeffs[0]     # [nm⁻¹]
    print(f"\nExponential fit: Lc(gap) = {A_fit:.2f} · exp({B_fit*1000:.4f} · gap/nm)  µm")
    print(f"Evanescent decay length: {1/B_fit:.1f} nm")

    # ── Final plots ───────────────────────────────────────────────────────────
    fig_sweep = plot_coupling_length_vs_gap(
        gaps_nm=GAP_VALUES_NM,
        Lc_um=Lc_results,
        save_path="../results/sim2_Lc_vs_gap.png",
        annotate=True,
    )
    plt.show()

    # Δneff vs gap (supermode method only)
    if mode == "supermode" and np.any(neff_even_arr > 0):
        delta_neff = neff_even_arr - neff_odd_arr
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.semilogy(GAP_VALUES_NM, delta_neff, "o-",
                     color="#1a6faf", lw=2, ms=8,
                     markerfacecolor="white", markeredgewidth=2)
        ax2.set_xlabel("Coupling gap  [nm]")
        ax2.set_ylabel("Δneff = neff,even − neff,odd  (log scale)")
        ax2.set_title("Supermode Index Splitting vs Gap\n(450×220 nm SOI, λ = 1550 nm)")
        ax2.grid(True, which="both", color="#e0e0e0")
        plt.tight_layout()
        plt.savefig("../results/sim2_delta_neff_vs_gap.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    np.save("../results/sim2_gaps_nm.npy",  GAP_VALUES_NM)
    np.save("../results/sim2_Lc_um.npy",    Lc_results)
    np.save("../results/sim2_neff_even.npy", neff_even_arr)
    np.save("../results/sim2_neff_odd.npy",  neff_odd_arr)
    print("\nResults saved to results/ directory.")

    return GAP_VALUES_NM, Lc_results


# ════════════════════════════════════════════════════════════════════════════
# 6. CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Directional Coupler Gap Sweep — Tidy3D"
    )
    parser.add_argument(
        "--method",
        choices=["supermode", "fdtd"],
        default="supermode",
        help=(
            "supermode → fast Δneff method (recommended for portfolio)\n"
            "fdtd      → full 3D FDTD power transfer (pedagogical)"
        ),
    )
    parser.add_argument(
        "--run",
        choices=["cloud", "local", "dryrun"],
        default="dryrun",
        dest="run_mode",
        help=(
            "cloud   → Tidy3D cloud (requires TIDY3D_API_KEY)\n"
            "local   → CPU (slow)\n"
            "dryrun  → synthetic data, no solve (default)"
        ),
    )
    args = parser.parse_args()
    run_sweep(mode=args.method, run_mode=args.run_mode)
