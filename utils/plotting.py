"""
plotting.py — Reusable visualization helpers for the silicon photonics project.

Design philosophy:
  - All plots use a consistent dark-on-light scientific style (matplotlib's
    default colormaps are overridden for physical clarity).
  - Field plots use RdBu_r (red = +E, blue = −E) — standard for EM fields.
  - Power / intensity plots use 'inferno' (perceptually uniform, colorblind-safe).
  - Coupling-length sweep uses a clean blue line with confidence markers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Optional

# ─── Global style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# ─── Color palette ───────────────────────────────────────────────────────────

PALETTE = {
    "si_fill":    "#c8d8e8",   # light blue-grey  — silicon waveguide region
    "box_fill":   "#f0ece0",   # pale cream        — SiO2 BOX
    "clad_fill":  "#ffffff",   # white             — air cladding
    "accent":     "#1a6faf",   # steel blue        — primary data lines
    "accent2":    "#d64e12",   # burnt orange      — secondary / comparison
    "grid":       "#e0e0e0",
}


# ─── Waveguide geometry overlay ──────────────────────────────────────────────

def add_waveguide_outline(
    ax: plt.Axes,
    wg_width: float,
    wg_height: float,
    center_y: float = 0.0,
    label: str = "Si WG",
    alpha: float = 0.25,
) -> None:
    """
    Overlay a shaded rectangle representing the waveguide cross-section
    on an existing Axes (e.g., a field-profile plot in the y-z plane).

    Parameters
    ----------
    ax         : matplotlib Axes to draw on
    wg_width   : waveguide width  [µm]
    wg_height  : waveguide height [µm]
    center_y   : y-center of waveguide in plot coordinates [µm]
    label      : legend label
    alpha      : fill transparency
    """
    rect = Rectangle(
        xy=(-wg_width / 2, center_y - wg_height / 2),
        width=wg_width,
        height=wg_height,
        linewidth=1.2,
        edgecolor="#2a4a6a",
        facecolor=PALETTE["si_fill"],
        alpha=alpha,
        label=label,
        zorder=3,
    )
    ax.add_patch(rect)


# ─── Field profile plots ──────────────────────────────────────────────────────

def plot_efield_xy(
    Ex: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    title: str = "Electric Field |Ex| — TE Mode",
    wg_width: Optional[float] = None,
    wg_height: Optional[float] = None,
    center_y: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the x-component of the electric field (dominant TE component)
    in the transverse plane (x-y cross-section through the waveguide).

    Why Ex?  For the fundamental TE mode in a horizontally-oriented
    waveguide, Ex is the dominant polarization component — it points
    across the waveguide width.  Plotting |Ex| shows where the mode
    energy is concentrated.

    Parameters
    ----------
    Ex        : 2D array of Ex values [shape: (Nx, Ny)]
    x_coords  : 1D array of x positions [µm]
    y_coords  : 1D array of y positions [µm]
    title     : plot title string
    wg_width  : if given, overlays waveguide outline
    wg_height : if given (with wg_width), overlays waveguide outline
    save_path : if given, saves the figure to this path
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Normalize for display (symmetric colormap centered at zero)
    vmax = np.max(np.abs(Ex))
    im = ax.pcolormesh(
        x_coords, y_coords, Ex.T,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ex  (normalized)", fontsize=10)

    # Optional geometry overlay
    if wg_width is not None and wg_height is not None:
        add_waveguide_outline(ax, wg_width, wg_height, center_y=center_y)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

    ax.set_xlabel("x  [µm]")
    ax.set_ylabel("y  [µm]")
    ax.set_title(title, pad=8)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_mode_intensity(
    intensity: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    title: str = "Mode Intensity  |E|²",
    wg_width: Optional[float] = None,
    wg_height: Optional[float] = None,
    center_y: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot total mode intensity |E|² = |Ex|² + |Ey|² + |Ez|².
    Uses 'inferno' colormap — perceptually uniform and physically intuitive
    (dark = no field, bright = high intensity).
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    im = ax.pcolormesh(
        x_coords, y_coords, intensity.T,
        cmap="inferno",
        vmin=0,
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|E|²  (normalized)", fontsize=10)

    if wg_width is not None and wg_height is not None:
        # White outline visible against dark 'inferno' background
        rect = Rectangle(
            xy=(-wg_width / 2, center_y - wg_height / 2),
            width=wg_width, height=wg_height,
            linewidth=1.5, edgecolor="white",
            facecolor="none", zorder=3, label="Si WG",
        )
        ax.add_patch(rect)
        ax.legend(loc="upper right", fontsize=9,
                  framealpha=0.5, labelcolor="white")

    ax.set_xlabel("x  [µm]")
    ax.set_ylabel("y  [µm]")
    ax.set_title(title, pad=8)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path)
    return fig


# ─── Propagation monitor (field vs z) ────────────────────────────────────────

def plot_propagation(
    power_z: np.ndarray,
    z_coords: np.ndarray,
    title: str = "Power Propagation Along Waveguide",
    label: str = "Transmitted power",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how guided power evolves along the z-axis (propagation direction).
    Ideally flat for a lossless waveguide — any slope indicates radiation loss
    or mode mismatch at the source injection plane.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(z_coords, power_z / np.max(power_z),
            color=PALETTE["accent"], linewidth=2.0, label=label)
    ax.axhline(1.0, color=PALETTE["grid"], linewidth=1.0, linestyle="--")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("z  [µm]  (propagation axis)")
    ax.set_ylabel("Normalized power")
    ax.set_title(title, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)

    if save_path:
        fig.savefig(save_path)
    return fig


# ─── Coupling length vs gap (directional coupler sweep) ──────────────────────

def plot_coupling_length_vs_gap(
    gaps_nm: np.ndarray,
    Lc_um: np.ndarray,
    save_path: Optional[str] = None,
    annotate: bool = True,
) -> plt.Figure:
    """
    Plot coupling length Lc [µm] as a function of evanescent coupling gap [nm].

    Physical interpretation:
      Lc is the length at which power fully transfers from waveguide A to B.
      It grows exponentially with gap because the evanescent field decays
      exponentially into the gap region.  Fitting log(Lc) vs gap gives the
      evanescent decay constant κ, which characterizes the coupling strength.

    Parameters
    ----------
    gaps_nm  : 1D array of gap distances [nm]
    Lc_um    : 1D array of coupling lengths [µm]
    save_path: if given, saves the figure to this path
    annotate : if True, label the exponential trend
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left panel: linear scale ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(gaps_nm, Lc_um, "o-",
            color=PALETTE["accent"], linewidth=2.0,
            markersize=7, markerfacecolor="white",
            markeredgewidth=2.0, label="Simulated Lc")

    # Exponential fit: Lc = A * exp(B * gap)
    try:
        log_Lc = np.log(Lc_um)
        coeffs  = np.polyfit(gaps_nm, log_Lc, deg=1)  # [B, log(A)]
        gap_fit = np.linspace(gaps_nm[0], gaps_nm[-1], 200)
        Lc_fit  = np.exp(np.polyval(coeffs, gap_fit))
        ax.plot(gap_fit, Lc_fit, "--",
                color=PALETTE["accent2"], linewidth=1.5,
                label=f"Exp. fit  (κ⁻¹ ≈ {1/coeffs[0]*1e-3:.1f} µm/nm)")
        if annotate:
            mid_idx = len(gap_fit) // 2
            ax.annotate(
                f"Lc ∝ e^(gap/λ_ev)",
                xy=(gap_fit[mid_idx], Lc_fit[mid_idx]),
                xytext=(gap_fit[mid_idx] - 15, Lc_fit[mid_idx] * 1.25),
                fontsize=9, color=PALETTE["accent2"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent2"]),
            )
    except Exception:
        pass  # fit is cosmetic; don't crash if data is noisy

    ax.set_xlabel("Coupling gap  [nm]")
    ax.set_ylabel("Coupling length Lc  [µm]")
    ax.set_title("Coupling Length vs Gap  (linear)", pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)

    # ── Right panel: semi-log scale (shows exponential dependence clearly) ──
    ax2 = axes[1]
    ax2.semilogy(gaps_nm, Lc_um, "o-",
                 color=PALETTE["accent"], linewidth=2.0,
                 markersize=7, markerfacecolor="white",
                 markeredgewidth=2.0, label="Simulated Lc")
    try:
        ax2.semilogy(gap_fit, Lc_fit, "--",
                     color=PALETTE["accent2"], linewidth=1.5,
                     label="Exp. fit")
    except Exception:
        pass

    ax2.set_xlabel("Coupling gap  [nm]")
    ax2.set_ylabel("Coupling length Lc  [µm]  (log scale)")
    ax2.set_title("Coupling Length vs Gap  (semi-log)", pad=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", color=PALETTE["grid"], linewidth=0.7)

    fig.suptitle(
        "Directional Coupler — Evanescent Coupling Sweep  (450 nm × 220 nm SOI, λ = 1550 nm)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ─── Coupler field snapshot (Ez along propagation) ───────────────────────────

def plot_coupler_field_top(
    field_xz: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    gap_nm: float,
    Lc_um: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Top-down (x-z) view of the field in a directional coupler.
    Shows the sinusoidal power exchange between the two waveguides.

    The beating pattern is the physical signature of the coupling:
    two supermodes (even + odd) with different effective indices propagate
    at slightly different phase velocities, and their interference produces
    the periodic power transfer with beat length = Lc.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    vmax = np.max(np.abs(field_xz))
    ax.pcolormesh(
        z_coords, x_coords, field_xz,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        shading="auto",
    )

    # Annotate coupling length
    if Lc_um > 0:
        ax.axvline(Lc_um, color="#ffd700", linewidth=1.5, linestyle="--",
                   label=f"Lc = {Lc_um:.1f} µm")
        ax.legend(loc="upper right", fontsize=9,
                  framealpha=0.7, labelcolor="white",
                  facecolor="#333333")

    ax.set_xlabel("z  [µm]  (propagation axis)")
    ax.set_ylabel("x  [µm]  (transverse axis)")
    ax.set_title(
        f"Coupler Top-Down Field  |  Gap = {gap_nm:.0f} nm  |  λ = 1550 nm",
        pad=8,
    )

    if save_path:
        fig.savefig(save_path)
    return fig
