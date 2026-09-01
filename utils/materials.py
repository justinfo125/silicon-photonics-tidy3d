"""
materials.py — Optical material definitions for SOI silicon photonics at 1550 nm.

All refractive indices are taken from standard literature values for
the O-band/C-band SOI platform (Si / SiO2 / air cladding):

  - Si  (crystalline): n ≈ 3.476  (Palik, 1985; verified against
                                    Aspnes & Studna 1983)
  - SiO2 (thermal oxide): n ≈ 1.444 (Malitson 1965)
  - Air cladding:       n = 1.000

Why these values matter physically:
  The large index contrast between Si (n=3.48) and SiO2 (n=1.44) — Δn ≈ 2.0 —
  is what makes tight optical confinement possible in SOI waveguides.
  A 450 nm × 220 nm Si ridge on SiO2 supports only the fundamental TE mode
  at 1550 nm precisely because this high contrast cuts off higher-order modes.
  This is the same platform used in Ayar Labs' TeraPHY chiplets.
"""

import tidy3d as td

# ─── Wavelength / Frequency ─────────────────────────────────────────────────

WAVELENGTH_UM = 1.55          # Free-space wavelength [µm]  (C-band center)
FREQ0 = td.C_0 / WAVELENGTH_UM  # Center frequency [Hz]

# ─── Waveguide Cross-Section (standard 220-nm SOI platform) ─────────────────

WG_WIDTH_UM  = 0.45   # 450 nm  — single-mode TE cutoff width
WG_HEIGHT_UM = 0.22   # 220 nm  — standard SOI device layer thickness
BOX_THICKNESS_UM = 2.0  # Buried oxide (BOX) thickness — thick enough that
                         # evanescent tail doesn't reach Si substrate
CLAD_THICKNESS_UM = 1.0  # Air or oxide over-cladding above waveguide top

# ─── Material Definitions ────────────────────────────────────────────────────

# Silicon (crystalline) — dispersionless at 1550 nm for FDTD simplicity.
# For broadband sweeps you would use a Sellmeier pole fit, but at a single
# frequency a real-valued permittivity is exact.
Si = td.Medium(permittivity=3.476**2)   # ε = n²

# Silicon dioxide (thermal oxide BOX + cladding if oxide-clad)
SiO2 = td.Medium(permittivity=1.444**2)

# Air cladding (used when the waveguide top surface is exposed)
Air = td.Medium(permittivity=1.0)

# ─── Convenience mapping ─────────────────────────────────────────────────────

MATERIALS = {
    "Si":   Si,
    "SiO2": SiO2,
    "Air":  Air,
}


def neff_estimate(n_core: float = 3.476, n_clad: float = 1.444) -> float:
    """
    Very rough effective index estimate using a linear interpolation
    between core and cladding indices, weighted by confinement factor Γ ≈ 0.85
    for the 450 × 220 nm TE mode.  Used only for sanity-checking mode results;
    the FDTD eigensolver gives the true value.
    """
    gamma = 0.85  # typical TE confinement factor for this cross-section
    return gamma * n_core + (1 - gamma) * n_clad
