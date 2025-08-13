
# cpw_resonator_tools.py
# A small helper library to design CPW resonators with consistent units and fewer surprises.
# - All frequencies are in Hz.
# - All lengths are in meters.
# - Phase velocity uses v = 1/sqrt(LC), which includes kinetic inductance when provided.
#
# Usage examples (see __main__):
#   python cpw_resonator_tools.py --yaml /path/to/params.yaml --print

from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Any
try:
    from scipy import pi, sinh
    from scipy.special import ellipk
except Exception as e:
    raise ImportError("This module requires SciPy (scipy.special.ellipk). Please install scipy.") from e

# Physical constants
mu_0 = 4e-7 * np.pi
epsilon_0 = 8.854187817e-12
speedoflight = 299_792_458.0

# Convenience units
GHz = 1e9
MHz = 1e6
kHz = 1e3
mm = 1e-3
um = 1e-6

@dataclass
class CPWGeometry:
    width: float      # center conductor width [m]
    gap: float        # gap to ground [m]
    h_sub: float      # substrate thickness [m]
    eps_r: float      # substrate relative permittivity
    sheet_inductance: float = 0.0  # [H/sq], kinetic inductance per square (L_k)

@dataclass
class CPWParams:
    # Derived parameters
    eps_eff: float
    L: float          # per-unit-length inductance [H/m]
    C: float          # per-unit-length capacitance [F/m]
    Z0: float         # characteristic impedance [Ohm]
    v: float          # phase velocity [m/s]

def _elliptic_ratio(a: float, b: float, h: float) -> float:
    """Helper used in CPW effective permittivity and L/C formulas."""
    k0 = float(a) / b
    k0p = np.sqrt(1 - k0**2)
    # Hammerstad–Jensen-like CPW approximation (as in user's code): use sinh-based k3
    k3 = sinh(np.pi * a / (4 * h)) / sinh(np.pi * b / (4 * h))
    k3p = np.sqrt(1 - k3**2)
    return (ellipk(k0p**2) * ellipk(k3**2)) / (ellipk(k0**2) * ellipk(k3p**2))

def compute_cpw_params(geom: CPWGeometry) -> CPWParams:
    a = geom.width
    b = geom.width + 2 * geom.gap
    h = geom.h_sub
    Ktwid = _elliptic_ratio(a, b, h)

    eps_eff = 1 + (geom.eps_r - 1) * Ktwid / 2.0

    # L and C per unit length (includes kinetic inductance term L_k/width)
    k0 = a / b
    k0p = np.sqrt(1 - k0**2)
    L_geo = (mu_0 / 4.0) * (ellipk(k0p**2) / ellipk(k0**2))
    L_kin = geom.sheet_inductance / max(a, 1e-18)  # avoid divide-by-zero
    L = L_geo + L_kin

    C = 4.0 * epsilon_0 * eps_eff * (ellipk(k0**2) / ellipk(k0p**2))
    Z0 = np.sqrt(L / C)
    v = 1.0 / np.sqrt(L * C)
    return CPWParams(eps_eff=eps_eff, L=L, C=C, Z0=Z0, v=v)

def resonator_length_from_target(f_target_Hz: float,
                                 cpw: CPWParams,
                                 resonator_type: float = 0.25,
                                 harmonic: int = 0,
                                 open_end_correction_m: float = 0.0) -> float:
    """Return total *electrical* resonator length in meters for a target frequency in Hz.

    resonator_type: 0.25 for quarter-wave, 0.5 for half-wave.
    harmonic: 0 means fundamental; otherwise same scheme as user’s code.
    open_end_correction_m: optional end correction to add for fringing (per open end).
                           For a λ/4 open-circuit end, use ΔL_open >= 0. You can pass 0.0 to disable.
    """
    if resonator_type == 0.25:
        length_factor = 0.25 * (2 * harmonic + 1)
        ends = 1  # one open end
    elif resonator_type == 0.5:
        length_factor = 0.5 * (harmonic + 1)
        ends = 2  # two open-ish ends (depends on implementation)
    else:
        raise ValueError("resonator_type must be 0.25 or 0.5")

    L_electrical = length_factor * cpw.v / f_target_Hz
    L_total = L_electrical + ends * max(open_end_correction_m, 0.0)
    return L_total

def resonator_frequency_from_length(length_m: float,
                                    cpw: CPWParams,
                                    resonator_type: float = 0.25,
                                    harmonic: int = 0,
                                    open_end_correction_m: float = 0.0) -> float:
    """Return center frequency in Hz for a given total length in meters (including any end-correction)."""
    if resonator_type == 0.25:
        length_factor = 0.25 * (2 * harmonic + 1)
        ends = 1
    elif resonator_type == 0.5:
        length_factor = 0.5 * (harmonic + 1)
        ends = 2
    else:
        raise ValueError("resonator_type must be 0.25 or 0.5")
    L_electrical = max(length_m - ends * max(open_end_correction_m, 0.0), 1e-18)
    return length_factor * cpw.v / L_electrical

def pretty_params_table(cpw: CPWParams) -> Dict[str, float]:
    return {
        "eps_eff": float(np.round(cpw.eps_eff, 3)),
        "Z0_ohm": float(np.round(cpw.Z0, 2)),
        "L_nH_per_m": float(np.round(cpw.L * 1e9, 3)),
        "C_pF_per_m": float(np.round(cpw.C * 1e12, 3)),
        "v_m_per_s": float(np.round(cpw.v, 1)),
    }

# ---------- Optional YAML workflow ----------
def load_yaml_params(d: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the user's YAML schema into SI units and return a dict."""
    res = d.get("resonator", {})
    geom = res.get("geometry", {})
    eps = res.get("eps", 11)
    h_um = d.get("resonator", {}).get("substrate_h", 1500.0)
    width_um = geom.get("w", 1.8)
    gap_um = geom.get("g", 1.0)
    sheet_L = float(res.get("sheet_inductance", 0.0))  # H/sq (set in YAML if desired)

    # Frequencies: YAML is in GHz; convert to Hz
    f_GHz = float(res.get("frequency", 7.7))
    f_corr_GHz = float(res.get("frequency_correction", 0.0))
    f_target_Hz = (f_GHz + f_corr_GHz) * GHz

    info = {
        "f_target_Hz": f_target_Hz,
        "res_type": float(res.get("type", 0.25)),
        "geom": CPWGeometry(width=width_um * um,
                            gap=gap_um * um,
                            h_sub=h_um * um,
                            eps_r=float(eps),
                            sheet_inductance=sheet_L)
    }
    return info

def design_from_yaml_dict(params: Dict[str, Any],
                          harmonic: int = 0,
                          open_end_correction_m: float = 0.0) -> Dict[str, Any]:
    info = load_yaml_params(params)
    cpw = compute_cpw_params(info["geom"])
    L_total = resonator_length_from_target(info["f_target_Hz"], cpw,
                                           resonator_type=info["res_type"],
                                           harmonic=harmonic,
                                           open_end_correction_m=open_end_correction_m)
    f_check = resonator_frequency_from_length(L_total, cpw,
                                              resonator_type=info["res_type"],
                                              harmonic=harmonic,
                                              open_end_correction_m=open_end_correction_m)
    out = {
        "target_f_GHz": info["f_target_Hz"] / GHz,
        "designed_length_um": L_total / um,
        "recomputed_f_GHz": f_check / GHz,
        "cpw": pretty_params_table(cpw),
    }
    return out

if __name__ == "__main__":
    import argparse, yaml, json, sys
    ap = argparse.ArgumentParser(description="Design a CPW resonator length safely (Hz in/out).")
    ap.add_argument("--yaml", type=str, help="Path to params.yaml", required=True)
    ap.add_argument("--harmonic", type=int, default=0, help="Harmonic index (0=fundamental)")
    ap.add_argument("--open_end_um", type=float, default=0.0, help="Open-end correction PER open end (um)")
    ap.add_argument("--print", action="store_true", help="Print a JSON summary")
    args = ap.parse_args()
    with open(args.yaml, "r") as f:
        params = yaml.safe_load(f)
    result = design_from_yaml_dict(params, harmonic=args.harmonic,
                                   open_end_correction_m=args.open_end_um * 1e-6)
    if args.print:
        print(json.dumps(result, indent=2))
    else:
        sys.stdout.write(json.dumps(result))
