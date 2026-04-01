"""
ExoHabitAI — Backend Utilities
utils.py: shared helpers for app.py
"""

import numpy as np
import pandas as pd

# ── Habitable Zone Constants ───────────────────────────────────────────────────
HZ_TEMP_MIN = 180   # Kelvin — inner edge
HZ_TEMP_MAX = 310   # Kelvin — outer edge
EARTH_RADIUS_MAX = 2.5  # Earth radii — super-Earth upper bound

# ── Feature metadata ───────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "radius_earth":          ("Planet Radius",          "Earth radii",   0.1,  30.0),
    "mass_earth":            ("Planet Mass",            "Earth masses",  0.01, 5000),
    "orbital_period":        ("Orbital Period",         "days",          0.5,  100000),
    "semimajor_axis":        ("Semi-major Axis",        "AU",            0.01, 100),
    "eq_temp_k":             ("Equilibrium Temperature","K",             50,   2000),
    "density":               ("Planet Density",         "g/cm³",         0.01, 100),
    "star_temp_k":           ("Star Temperature",       "K",             2000, 50000),
    "star_luminosity":       ("Star Luminosity",        "L☉",           -5,   6),
    "star_metallicity":      ("Star Metallicity",       "[Fe/H]",        -3,   3),
    "habitability_score":    ("Habitability Score",     "index",         0,    1),
    "stellar_compatibility": ("Stellar Compatibility",  "index",         0,    2),
    "orbital_stability":     ("Orbital Stability",      "index",         0,    20),
}

# ── Validation bounds ──────────────────────────────────────────────────────────
BOUNDS = {k: (v[2], v[3]) for k, v in FEATURE_DESCRIPTIONS.items()}


def validate_bounds(data: dict) -> list:
    """Return list of out-of-range warnings (non-blocking)."""
    warnings = []
    for field, (lo, hi) in BOUNDS.items():
        if field in data:
            try:
                val = float(data[field])
                if not (lo <= val <= hi):
                    warnings.append(f"{field} value {val} is outside expected range [{lo}, {hi}]")
            except (ValueError, TypeError):
                pass
    return warnings


def compute_hz_check(eq_temp_k: float) -> dict:
    """Check if equilibrium temperature falls in the habitable zone."""
    in_hz = HZ_TEMP_MIN <= eq_temp_k <= HZ_TEMP_MAX
    return {
        "in_habitable_zone": in_hz,
        "hz_min_k":          HZ_TEMP_MIN,
        "hz_max_k":          HZ_TEMP_MAX,
        "eq_temp_k":         eq_temp_k,
        "message":           "Within habitable zone temperature range" if in_hz
                             else f"Outside habitable zone ({HZ_TEMP_MIN}–{HZ_TEMP_MAX} K)",
    }


def format_planet_summary(data: dict) -> dict:
    """Return a human-readable summary of planet parameters."""
    return {
        "Planet Radius":          f"{data.get('radius_earth', 'N/A')} R⊕",
        "Planet Mass":            f"{data.get('mass_earth', 'N/A')} M⊕",
        "Orbital Period":         f"{data.get('orbital_period', 'N/A')} days",
        "Equilibrium Temp":       f"{data.get('eq_temp_k', 'N/A')} K",
        "Star Type":              data.get("star_spectype", "Unknown"),
        "Star Temperature":       f"{data.get('star_temp_k', 'N/A')} K",
        "Habitability Score":     data.get("habitability_score", "N/A"),
        "Stellar Compatibility":  data.get("stellar_compatibility", "N/A"),
        "Orbital Stability":      data.get("orbital_stability", "N/A"),
    }


def score_to_percentage_label(score: float) -> str:
    """Convert probability score (0-100) to descriptive label."""
    if score >= 90: return "Excellent candidate for life"
    if score >= 70: return "Strong habitability indicators"
    if score >= 50: return "Moderate habitability potential"
    if score >= 30: return "Marginal habitability"
    if score >= 10: return "Low habitability probability"
    return "Extremely unlikely to support life"
