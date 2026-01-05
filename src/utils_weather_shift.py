"""
Applique des décalages météorologiques pour tester la robustesse (weather shift).

Modes :
- none     : pas de modification
- seasonal : biais constant sur toute la saison (pluie / ET0)
- shocks   : chocs ponctuels sur un nombre de jours
"""
from typing import Any, Dict, Tuple
import numpy as np


def apply_weather_shift(
    rain: np.ndarray,
    et0: np.ndarray,
    cfg: Dict[str, Any] | None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne (rain_shifted, et0_shifted) selon la config.
    cfg attend les clés : robustness_eval, rain_shift_pct, et0_shift_pct, shift_mode,
    shock_days, shock_window.
    """
    if not cfg or not cfg.get("robustness_eval", False):
        return rain, et0

    rain_shift_pct = cfg.get("rain_shift_pct", 0)
    et0_shift_pct = cfg.get("et0_shift_pct", 0)
    mode = cfg.get("shift_mode", "none")
    shock_days = int(cfg.get("shock_days", 0) or 0)
    shock_window = int(cfg.get("shock_window", 1) or 1)

    r = np.array(rain, dtype=float, copy=True)
    e = np.array(et0, dtype=float, copy=True)

    def _apply_indices(indices):
        if indices.size == 0:
            return
        r[indices] *= 1.0 + rain_shift_pct / 100.0
        e[indices] *= 1.0 + et0_shift_pct / 100.0

    if mode == "seasonal":
        _apply_indices(np.arange(len(r)))
        return r, e

    if mode == "shocks" and shock_days > 0:
        gen = rng if rng is not None else np.random.default_rng()
        n_days = len(r)
        shock_days = min(shock_days, n_days)
        indices = set()
        # On peut regrouper en petites fenêtres pour éviter les duplications
        while len(indices) < shock_days:
            start = int(gen.integers(0, n_days))
            span = min(shock_window, n_days - start)
            for k in range(span):
                indices.add(start + k)
                if len(indices) >= shock_days:
                    break
        _apply_indices(np.fromiter(indices, dtype=int))
        return r, e

    # mode none ou inconnu
    return r, e
