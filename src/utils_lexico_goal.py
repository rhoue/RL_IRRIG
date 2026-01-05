"""
Lexicographic goal programming utilities (Option A).

Provides a reward wrapper that can recompute rewards from tiered deviations
exposed in env `info` under the key `lexico_deviations` (list/tuple of floats).
If the key is absent, the base reward is left unchanged.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

try:
    import gymnasium as gym  # type: ignore
except ImportError:  # pragma: no cover - gymnasium optional
    gym = None


@dataclass
class LexicoConfig:
    enabled: bool = False
    lambdas: Optional[Iterable[float]] = None  # priority weights λ1 >> λ2 >> ...
    dense_factor: float = 0.0  # optional scaling for dense deviations


def compose_lexico_reward(
    base_reward: float, deviations: Optional[Iterable[float]], config: LexicoConfig
) -> float:
    """
    Compose a lexicographic-style reward:
    - If pas de déviation: on garde la reward d'origine.
    - Si déviations présentes: on soustrait la pénalité lexicographique au reward de base.
    """
    if not config.enabled or deviations is None:
        return base_reward
    lambdas = list(config.lambdas or [])
    devs = list(deviations)
    # pad lambdas with diminishing weights if fewer provided
    while len(lambdas) < len(devs):
        next_lambda = lambdas[-1] * 0.01 if lambdas else 1.0
        lambdas.append(next_lambda)
    penalty = sum(l * d for l, d in zip(lambdas, devs))
    return base_reward - penalty


class LexicoRewardWrapper(gym.Wrapper if gym else object):  # type: ignore
    """
    Gym wrapper applying lexicographic reward composition when enabled.
    Expects env.info['lexico_deviations'] to hold an iterable of deviations.
    """

    def __init__(self, env: Any, config: LexicoConfig):
        if gym is None:
            raise ImportError("gymnasium is required for LexicoRewardWrapper")
        super().__init__(env)
        self.config = config

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        deviations = info.get("lexico_deviations")
        reward = compose_lexico_reward(reward, deviations, self.config)
        return obs, reward, terminated, truncated, info


def wrap_with_lexico(env: Any, config: LexicoConfig) -> Any:
    """Wrap env with lexicographic reward if enabled, else return env unchanged."""
    if config.enabled and gym is not None:
        return LexicoRewardWrapper(env, config)
    return env


def build_lexico_config_from_session(session_state: Dict[str, Any]) -> LexicoConfig:
    """
    Build a LexicoConfig from session_state.proposal_a_config if present.
    Mapping heuristic (priorité aux objectifs Option A):
    - enabled: proposal_a_config.enabled
    - lambdas: si goal_spec présent → [λP1, λP2, λP3], sinon ancien triplet
    - dense_factor: residual_alpha (not used in wrapper yet)
    """
    cfg_raw = session_state.get("proposal_a_config", {}) if session_state else {}
    goal_spec = cfg_raw.get("goal_spec") or session_state.get("goal_spec", {}) if session_state else {}
    enabled = cfg_raw.get("enabled", False)

    if goal_spec and isinstance(goal_spec, dict):
        lam = goal_spec.get("lambdas", {})
        lambdas_list = [lam.get("P1", 0.0), lam.get("P2", 0.0), lam.get("P3", 0.0)]
    else:
        lambdas_list = [
            cfg_raw.get("feasibility_penalty", 0.0),
            cfg_raw.get("stability_margin", 0.0),
            cfg_raw.get("teacher_weight", 0.0),
        ]
    dense_factor = cfg_raw.get("residual_alpha", 0.0)
    return LexicoConfig(enabled=enabled, lambdas=lambdas_list, dense_factor=dense_factor)
