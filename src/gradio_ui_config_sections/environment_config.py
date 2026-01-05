"""
Gradio helpers for environment parameters.
"""

from typing import Any, Dict

import gradio as gr

from src.utils_physics_config import (
    DEFAULT_MAX_IRRIGATION,
    DEFAULT_SEASON_LENGTH,
    DEFAULT_SEED,
    get_env_params_ranges,
)


def build_environment_config(language: str = "fr") -> Dict[str, Any]:
    env_ranges = get_env_params_ranges()
    labels = {
        "header": "### 🎯 Paramètres d'environnement / Environment parameters",
        "season_length": "Longueur de saison (jours) / Season length (days)",
        "max_irrig": "Irrigation maximale (mm/jour) / Maximum irrigation (mm/day)",
        "seed": "Graine aléatoire / Random seed",
    }

    gr.Markdown(labels["header"])
    season_length = gr.Slider(
        env_ranges["season_length"]["min"],
        env_ranges["season_length"]["max"],
        value=DEFAULT_SEASON_LENGTH,
        step=env_ranges["season_length"]["step"],
        label=labels["season_length"],
    )
    max_irrigation = gr.Slider(
        env_ranges["max_irrigation"]["min"],
        env_ranges["max_irrigation"]["max"],
        value=DEFAULT_MAX_IRRIGATION,
        step=env_ranges["max_irrigation"]["step"],
        label=labels["max_irrig"],
    )
    seed = gr.Number(
        value=DEFAULT_SEED,
        label=labels["seed"],
        precision=0,
    )
    return {"season_length": season_length, "max_irrigation": max_irrigation, "seed": seed}
