"""
Aides UI pour configurer l'environnement (durée de saison, irrigation max, seed).
"""
from typing import Any, Dict

import streamlit as st

from src.utils_physics_config import (
    DEFAULT_MAX_IRRIGATION,
    DEFAULT_SEASON_LENGTH,
    DEFAULT_SEED,
    get_env_params_ranges,
)


def render_environment_config(language: str = "fr") -> Dict[str, Any]:
    """
    Affiche les contrôles d'environnement et les enregistre dans session_state.
    Retourne un dict avec durée de saison, irrigation max et graine aléatoire.
    """
    env_ranges = get_env_params_ranges()

    labels = {
        "header": "### 🎯 Paramètres d'environnement" if language == "fr" else "### 🎯 Environment parameters",
        "season_length": "Longueur de saison (jours)" if language == "fr" else "Season length (days)",
        "season_length_help": "Durée de la saison culturale en jours" if language == "fr" else "Crop season length in days",
        "max_irrig": "Irrigation maximale (mm/jour)" if language == "fr" else "Maximum irrigation (mm/day)",
        "seed": "Graine aléatoire" if language == "fr" else "Random seed",
        "seed_help": "Pour la reproductibilité" if language == "fr" else "For reproducibility",
    }

    st.markdown(labels["header"])

    season_length = st.slider(
        labels["season_length"],
        min_value=env_ranges["season_length"]["min"],
        max_value=env_ranges["season_length"]["max"],
        value=DEFAULT_SEASON_LENGTH,
        step=env_ranges["season_length"]["step"],
        help=labels["season_length_help"],
    )

    max_irrigation = st.slider(
        labels["max_irrig"],
        min_value=env_ranges["max_irrigation"]["min"],
        max_value=env_ranges["max_irrigation"]["max"],
        value=DEFAULT_MAX_IRRIGATION,
        step=env_ranges["max_irrigation"]["step"],
    )

    seed = st.number_input(
        labels["seed"],
        min_value=env_ranges["seed"]["min"],
        max_value=env_ranges["seed"]["max"],
        value=DEFAULT_SEED,
        step=env_ranges["seed"]["step"],
        help=labels["seed_help"],
    )

    # Paramètres agrégés pour l'environnement physique
    env_params: Dict[str, Any] = {
        "season_length": season_length,
        "max_irrigation": max_irrigation,
        "seed": seed,
    }

    st.session_state.environment_params = env_params
    st.session_state.season_length = season_length
    st.session_state.max_irrigation = max_irrigation
    st.session_state.seed = seed
    return env_params
