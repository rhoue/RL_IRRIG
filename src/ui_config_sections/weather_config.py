"""
Aides UI pour configurer les paramètres météorologiques (ET0 + pluie).
"""
from typing import Dict

import streamlit as st

from src.utils_physics_config import get_default_weather_config, get_weather_params_ranges


def render_weather_config(language: str = "fr") -> Dict[str, float]:
    """
    Affiche les contrôles météo et les enregistre dans session_state.
    Retourne le dictionnaire des paramètres météo.
    """
    default_weather = get_default_weather_config()
    weather_ranges = get_weather_params_ranges()

    labels = {
        "header": "### 🌦️ Paramètres météorologiques" if language == "fr" else "### 🌦️ Weather parameters",
        "et0_header": "#### ET0 (évapotranspiration de référence)" if language == "fr" else "#### ET0 (reference evapotranspiration)",
        "et0_base": "ET0 base (mm/j)" if language == "fr" else "ET0 base (mm/day)",
        "et0_amp": "ET0 amplitude",
        "et0_noise": "ET0 bruit" if language == "fr" else "ET0 noise",
        "rain_header": "#### Pluie" if language == "fr" else "#### Rainfall",
        "p_rain_early": "Prob. pluie début saison" if language == "fr" else "Rain prob. early season",
        "p_rain_mid": "Prob. pluie milieu saison" if language == "fr" else "Rain prob. mid season",
        "p_rain_late": "Prob. pluie fin saison" if language == "fr" else "Rain prob. late season",
        "rain_min": "Pluie min (mm)" if language == "fr" else "Rain min (mm)",
        "rain_max": "Pluie max (mm)" if language == "fr" else "Rain max (mm)",
    }

    st.markdown(labels["header"])

    st.markdown(labels["et0_header"])
    col1, col2, col3 = st.columns(3)
    with col1:
        et0_base = st.number_input(
            labels["et0_base"],
            min_value=weather_ranges["et0_base"]["min"],
            max_value=weather_ranges["et0_base"]["max"],
            value=default_weather["et0_base"],
            step=weather_ranges["et0_base"]["step"],
        )
    with col2:
        et0_amp = st.number_input(
            labels["et0_amp"],
            min_value=weather_ranges["et0_amp"]["min"],
            max_value=weather_ranges["et0_amp"]["max"],
            value=default_weather["et0_amp"],
            step=weather_ranges["et0_amp"]["step"],
        )
    with col3:
        et0_noise = st.number_input(
            labels["et0_noise"],
            min_value=weather_ranges["et0_noise"]["min"],
            max_value=weather_ranges["et0_noise"]["max"],
            value=default_weather["et0_noise"],
            step=weather_ranges["et0_noise"]["step"],
        )

    st.markdown(labels["rain_header"])
    col1, col2 = st.columns(2)
    with col1:
        p_rain_early = st.slider(
            labels["p_rain_early"],
            min_value=weather_ranges["p_rain_early"]["min"],
            max_value=weather_ranges["p_rain_early"]["max"],
            value=default_weather["p_rain_early"],
            step=weather_ranges["p_rain_early"]["step"],
        )
        p_rain_mid = st.slider(
            labels["p_rain_mid"],
            min_value=weather_ranges["p_rain_mid"]["min"],
            max_value=weather_ranges["p_rain_mid"]["max"],
            value=default_weather["p_rain_mid"],
            step=weather_ranges["p_rain_mid"]["step"],
        )
        p_rain_late = st.slider(
            labels["p_rain_late"],
            min_value=weather_ranges["p_rain_late"]["min"],
            max_value=weather_ranges["p_rain_late"]["max"],
            value=default_weather["p_rain_late"],
            step=weather_ranges["p_rain_late"]["step"],
        )
    with col2:
        rain_min = st.number_input(
            labels["rain_min"],
            min_value=weather_ranges["rain_min"]["min"],
            max_value=weather_ranges["rain_min"]["max"],
            value=default_weather["rain_min"],
            step=weather_ranges["rain_min"]["step"],
        )
        rain_max = st.number_input(
            labels["rain_max"],
            min_value=weather_ranges["rain_max"]["min"],
            max_value=weather_ranges["rain_max"]["max"],
            value=default_weather["rain_max"],
            step=weather_ranges["rain_max"]["step"],
        )

    # Paramètres météo consolidés (ET0 + pluie)
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }

    # Mise à disposition pour l'ensemble des scénarios
    st.session_state.weather_params = weather_params
    return weather_params
