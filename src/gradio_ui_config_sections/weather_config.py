"""
Gradio helpers for weather parameters.
"""

from typing import Dict

import gradio as gr

from src.utils_physics_config import get_default_weather_config, get_weather_params_ranges


def build_weather_config(language: str = "fr") -> Dict[str, gr.Component]:
    default_weather = get_default_weather_config()
    weather_ranges = get_weather_params_ranges()

    labels = {
        "header": "### 🌦️ Paramètres météorologiques / Weather parameters",
        "et0_header": "#### ET0 (évapotranspiration de référence) / ET0 (reference evapotranspiration)",
        "et0_base": "ET0 base (mm/j) / ET0 base (mm/day)",
        "et0_amp": "ET0 amplitude / ET0 amplitude",
        "et0_noise": "ET0 bruit / ET0 noise",
        "rain_header": "#### Pluie / Rainfall",
        "p_rain_early": "Prob. pluie début saison / Rain prob. early season",
        "p_rain_mid": "Prob. pluie milieu saison / Rain prob. mid season",
        "p_rain_late": "Prob. pluie fin saison / Rain prob. late season",
        "rain_min": "Pluie min (mm) / Rain min (mm)",
        "rain_max": "Pluie max (mm) / Rain max (mm)",
    }

    gr.Markdown(labels["header"])
    gr.Markdown(labels["et0_header"])
    with gr.Row():
        et0_base = gr.Number(
            value=default_weather["et0_base"],
            label=labels["et0_base"],
        )
        et0_amp = gr.Number(
            value=default_weather["et0_amp"],
            label=labels["et0_amp"],
        )
        et0_noise = gr.Number(
            value=default_weather["et0_noise"],
            label=labels["et0_noise"],
        )

    gr.Markdown(labels["rain_header"])
    with gr.Row():
        p_rain_early = gr.Slider(
            weather_ranges["p_rain_early"]["min"],
            weather_ranges["p_rain_early"]["max"],
            value=default_weather["p_rain_early"],
            step=weather_ranges["p_rain_early"]["step"],
            label=labels["p_rain_early"],
        )
        p_rain_mid = gr.Slider(
            weather_ranges["p_rain_mid"]["min"],
            weather_ranges["p_rain_mid"]["max"],
            value=default_weather["p_rain_mid"],
            step=weather_ranges["p_rain_mid"]["step"],
            label=labels["p_rain_mid"],
        )
        p_rain_late = gr.Slider(
            weather_ranges["p_rain_late"]["min"],
            weather_ranges["p_rain_late"]["max"],
            value=default_weather["p_rain_late"],
            step=weather_ranges["p_rain_late"]["step"],
            label=labels["p_rain_late"],
        )
    with gr.Row():
        rain_min = gr.Number(
            value=default_weather["rain_min"],
            label=labels["rain_min"],
        )
        rain_max = gr.Number(
            value=default_weather["rain_max"],
            label=labels["rain_max"],
        )

    return {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
