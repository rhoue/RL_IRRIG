"""
Gradio helpers for soil/tension parameters.
"""

from typing import Dict

import gradio as gr

from src.utils_physics_config import get_default_soil_config, get_soil_params_ranges


def build_soil_and_tension_config(language: str = "fr") -> Dict[str, gr.Component]:
    default_soil = get_default_soil_config()
    soil_ranges = get_soil_params_ranges()

    labels = {
        "header": "### 🌱 Paramètres du sol / Soil parameters",
        "z_r": "Profondeur zone racinaire Z_r (mm) / Root zone depth Z_r (mm)",
        "theta_s": "θ_s (saturation) / θ_s (saturation)",
        "theta_fc": "θ_fc (capacité au champ) / θ_fc (field capacity)",
        "theta_wp": "θ_wp (point de flétrissement) / θ_wp (wilting point)",
        "tension_header": "#### Paramètres de tension (cbar) / Tension parameters (cbar)",
        "psi_sat": "ψ_sat / ψ_sat",
        "psi_fc": "ψ_fc / ψ_fc",
        "psi_wp": "ψ_wp / ψ_wp",
        "k_d": "k_d (coefficient drainage) / k_d (drainage coefficient)",
        "eta_I": "η_I (efficacité irrigation) / η_I (irrigation efficiency)",
        "psi_ET": "ψ_ET_crit (seuil de stress) / ψ_ET_crit (stress threshold)",
    }

    gr.Markdown(labels["header"])
    Z_r = gr.Slider(
        soil_ranges["Z_r"]["min"],
        soil_ranges["Z_r"]["max"],
        value=default_soil["Z_r"],
        step=soil_ranges["Z_r"]["step"],
        label=labels["z_r"],
    )
    with gr.Row():
        theta_s = gr.Slider(
            soil_ranges["theta_s"]["min"],
            soil_ranges["theta_s"]["max"],
            value=default_soil["theta_s"],
            step=soil_ranges["theta_s"]["step"],
            label=labels["theta_s"],
        )
        theta_fc = gr.Slider(
            soil_ranges["theta_fc"]["min"],
            soil_ranges["theta_fc"]["max"],
            value=default_soil["theta_fc"],
            step=soil_ranges["theta_fc"]["step"],
            label=labels["theta_fc"],
        )
        theta_wp = gr.Slider(
            soil_ranges["theta_wp"]["min"],
            soil_ranges["theta_wp"]["max"],
            value=default_soil["theta_wp"],
            step=soil_ranges["theta_wp"]["step"],
            label=labels["theta_wp"],
        )

    gr.Markdown(labels["tension_header"])
    with gr.Row():
        psi_sat = gr.Number(
            value=default_soil["psi_sat"],
            label=labels["psi_sat"],
        )
        psi_fc = gr.Number(
            value=default_soil["psi_fc"],
            label=labels["psi_fc"],
        )
        psi_wp = gr.Number(
            value=default_soil["psi_wp"],
            label=labels["psi_wp"],
        )
    with gr.Row():
        k_d = gr.Slider(
            soil_ranges["k_d"]["min"],
            soil_ranges["k_d"]["max"],
            value=default_soil["k_d"],
            step=soil_ranges["k_d"]["step"],
            label=labels["k_d"],
        )
        eta_I = gr.Slider(
            soil_ranges["eta_I"]["min"],
            soil_ranges["eta_I"]["max"],
            value=default_soil["eta_I"],
            step=soil_ranges["eta_I"]["step"],
            label=labels["eta_I"],
        )
        psi_ET_crit = gr.Number(
            value=default_soil["psi_ET_crit"],
            label=labels["psi_ET"],
        )

    return {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
