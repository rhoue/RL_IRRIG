"""
Aides UI pour configurer les paramètres de sol et de tension (sidebar Streamlit).
"""
from typing import Dict

import streamlit as st

from src.utils_physics_config import get_default_soil_config, get_soil_params_ranges


def render_soil_and_tension_config(language: str = "fr") -> Dict[str, float]:
    """
    Affiche les curseurs de configuration du sol et sauvegarde dans session_state.
    Retourne le dictionnaire des paramètres de sol.
    """
    default_soil = get_default_soil_config()
    soil_ranges = get_soil_params_ranges()

    labels = {
        "header": "### 🌱 Paramètres du sol" if language == "fr" else "### 🌱 Soil parameters",
        "z_r": "Profondeur zone racinaire Z_r (mm)" if language == "fr" else "Root zone depth Z_r (mm)",
        "z_r_help": "Profondeur de la zone racinaire en millimètres" if language == "fr" else "Root zone depth in millimeters",
        "theta_s": "θ_s (saturation)",
        "theta_s_help": "Teneur en eau volumique à saturation" if language == "fr" else "Volumetric water content at saturation",
        "theta_fc": "θ_fc (capacité au champ)" if language == "fr" else "θ_fc (field capacity)",
        "theta_fc_help": "Teneur en eau à la capacité au champ" if language == "fr" else "Water content at field capacity",
        "theta_wp": "θ_wp (point de flétrissement)" if language == "fr" else "θ_wp (wilting point)",
        "theta_wp_help": "Teneur en eau au point de flétrissement" if language == "fr" else "Water content at wilting point",
        "tension_header": "#### Paramètres de tension (cbar)" if language == "fr" else "#### Tension parameters (cbar)",
        "kd": "k_d (coefficient drainage)",
        "kd_help": "Coefficient de drainage" if language == "fr" else "Drainage coefficient",
        "eta_I": "η_I (efficacité irrigation)" if language == "fr" else "η_I (irrigation efficiency)",
        "eta_I_help": "Efficacité d'irrigation (0-1)" if language == "fr" else "Irrigation efficiency (0-1)",
        "psi_ET": "ψ_ET_crit (seuil de stress)" if language == "fr" else "ψ_ET_crit (stress threshold)",
        "psi_ET_help": "Seuil de stress ET" if language == "fr" else "ET stress threshold",
    }

    st.markdown(labels["header"])

    Z_r = st.slider(
        labels["z_r"],
        min_value=soil_ranges["Z_r"]["min"],
        max_value=soil_ranges["Z_r"]["max"],
        value=default_soil["Z_r"],
        step=soil_ranges["Z_r"]["step"],
        help=labels["z_r_help"],
    )

    col1, col2 = st.columns(2)
    with col1:
        theta_s = st.slider(
            "θ_s (saturation)",
            min_value=soil_ranges["theta_s"]["min"],
            max_value=soil_ranges["theta_s"]["max"],
            value=default_soil["theta_s"],
            step=soil_ranges["theta_s"]["step"],
            help=labels["theta_s_help"],
        )
        theta_fc = st.slider(
            labels["theta_fc"],
            min_value=soil_ranges["theta_fc"]["min"],
            max_value=soil_ranges["theta_fc"]["max"],
            value=default_soil["theta_fc"],
            step=soil_ranges["theta_fc"]["step"],
            help=labels["theta_fc_help"],
        )
    with col2:
        theta_wp = st.slider(
            labels["theta_wp"],
            min_value=soil_ranges["theta_wp"]["min"],
            max_value=soil_ranges["theta_wp"]["max"],
            value=default_soil["theta_wp"],
            step=soil_ranges["theta_wp"]["step"],
            help=labels["theta_wp_help"],
        )

    st.markdown(labels["tension_header"])
    col1, col2, col3 = st.columns(3)
    with col1:
        psi_sat = st.number_input(
            "ψ_sat",
            min_value=soil_ranges["psi_sat"]["min"],
            max_value=soil_ranges["psi_sat"]["max"],
            value=default_soil["psi_sat"],
            step=soil_ranges["psi_sat"]["step"],
        )
    with col2:
        psi_fc = st.number_input(
            "ψ_fc",
            min_value=soil_ranges["psi_fc"]["min"],
            max_value=soil_ranges["psi_fc"]["max"],
            value=default_soil["psi_fc"],
            step=soil_ranges["psi_fc"]["step"],
        )
    with col3:
        psi_wp = st.number_input(
            "ψ_wp",
            min_value=soil_ranges["psi_wp"]["min"],
            max_value=soil_ranges["psi_wp"]["max"],
            value=default_soil["psi_wp"],
            step=soil_ranges["psi_wp"]["step"],
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        k_d = st.slider(
            "k_d (coefficient drainage)",
            min_value=soil_ranges["k_d"]["min"],
            max_value=soil_ranges["k_d"]["max"],
            value=default_soil["k_d"],
            step=soil_ranges["k_d"]["step"],
            help=labels["kd_help"],
        )
    with col2:
        eta_I = st.slider(
            "η_I (efficacité irrigation)",
            min_value=soil_ranges["eta_I"]["min"],
            max_value=soil_ranges["eta_I"]["max"],
            value=default_soil["eta_I"],
            step=soil_ranges["eta_I"]["step"],
            help=labels["eta_I_help"],
        )
    with col3:
        psi_ET_crit = st.number_input(
            labels["psi_ET"],
            min_value=soil_ranges["psi_ET_crit"]["min"],
            max_value=soil_ranges["psi_ET_crit"]["max"],
            value=default_soil["psi_ET_crit"],
            step=soil_ranges["psi_ET_crit"]["step"],
            help=labels["psi_ET_help"],
        )

    # Collecte finale des paramètres de sol/tension
    soil_params = {
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

    # Mise à disposition pour tous les scénarios
    st.session_state.soil_params = soil_params
    return soil_params
