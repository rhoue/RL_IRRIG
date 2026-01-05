"""
Aides UI pour configurer l'architecture MLP de la politique (net_arch).
"""
from typing import Any, Dict, Optional

import streamlit as st


def render_mlp_policy_config(policy_type: str, language: str = "fr", key_prefix: str = "") -> Optional[Dict[str, Any]]:
    """
    Affiche les contrôles d'architecture MLP lorsque la politique est MlpPolicy.
    Retourne un policy_kwargs compatible stable-baselines3, ou None pour les valeurs par défaut.
    """
    if policy_type != "MlpPolicy":
        return None

    header = "🧠 Paramètres MLP de la politique" if language == "fr" else "🧠 MLP policy parameters"
    helper = (
        "Définir la taille des couches cachées pour la politique MLP. "
        "Laisser les valeurs par défaut revient au comportement standard de Stable-Baselines3 (64x64)."
        if language == "fr"
        else "Set hidden layer sizes for the MLP policy. Leaving defaults keeps Stable-Baselines3 standard (64x64)."
    )
    prefix = f"{key_prefix}_" if key_prefix else ""

    with st.expander(header, expanded=False):
        st.markdown(helper)
        hidden_layer_1 = st.number_input(
            "Taille couche cachée 1" if language == "fr" else "Hidden layer 1 size",
            min_value=16,
            max_value=512,
            value=64,
            step=16,
            key=f"{prefix}mlp_hidden1",
        )
        hidden_layer_2 = st.number_input(
            "Taille couche cachée 2 (0 pour désactiver)" if language == "fr" else "Hidden layer 2 size (0 to disable)",
            min_value=0,
            max_value=512,
            value=64,
            step=16,
            key=f"{prefix}mlp_hidden2",
        )

        net_arch = [hidden_layer_1]
        if hidden_layer_2 > 0:
            net_arch.append(hidden_layer_2)

        # Policy kwargs pour PPO (stable-baselines3)
        return {"net_arch": net_arch}

    return None
