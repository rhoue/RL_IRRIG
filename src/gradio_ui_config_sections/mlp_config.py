"""
Gradio helpers for MLP policy configuration.
"""

from typing import Any, Dict, Optional

import gradio as gr


def build_mlp_policy_config(policy_type: str, language: str = "fr") -> Optional[Dict[str, Any]]:
    if policy_type != "MlpPolicy":
        return None

    header = "🧠 Paramètres MLP de la politique / MLP policy parameters"
    helper = (
        "Définir la taille des couches cachées pour la politique MLP. "
        "Laisser les valeurs par défaut revient au comportement standard de Stable-Baselines3 (64x64). "
        "/ Set hidden layer sizes for the MLP policy. Leaving defaults keeps Stable-Baselines3 standard (64x64)."
    )

    with gr.Accordion(header, open=False):
        gr.Markdown(helper)
        hidden_layer_1 = gr.Number(
            label="Taille couche cachée 1 / Hidden layer 1 size",
            value=64,
            precision=0,
        )
        hidden_layer_2 = gr.Number(
            label="Taille couche cachée 2 (0 pour désactiver) / Hidden layer 2 size (0 to disable)",
            value=64,
            precision=0,
        )
    return {"hidden_layer_1": hidden_layer_1, "hidden_layer_2": hidden_layer_2}
