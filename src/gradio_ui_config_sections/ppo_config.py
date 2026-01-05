"""
Gradio helpers for PPO training configuration.
"""

from typing import Any, Dict

import gradio as gr

from src.utils_ppo_training import get_default_ppo_hyperparams_ui, get_ppo_hyperparams_ui_ranges


def build_ppo_training_section(language: str) -> Dict[str, Any]:
    labels = {
        "fr": {
            "ppo_params": "### 🚀 Paramètres d'entraînement PPO / PPO training parameters",
            "ppo_desc": (
                "<div style=\"background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; "
                "margin-bottom: 1rem; border-left: 4px solid #ffc107;\">"
                "<strong>PPO (Proximal Policy Optimization)</strong> est un algorithme d'apprentissage par renforcement "
                "qui apprend une politique d'irrigation optimale en explorant l'espace d'actions et en optimisant "
                "les récompenses cumulées.</div>"
            ),
            "total_steps": "Nombre total de pas d'entraînement / Total training steps",
            "policy_type": "Type de politique / Policy type",
            "hyperparams": "### 📊 Hyperparamètres PPO / PPO hyperparameters",
            "hyper_desc": (
                "<div style=\"background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; "
                "margin-bottom: 1rem; font-size: 0.9rem;\">"
                "<strong>PRINCIPE PPO :</strong> Optimise la politique en limitant les mises à jour trop importantes "
                "via un mécanisme de clipping. Cela stabilise l'apprentissage et évite la dégradation de la performance."
                "<br><strong>Objectif :</strong> Maximiser les récompenses cumulées tout en maintenant la stabilité."
                "</div>"
            ),
            "adv_params": "Hyperparamètres avancés / Advanced hyperparameters",
        },
        "en": {
            "ppo_params": "### 🚀 PPO training parameters",
            "ppo_desc": (
                "<div style=\"background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; "
                "margin-bottom: 1rem; border-left: 4px solid #ffc107;\">"
                "<strong>PPO (Proximal Policy Optimization)</strong> is a reinforcement learning algorithm "
                "that learns an optimal irrigation policy by exploring the action space and optimizing "
                "cumulative rewards.</div>"
            ),
            "total_steps": "Total training steps / Nombre total de pas d'entraînement",
            "policy_type": "Policy type / Type de politique",
            "hyperparams": "### 📊 PPO hyperparameters / Hyperparamètres PPO",
            "hyper_desc": (
                "<div style=\"background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; "
                "margin-bottom: 1rem; font-size: 0.9rem;\">"
                "<strong>PPO PRINCIPLE:</strong> Optimizes the policy while limiting large updates via clipping. "
                "This stabilizes learning and avoids performance collapse."
                "<br><strong>Goal:</strong> Maximize cumulative rewards while staying stable."
                "</div>"
            ),
            "adv_params": "Advanced hyperparameters / Hyperparamètres avancés",
        },
    }[language]

    ppo_defaults = get_default_ppo_hyperparams_ui()
    ppo_ranges = get_ppo_hyperparams_ui_ranges()

    gr.Markdown(labels["ppo_params"])
    gr.HTML(labels["ppo_desc"])

    with gr.Row():
        total_timesteps = gr.Number(
            value=ppo_defaults["total_timesteps"],
            label=labels["total_steps"],
            precision=0,
        )
        policy_type = gr.Dropdown(
            choices=["MlpPolicy", "CnnPolicy"],
            value="MlpPolicy",
            label=labels["policy_type"],
        )

    gr.Markdown(labels["hyperparams"])
    gr.HTML(labels["hyper_desc"])

    with gr.Accordion(labels["adv_params"], open=False):
        n_steps = gr.Number(
            value=ppo_defaults["n_steps"],
            label="n_steps",
            precision=0,
        )
        batch_size = gr.Number(
            value=ppo_defaults["batch_size"],
            label="batch_size",
            precision=0,
        )
        learning_rate = gr.Number(
            value=ppo_defaults["learning_rate"],
            label="learning_rate",
            precision=5,
        )
        gamma = gr.Slider(
            ppo_ranges["gamma"]["min"],
            ppo_ranges["gamma"]["max"],
            value=ppo_defaults["gamma"],
            step=ppo_ranges["gamma"]["step"],
            label="gamma (discount factor)",
        )
        gae_lambda = gr.Slider(
            ppo_ranges["gae_lambda"]["min"],
            ppo_ranges["gae_lambda"]["max"],
            value=ppo_defaults["gae_lambda"],
            step=ppo_ranges["gae_lambda"]["step"],
            label="gae_lambda",
        )
        clip_range = gr.Slider(
            ppo_ranges["clip_range"]["min"],
            ppo_ranges["clip_range"]["max"],
            value=ppo_defaults["clip_range"],
            step=ppo_ranges["clip_range"]["step"],
            label="clip_range",
        )
        ent_coef = gr.Number(
            value=ppo_defaults["ent_coef"],
            label="ent_coef",
            precision=5,
        )
        vf_coef = gr.Number(
            value=ppo_defaults["vf_coef"],
            label="vf_coef",
            precision=5,
        )

    return {
        "total_timesteps": total_timesteps,
        "policy_type": policy_type,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
    }
