"""
Aides UI pour configurer l'entraînement PPO (pas de rollout et hyperparamètres).
"""
from typing import Any, Dict, Tuple

import streamlit as st

from src.utils_ppo_training import (
    get_default_ppo_hyperparams_ui,
    get_ppo_hyperparams_ui_ranges,
)


def render_ppo_training_section(language: str) -> Tuple[int, str, Dict[str, Any]]:
    """
    Affiche les contrôles d'entraînement PPO et les hyperparamètres associés.

    Retourne:
        total_timesteps, policy_type, ppo_kwargs
    """
    t2 = {
        "fr": {
            "ppo_params": "### 🚀 Paramètres d'entraînement PPO",
            "ppo_desc": """
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #ffc107;">
            <strong>PPO (Proximal Policy Optimization)</strong> est un algorithme d'apprentissage par renforcement
            qui apprend une politique d'irrigation optimale en explorant l'espace d'actions et en optimisant
            les récompenses cumulées.
        </div>
        """,
            "total_steps": "Nombre total de pas d'entraînement",
            "total_steps_help": "Nombre total de pas de simulation pour l'entraînement. Plus élevé = meilleure politique mais plus long.",
            "policy_type": "Type de politique",
            "policy_help": "MlpPolicy : réseau de neurones MLP (recommandé pour données tabulaires). CnnPolicy : CNN (pour images).",
            "hyperparams": "### 📊 Hyperparamètres PPO",
            "hyper_desc": """
        <div style="background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
            <strong>PRINCIPE PPO :</strong> Optimise la politique en limitant les mises à jour trop importantes
            via un mécanisme de clipping. Cela stabilise l'apprentissage et évite la dégradation de la performance.
            <br><strong>Objectif :</strong> Maximiser les récompenses cumulées tout en maintenant la stabilité.
        </div>
        """,
            "adv_params": "Hyperparamètres avancés",
        },
        "en": {
            "ppo_params": "### 🚀 PPO training parameters",
            "ppo_desc": """
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #ffc107;">
            <strong>PPO (Proximal Policy Optimization)</strong> is a reinforcement learning algorithm
            that learns an optimal irrigation policy by exploring the action space and optimizing
            cumulative rewards.
        </div>
        """,
            "total_steps": "Total training steps",
            "total_steps_help": "Total simulation steps for training. Higher = better policy but longer runtime.",
            "policy_type": "Policy type",
            "policy_help": "MlpPolicy: MLP neural network (recommended for tabular data). CnnPolicy: CNN (for images).",
            "hyperparams": "### 📊 PPO hyperparameters",
            "hyper_desc": """
        <div style="background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
            <strong>PPO PRINCIPLE:</strong> Optimizes the policy while limiting large updates via clipping.
            This stabilizes learning and avoids performance collapse.
            <br><strong>Goal:</strong> Maximize cumulative rewards while staying stable.
        </div>
        """,
            "adv_params": "Advanced hyperparameters",
            "adv_desc": """
            **HYPERPARAMETER EXPLANATIONS:**

            - **n_steps**: Number of steps collected before each policy update.
              Higher = more data per update, but slower.

            - **batch_size**: Mini-batch size for optimization.
              Higher = more stable gradients, but more memory.

            - **learning_rate**: Optimizer learning rate.
              Higher = faster learning but less stable.

            - **gamma**: Discount factor (0-1). Closer to 1 = future rewards matter more.
              For long horizons (seasons), use 0.99.

            - **gae_lambda**: GAE (Generalized Advantage Estimation) parameter.
              Blends advantages across multiple horizons. 0.95 is a good default.

            - **clip_range**: Clipping range to limit policy updates.
              Prevents overly large updates that can hurt performance.

            - **ent_coef**: Entropy coefficient to encourage exploration.
              Higher = more exploration, but can slow convergence.

            - **vf_coef**: Weight of the value function loss (critic).
              Balances learning the policy and estimating the value.
            """,
        },
    }[language]

    st.markdown(t2["ppo_params"])
    st.markdown(t2["ppo_desc"], unsafe_allow_html=True)

    ppo_defaults = get_default_ppo_hyperparams_ui()
    ppo_ranges = get_ppo_hyperparams_ui_ranges()

    col1, col2 = st.columns(2)
    with col1:
        total_timesteps = st.number_input(
            t2["total_steps"],
            min_value=ppo_ranges["total_timesteps"]["min"],
            max_value=ppo_ranges["total_timesteps"]["max"],
            value=ppo_defaults["total_timesteps"],
            step=ppo_ranges["total_timesteps"]["step"],
            help=t2["total_steps_help"],
        )
    with col2:
        policy_type = st.selectbox(
            t2["policy_type"],
            options=["MlpPolicy", "CnnPolicy"],
            index=0,
            help=t2["policy_help"],
        )

    st.markdown(t2["hyperparams"])
    st.markdown(t2["hyper_desc"], unsafe_allow_html=True)

    n_steps = ppo_defaults["n_steps"]
    batch_size = ppo_defaults["batch_size"]
    learning_rate = ppo_defaults["learning_rate"]
    gamma = ppo_defaults["gamma"]
    gae_lambda = ppo_defaults["gae_lambda"]
    clip_range = ppo_defaults["clip_range"]
    ent_coef = ppo_defaults["ent_coef"]
    vf_coef = ppo_defaults["vf_coef"]

    with st.expander(t2["adv_params"], expanded=False):
        # Texte d'explication en fonction de la langue
        if language == "fr":
            st.markdown(
                """
                **EXPLICATION DES HYPERPARAMÈTRES :**

                - **n_steps** : Nombre de pas collectés avant chaque mise à jour de la politique.
                  Plus élevé = plus de données par update, mais plus lent.

                - **batch_size** : Taille des mini-batches pour l'optimisation.
                  Plus élevé = gradients plus stables, mais plus de mémoire.

                - **learning_rate** : Taux d'apprentissage pour l'optimiseur.
                  Plus élevé = apprentissage plus rapide mais moins stable.

                - **gamma** : Facteur de discount (0-1). Plus proche de 1 = récompenses futures plus importantes.
                  Pour horizons longs (saisons), utiliser 0.99.

                - **gae_lambda** : Paramètre GAE (Generalized Advantage Estimation).
                  Combine les avantages à plusieurs horizons temporels. 0.95 est une bonne valeur par défaut.

                - **clip_range** : Plage de clipping pour limiter les changements de politique.
                  Empêche les mises à jour trop importantes qui pourraient dégrader la performance.

                - **ent_coef** : Coefficient d'entropie pour encourager l'exploration.
                  Plus élevé = plus d'exploration, mais peut ralentir la convergence.

                - **vf_coef** : Poids de la perte de la fonction valeur (critique).
                  Balance entre l'apprentissage de la politique et l'estimation de la valeur.
                """
            )
        else:
            st.markdown(t2["adv_desc"])

        n_steps = st.number_input(
            "n_steps",
            min_value=ppo_ranges["n_steps"]["min"],
            max_value=ppo_ranges["n_steps"]["max"],
            value=ppo_defaults["n_steps"],
            step=ppo_ranges["n_steps"]["step"],
            help="Nombre de pas collectés avant chaque update (rollout length)",
        )

        batch_size = st.number_input(
            "batch_size",
            min_value=ppo_ranges["batch_size"]["min"],
            max_value=ppo_ranges["batch_size"]["max"],
            value=ppo_defaults["batch_size"],
            step=ppo_ranges["batch_size"]["step"],
            help="Taille des mini-batches pour l'optimisation",
        )

        learning_rate = st.number_input(
            "learning_rate",
            min_value=ppo_ranges["learning_rate"]["min"],
            max_value=ppo_ranges["learning_rate"]["max"],
            value=ppo_defaults["learning_rate"],
            step=ppo_ranges["learning_rate"]["step"],
            format="%.5f",
            help="Taux d'apprentissage (Adam optimizer)",
        )

        gamma = st.slider(
            "gamma (discount factor)",
            min_value=ppo_ranges["gamma"]["min"],
            max_value=ppo_ranges["gamma"]["max"],
            value=ppo_defaults["gamma"],
            step=ppo_ranges["gamma"]["step"],
            help="Facteur de discount : importance des récompenses futures. 0.99 = très important (horizons longs).",
        )

        gae_lambda = st.slider(
            "gae_lambda",
            min_value=ppo_ranges["gae_lambda"]["min"],
            max_value=ppo_ranges["gae_lambda"]["max"],
            value=ppo_defaults["gae_lambda"],
            step=ppo_ranges["gae_lambda"]["step"],
            help="Paramètre GAE : combine avantages à plusieurs horizons. 0.95 = bon compromis.",
        )

        clip_range = st.slider(
            "clip_range",
            min_value=ppo_ranges["clip_range"]["min"],
            max_value=ppo_ranges["clip_range"]["max"],
            value=ppo_defaults["clip_range"],
            step=ppo_ranges["clip_range"]["step"],
            help="Plage de clipping : limite les changements de politique à ±20% (stabilise l'apprentissage).",
        )

        ent_coef = st.number_input(
            "ent_coef (exploration)",
            min_value=ppo_ranges["ent_coef"]["min"],
            max_value=ppo_ranges["ent_coef"]["max"],
            value=ppo_defaults["ent_coef"],
            step=ppo_ranges["ent_coef"]["step"],
            format="%.3f",
            help="Coefficient d'entropie : encourage l'exploration. 0 = pas d'exploration bonus (déterministe).",
        )

        vf_coef = st.slider(
            "vf_coef (valeur)",
            min_value=ppo_ranges["vf_coef"]["min"],
            max_value=ppo_ranges["vf_coef"]["max"],
            value=ppo_defaults["vf_coef"],
            step=ppo_ranges["vf_coef"]["step"],
            help="Coefficient de la perte valeur : poids de l'apprentissage du critique (fonction valeur).",
        )

    ppo_kwargs = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "verbose": 1,
    }

    return int(total_timesteps), policy_type, ppo_kwargs
