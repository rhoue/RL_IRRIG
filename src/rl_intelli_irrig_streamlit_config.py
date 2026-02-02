"""
Application Streamlit pour la configuration et l'entraînement d'un agent RL d'irrigation intelligente.

Cette application permet de :
- Configurer les paramètres du sol (PhysicalBucket)
- Configurer les paramètres météorologiques (pluie, ET0, Kc)
- Configurer l'entraînement PPO (nombre de pas, politique, etc.)
- Configurer les hyperparamètres PPO (learning rate, gamma, etc.)
- Visualiser les résultats d'entraînement et d'évaluation
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional
import sys
import os

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports pour l'environnement et l'entraînement
PPO_AVAILABLE = False
BaseCallback = None
Wrapper = None
gym = None
spaces = None
Monitor = None
DummyVecEnv = None
PPO = None

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    from gymnasium import Wrapper  # type: ignore
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.monitor import Monitor  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
    PPO_AVAILABLE = True
except ImportError as e:
    # Garder les variables définies pour éviter les erreurs NameError
    import sys
    import traceback
    # Afficher l'erreur dans les logs pour le débogage
    print(f"Warning: Import error for RL libraries: {e}", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Python path: {sys.path[:3]}", file=sys.stderr)  # Afficher les 3 premiers chemins
    traceback.print_exc(file=sys.stderr)

# Imports pour PyTorch et Neural ODE
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    from torch.utils.data import Dataset, DataLoader  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    optim = None
    Dataset = None
    DataLoader = None


# Configuration de la page
st.set_page_config(
    page_title="Configuration RL Irrigation Intelligente",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS DES UTILITAIRES
# ============================================================================

# ----------------------------------------------------------------------------
# UTILITAIRES INTERFACE UTILISATEUR (utils_ui.py)
# ----------------------------------------------------------------------------
# Fonctions et classes pour l'interface Streamlit :
# - apply_custom_css() : Applique le CSS personnalisé (largeur maximale, styles)
# - format_time() : Formate les durées en format lisible (h, min, s)
# - display_info_box() : Affiche un encadré d'information stylisé
# - display_warning_box() : Affiche un encadré d'avertissement stylisé
# - display_info_small() : Affiche un petit message d'information
# - create_progress_ui() : Crée l'interface de progression (barre, statut, ETA)
# - create_training_callbacks() : Crée les callbacks PPO pour suivre l'entraînement
# - finalize_training_ui() : Finalise l'affichage après l'entraînement
# - display_training_metrics() : Affiche les métriques d'entraînement (tableaux, graphiques)
# - ProgressCallback : Callback PPO pour mettre à jour la barre de progression Streamlit
# - MetricsCallback : Callback PPO pour collecter les métriques d'entraînement
from src.utils_ui import (
    apply_custom_css, format_time, display_info_box, display_warning_box,
    display_info_small, create_progress_ui, create_training_callbacks,
    finalize_training_ui, display_training_metrics, ProgressCallback, MetricsCallback
)

# ----------------------------------------------------------------------------
# UTILITAIRES VISUALISATION (utils_plot.py)
# ----------------------------------------------------------------------------
# Fonctions pour générer les graphiques matplotlib/seaborn :
# - configure_matplotlib() : Configure matplotlib (taille, style, backend)
# - plot_episode_rollout() : Trace les graphiques d'un épisode (tension, réserve, irrigation, pluie, flux)
#   * 4 panneaux : tension ψ, réserve S, irrigation/pluie, flux (ETc, drainage)
#   * Utilisé pour visualiser les résultats des scénarios 2, 3, 4, 5, 6
# - plot_scenario1() : Trace les graphiques du scénario 1 (règles simples)
#   * Même structure que plot_episode_rollout mais adapté aux résultats de simulate_scenario1
from src.utils_plot import (
    configure_matplotlib, plot_episode_rollout, plot_scenario1
)
from src.data_loader import load_data_for_simulation
from src.validation.era5_land_checks import summarize_era5_land_validation
from src.validation.reporting import (
    format_validation_table,
    rollout_to_sim_outputs,
)
from src.data_loader import load_data_for_simulation
from src.validation.reporting import (
    format_validation_table,
    rollout_to_sim_outputs,
)

# ----------------------------------------------------------------------------
# UTILITAIRES SCÉNARIO 3 : NEURAL ODE (utils_neuro_ode.py)
# ----------------------------------------------------------------------------
# Classes et fonctions pour le scénario 3 (modèle hybride Physique + Neural ODE) :
# - ResidualODEModel : Modèle PyTorch qui apprend une correction Δψ à partir de [ψ_t, I_t, R_t, ET0_t]
#   * Architecture : MLP avec 2 couches cachées (64 neurones, activation Tanh)
#   * Principe : ψ_{t+1} = ψ_{t+1}^{physique} + Δψ (correction neuronale)
#   * Utilisé dans IrrigationEnvPhysical (utils_env_modeles.py) pour corriger la prédiction physique
# - ResidualODEDataset : Dataset PyTorch pour pré-entraîner le modèle résiduel
#   * Génère des trajectoires simulées avec le modèle physique
#   * Crée des paires (X_t, Δψ_t) où Δψ_t est la différence entre prédiction physique et observation
# - pretrain_residual_ode() : Pré-entraîne le modèle résiduel sur des données simulées
#   * Génère N_traj trajectoires, calcule les corrections, entraîne avec MSE loss
#   * Retourne le modèle pré-entraîné prêt à être utilisé dans l'environnement RL
# - train_ppo_hybrid_ode() : Entraîne un agent PPO sur l'environnement hybride (physique + Neural ODE)
#   * Crée IrrigationEnvPhysical avec residual_ode=modèle pré-entraîné
#   * Entraîne PPO avec callbacks pour suivi de progression
#   * Retourne le modèle PPO entraîné et les métriques d'entraînement
from src.utils_neuro_ode import (
    ResidualODEModel, ResidualODEDataset, pretrain_residual_ode, train_ppo_hybrid_ode
)
# Variante continue du Neural ODE (intégration continue via torchdiffeq ou Euler)
from src.utils_neuro_ode_cont import (
    ContinuousResidualODE, ContinuousResidualODEDataset,
    pretrain_continuous_residual_ode, train_ppo_hybrid_ode_cont
)
from src.utils_ui_ai import render_ai_assistant_sidebar

# ----------------------------------------------------------------------------
# UTILITAIRES MODÈLE PHYSIQUE (utils_physical_model.py)
# ----------------------------------------------------------------------------
# Classes et fonctions pour le modèle physique et les règles d'irrigation :
# - PhysicalBucket : Modèle physique de type "bucket" pour le bilan hydrique du sol
#   * Représente le sol comme un réservoir d'eau dans la zone racinaire
#   * Variables : S (réserve en mm), ψ (tension en cbar), relation S ↔ ψ via courbe de rétention
#   * Bilan hydrique : S_{t+1} = S_t + η_I × I_t + R_t - ETc_t - D(S_t)
#   * Méthodes : S_to_psi(), psi_to_S(), f_ET(), drainage()
#   * Utilisé par tous les scénarios comme base physique
# - rule_seuil_unique() : Règle d'irrigation à seuil unique (Scénario 1)
#   * Irrigue si ψ_t > threshold_cbar avec une dose fixe
#   * Peut réduire l'irrigation si pluie prévue > seuil
# - rule_bande_confort() : Règle d'irrigation à bande de confort (Scénario 1)
#   * Irrigue uniquement si ψ_t sort de la bande [psi_low, psi_high]
#   * Dose fixe quand on sort par le haut
# - rule_proportionnelle() : Règle d'irrigation proportionnelle (Scénario 1)
#   * Dose proportionnelle à l'écart : I_t = k_I × max(0, ψ_t - ψ_target)
#   * Plus le sol est sec, plus on irrigue
# - simulate_scenario1() : Simule le scénario 1 (modèle physique + règle simple)
#   * Exécute une saison complète en appliquant une règle d'irrigation fixe
#   * Retourne un dictionnaire avec les historiques (psi, S, I, rain, ETc, D, etc.)
# - make_env() : Fabrique une fonction d'initialisation d'environnement Gymnasium avec Monitor
#   * Utilise IrrigationEnvPhysical de utils_env_gymnasium.py (scénario 2)
#   * Supporte weather_params pour personnaliser la génération météo
#   * Retourne une fonction _init() qui crée l'environnement avec Monitor
#   * Utilisé pour l'entraînement PPO du scénario 2
# - evaluate_episode() : Évalue un modèle PPO en exécutant un épisode complet
#   * Supporte tous les scénarios (2, 3, 4, 5) avec ou sans modèles résiduels
#   * Mode déterministe (pas d'exploration)
#   * Retourne un dictionnaire avec les historiques complets de l'épisode
from src.utils_physical_model import (
    PhysicalBucket, rule_seuil_unique, rule_bande_confort, rule_proportionnelle,
    simulate_scenario1, make_env, evaluate_episode
)
from src.utils_physics_config import (
    DEFAULT_MAX_IRRIGATION,
    DEFAULT_SEASON_LENGTH,
    DEFAULT_SEED,
    get_default_soil_config,
    get_default_weather_config,
    get_rule_bande_confort_config,
    get_rule_bande_confort_ranges,
    get_rule_proportionnelle_config,
    get_rule_proportionnelle_ranges,
    get_rule_seuil_unique_config,
    get_rule_seuil_unique_ranges,
)

# ----------------------------------------------------------------------------
# UTILITAIRES MÉTÉOROLOGIE (utils_weather.py)
# ----------------------------------------------------------------------------
# Fonction pour générer les données météorologiques :
# - generate_weather() : Génère des séries temporelles de pluie, ET0 et Kc pour une saison
#   * Pluie : distribution gamma avec événements intermittents (beaucoup de jours secs, quelques jours pluvieux)
#   * ET0 : distribution normale avec variation saisonnière (base + amplitude + bruit)
#   * Kc : coefficient cultural variant selon les phases de croissance (initial, développement, mi-saison, fin)
#   * Paramètres configurables : probabilités de pluie par période, amplitudes, etc.
#   * Déterministe pour une graine donnée (reproductibilité)
#   * Utilisé par tous les scénarios pour générer les conditions météorologiques
from src.utils_weather import generate_weather

# ----------------------------------------------------------------------------
# ENVIRONNEMENT GYMNASIUM SCÉNARIO 2 (utils_env_gymnasium.py)
# ----------------------------------------------------------------------------
# Classe d'environnement Gymnasium pour le scénario 2 (RL sur modèle physique simple) :
# - IrrigationEnvPhysical : Environnement Gymnasium pour l'apprentissage par renforcement
#   * Observation : o_t = [ψ_t, rain_t, et0_t, Kc_t] (4 dimensions)
#   * Action : a_t = I_t ∈ [0, max_irrigation] (continue, 1 dimension)
#   * Transition : S_{t+1} = f_physique(S_t, I_t, rain_t, ETc_t, D_t) via PhysicalBucket
#   * Récompense : r_t = -|ψ_t - clip(ψ_t, 20, 60)| / 10.0 - 0.05 × I_t
#   * Supporte weather_params pour personnaliser la génération météo
#   * Utilise PhysicalBucket et generate_weather (modularité)
#   * Diffère de utils_env_modeles.py : version simple sans modèles résiduels, supporte weather_params
from src.utils_env_gymnasium import IrrigationEnvPhysical

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#        comme encodeur pour mapper les observations vers l'espace latent
# - LatentTransitionODE : Modèle Neural ODE pour prédire les transitions dans l'espace latent
#   * Architecture : MLP avec 2 couches cachées (64 neurones, Tanh)
#   * Utilisé pour prédire les transitions futures dans l'espace latent (planning)
#   * Entraîne le modèle de transition pour prédire z_{t+1} à partir de z_t, a_t, inputs_exog
# ----------------------------------------------------------------------------
# UI CONFIGURATION SECTIONS (modularized)
# ----------------------------------------------------------------------------
from src.ui_config_sections import (
    render_environment_config,
    render_mlp_policy_config,
    render_ppo_training_section,
    render_soil_and_tension_config,
    render_weather_config,
)

# Configuration matplotlib pour des graphiques plus larges
configure_matplotlib()

# Style
apply_custom_css()


def _get_weather_source_cfg():
    """
    Retourne (data_source, data_path, era5_land_cfg) en fonction de la sélection UI.
    """
    source = st.session_state.get("weather_source", "synthetic")
    path = st.session_state.get("era5_path", "")
    freq = st.session_state.get("era5_freq", "1D")
    if source == "era5_land" and path:
        return "era5_land", path, {"use_era5_land": True, "data_path": path, "resample_freq": freq}
    return "synthetic", None, None



# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================
def main():
    """
    Point d'entrée de l'application Streamlit pour configurer, entraîner et évaluer
    les scénarios d'irrigation intelligente.

    SCÉNARIOS COUVERTS :
    - Scénario 1 : Modèle physique + règles simples (seuil, bande, proportionnelle)
    - Scénario 2 : PPO sur environnement physique
    - Scénario 3 : PPO hybride avec Neural ODE résiduel
    - Scénario 3b : PPO hybride avec Neural ODE continu

    ORGANISATION UI :
    - Sidebar : configuration sol/météo/hyperparamètres et options d'entraînement
    - Onglets : un par scénario + sections Évaluation et Visualisation
    - État de session : conserve modèles, historiques et rollouts entre reruns

    PARCOURS TYPE :
    1) Régler les paramètres dans la sidebar
    2) Lancer les scénarios souhaités (simulation règles ou entraînement PPO)
    3) Évaluer les modèles sur une saison cible
    4) Comparer les métriques et visualisations dans les onglets dédiés
    """
    # Initialisation de l'état de session Streamlit
    # Streamlit conserve ces variables entre les reruns de l'application
    if "ppo_model" not in st.session_state:
        st.session_state.ppo_model = None  # Modèle PPO entraîné (scénario 2)
    if "training_history" not in st.session_state:
        st.session_state.training_history = None  # Historique d'entraînement
    if "scenario1_result" not in st.session_state:
        st.session_state.scenario1_result = None  # Résultats du scénario 1
    if "scenario2_rollout" not in st.session_state:
        st.session_state.scenario2_rollout = None  # Résultats de l'évaluation scénario 2
    if "scenario3_rollout" not in st.session_state:
        st.session_state.scenario3_rollout = None  # Résultats de l'évaluation scénario 3
    if "scenario3b_rollout" not in st.session_state:
        st.session_state.scenario3b_rollout = None  # Résultats de l'évaluation scénario 3b (Neural ODE continu)

    # Sélecteur de langue global (zone principale) pour toutes les interfaces
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "en"
    lang_choice_global = st.selectbox(
        "Langue / Language",
        options=["Français", "English"],
        index=0 if st.session_state.ui_language == "fr" else 1,
        key="ui_language_global",
    )
    st.session_state.ui_language = "en" if lang_choice_global == "English" else "fr"
    current_lang = st.session_state.ui_language

    # Titre principal basé sur la langue sélectionnée
    st.title(
        "💧 Irrigation Intelligente avec Apprentissage par Renforcement"
        if current_lang == "fr"
        else "💧 Smart Irrigation with Reinforcement Learning"
    )

    # Rétrocompatibilité : garder evaluation_rollout pour les anciens codes
    if "evaluation_rollout" not in st.session_state:
        st.session_state.evaluation_rollout = None
    
    # ========================================================================
    # SIDEBAR : CONFIGURATION DES PARAMÈTRES
    # ========================================================================
    
    with st.sidebar:
        # Display logo in sidebar header
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "logo.jpg")
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, width='stretch')
        else:
            # Fallback: try relative path from src directory
            logo_path = os.path.join("images", "logo_rv.svg")
            if os.path.exists(logo_path):
                st.sidebar.image(logo_path, width='stretch')

        st.markdown("&nbsp;")  # petit espace entre le logo et le contenu

        current_lang = st.session_state.get("ui_language", "fr")

        render_ai_assistant_sidebar(current_lang)

        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        render_soil_and_tension_config(language=current_lang)
        render_weather_config(language=current_lang)
        render_environment_config(language=current_lang)

    # Valeurs d'environnement disponibles pour le reste de l'application
    season_length = st.session_state.get("season_length", DEFAULT_SEASON_LENGTH)
    max_irrigation = st.session_state.get("max_irrigation", DEFAULT_MAX_IRRIGATION)
    seed = st.session_state.get("seed", DEFAULT_SEED)
    
    # ========================================================================
    # CORPS PRINCIPAL : SIMULATION ET ENTRAÎNEMENT
    # ========================================================================
    
    st.divider()
    current_lang = st.session_state.get("ui_language", "fr")
    tr = lambda fr, en: en if current_lang == "en" else fr
    st.markdown(tr("### 🎮 Simulation et Entraînement", "### 🎮 Simulation and Training"))
    
    # Onglets pour organiser l'interface (bilingue)
    tab_labels = [
        tr("🌱 Scénario 1 (Règles simples)", "🌱 Scenario 1 (Simple rules)"),
        tr("🎓 Scénario 2 (Entraînement PPO)", "🎓 Scenario 2 (PPO training)"),
        tr("🔬 Scénario 3 (Neural ODE)", "🔬 Scenario 3 (Neural ODE)"),
        tr("🧠 Scénario 3b (Neural ODE continu)", "🧠 Scenario 3b (Continuous Neural ODE)"),
        tr("📈 Évaluation", "📈 Evaluation"),
        tr("📊 Visualisation", "📊 Visualization"),
        tr("⚖️ Comparaison", "⚖️ Comparison"),
        tr("🧪 Validation ERA5-Land (avancée)", "🧪 Advanced ERA5-Land validation"),
    ]
    tab1, tab2, tab3, tab3b, tab5, tab6, tab7, tab8 = st.tabs(tab_labels)
    
    # ========================================================================
    # ONGLET 1 : SCÉNARIO 1 - MODÈLE PHYSIQUE + RÈGLES SIMPLES
    # ========================================================================
    with tab1:
        language = st.session_state.get("ui_language", "fr")
        t = {
            "fr": {
                "header": "🌱 Scénario 1 — Règles simples",
                "desc": "Le **Scénario 1** utilise un modèle physique (bucket) pour simuler le bilan hydrique du sol et applique des règles d'irrigation simples basées sur des seuils de tension matricielle ψ.",
                "config": "### ⚙️ Configuration de la règle d'irrigation",
                "rule_label": "Type de règle d'irrigation",
                "rule_help": "Choisissez le type de règle d'irrigation à appliquer",
                "options": {
                    "threshold": "Seuil unique",
                    "band": "Bande de confort",
                    "prop": "Proportionnelle",
                },
                "threshold_label": "Seuil de tension (cbar)",
                "threshold_help": "Si ψ dépasse ce seuil, on irrigue",
                "dose_label": "Dose d'irrigation (mm)",
                "rain_label": "Seuil pluie prévue (mm)",
                "rain_help": "Si pluie prévue > seuil, réduire l'irrigation",
                "reduce_label": "Facteur de réduction si pluie",
                "reduce_help": "Facteur de réduction de l'irrigation si pluie imminente",
                "psi_low": "ψ bas (cbar)",
                "psi_low_help": "Limite basse de la bande de confort",
                "psi_high": "ψ haut (cbar)",
                "psi_high_help": "Limite haute de la bande de confort",
                "psi_target": "ψ cible (cbar)",
                "psi_target_help": "Tension cible à maintenir",
                "k_i": "Coefficient k_I",
                "k_i_help": "Coefficient de proportionnalité",
                "info_prop": "Irrigation = k_I × (ψ - ψ_cible) si ψ > ψ_cible",
                "simulate": "🌱 Simuler le scénario 1",
            },
            "en": {
                "header": "🌱 Scenario 1 — Simple rules",
                "desc": "Scenario 1 uses a physical bucket model to simulate the soil water balance and applies simple irrigation rules based on matric tension thresholds ψ.",
                "config": "### ⚙️ Irrigation rule configuration",
                "rule_label": "Irrigation rule type",
                "rule_help": "Choose the irrigation rule to apply",
                "options": {
                    "threshold": "Single threshold",
                    "band": "Comfort band",
                    "prop": "Proportional",
                },
                "threshold_label": "Tension threshold (cbar)",
                "threshold_help": "If ψ exceeds this threshold, irrigate",
                "dose_label": "Irrigation dose (mm)",
                "rain_label": "Forecast rain threshold (mm)",
                "rain_help": "If forecast rain > threshold, reduce irrigation",
                "reduce_label": "Reduction factor if rain",
                "reduce_help": "Reduction of irrigation when rain is imminent",
                "psi_low": "ψ low (cbar)",
                "psi_low_help": "Lower bound of comfort band",
                "psi_high": "ψ high (cbar)",
                "psi_high_help": "Upper bound of comfort band",
                "psi_target": "ψ target (cbar)",
                "psi_target_help": "Target tension to maintain",
                "k_i": "Coefficient k_I",
                "k_i_help": "Proportionality coefficient",
                "info_prop": "Irrigation = k_I × (ψ - ψ_target) if ψ > ψ_target",
                "simulate": "🌱 Run Scenario 1",
            },
        }[language]

        # Récupération des paramètres depuis la sidebar (avec fallback par défaut)
        soil_params = st.session_state.get("soil_params") or get_default_soil_config()
        weather_params = st.session_state.get("weather_params") or get_default_weather_config()
        season_length = st.session_state.get("season_length", DEFAULT_SEASON_LENGTH)
        max_irrigation = st.session_state.get("max_irrigation", DEFAULT_MAX_IRRIGATION)
        seed = st.session_state.get("seed", DEFAULT_SEED)

        st.markdown(f'<h2 class="section-header">{t["header"]}</h2>', unsafe_allow_html=True)
        st.markdown(t["desc"])
        st.markdown(t["config"])
        
        rule_options = [
            ("threshold", t["options"]["threshold"]),
            ("band", t["options"]["band"]),
            ("prop", t["options"]["prop"]),
        ]
        rule_label_to_key = {label: key for key, label in rule_options}
        selected_rule_label = st.selectbox(
            t["rule_label"],
            options=[label for _, label in rule_options],
            index=0,
            help=t["rule_help"]
        )
        rule_type = rule_label_to_key[selected_rule_label]
        
        # Paramètres selon le type de règle
        rule_kwargs = {}
        
        if rule_type == "threshold":
            rule_defaults = get_rule_seuil_unique_config()
            rule_ranges = get_rule_seuil_unique_ranges()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                threshold_cbar = st.number_input(
                    t["threshold_label"],
                    min_value=rule_ranges["threshold_cbar"]["min"],
                    max_value=rule_ranges["threshold_cbar"]["max"],
                    value=rule_defaults["threshold_cbar"],
                    step=rule_ranges["threshold_cbar"]["step"],
                    help=t["threshold_help"],
                    key="rule_threshold_cbar",
                )
            with col2:
                dose_mm = st.number_input(
                    t["dose_label"],
                    min_value=rule_ranges["dose_mm"]["min"],
                    max_value=min(max_irrigation, rule_ranges["dose_mm"]["max"]),
                    value=min(rule_defaults["dose_mm"], max_irrigation),
                    step=rule_ranges["dose_mm"]["step"]
                )
            with col3:
                rain_threshold_mm = st.number_input(
                    t["rain_label"],
                    min_value=rule_ranges["rain_threshold_mm"]["min"],
                    max_value=rule_ranges["rain_threshold_mm"]["max"],
                    value=rule_defaults["rain_threshold_mm"],
                    step=rule_ranges["rain_threshold_mm"]["step"],
                    help=t["rain_help"]
                )
            
            reduce_factor = st.slider(
                t["reduce_label"],
                min_value=rule_ranges["reduce_factor"]["min"],
                max_value=rule_ranges["reduce_factor"]["max"],
                value=rule_defaults["reduce_factor"],
                step=rule_ranges["reduce_factor"]["step"],
                help=t["reduce_help"]
            )
            
            rule_kwargs = {
                "threshold_cbar": threshold_cbar,
                "dose_mm": dose_mm,
                "rain_threshold_mm": rain_threshold_mm,
                "reduce_factor": reduce_factor
            }
            rule_fn = rule_seuil_unique
            
        elif rule_type == "band":
            rule_defaults = get_rule_bande_confort_config()
            rule_ranges = get_rule_bande_confort_ranges()
            col1, col2, col3 = st.columns(3)
            with col1:
                psi_low = st.number_input(
                    t["psi_low"],
                    min_value=rule_ranges["psi_low"]["min"],
                    max_value=rule_ranges["psi_low"]["max"],
                    value=rule_defaults["psi_low"],
                    step=rule_ranges["psi_low"]["step"],
                    help=t["psi_low_help"]
                )
            with col2:
                psi_high = st.number_input(
                    t["psi_high"],
                    min_value=rule_ranges["psi_high"]["min"],
                    max_value=rule_ranges["psi_high"]["max"],
                    value=rule_defaults["psi_high"],
                    step=rule_ranges["psi_high"]["step"],
                    help=t["psi_high_help"]
                )
            with col3:
                dose_mm = st.number_input(
                    t["dose_label"],
                    min_value=rule_ranges["dose_mm"]["min"],
                    max_value=min(max_irrigation, rule_ranges["dose_mm"]["max"]),
                    value=min(rule_defaults["dose_mm"], max_irrigation),
                    step=rule_ranges["dose_mm"]["step"],
                )
            
            rule_kwargs = {
                "psi_low": psi_low,
                "psi_high": psi_high,
                "dose_mm": dose_mm
            }
            rule_fn = rule_bande_confort
            
        else:  # Proportionnelle
            rule_defaults = get_rule_proportionnelle_config()
            rule_ranges = get_rule_proportionnelle_ranges()
            col1, col2, col3 = st.columns(3)
            with col1:
                psi_target = st.number_input(
                    t["psi_target"],
                    min_value=rule_ranges["psi_target"]["min"],
                    max_value=rule_ranges["psi_target"]["max"],
                    value=rule_defaults["psi_target"],
                    step=rule_ranges["psi_target"]["step"],
                    help=t["psi_target_help"]
                )
            with col2:
                k_I = st.number_input(
                    t["k_i"],
                    min_value=rule_ranges["k_I"]["min"],
                    max_value=rule_ranges["k_I"]["max"],
                    value=rule_defaults["k_I"],
                    step=rule_ranges["k_I"]["step"],
                    format="%.2f",
                    help=t["k_i_help"]
                )
            with col3:
                st.info(t["info_prop"])
            
            rule_kwargs = {
                "psi_target": psi_target,
                "k_I": k_I
            }
            rule_fn = rule_proportionnelle
        
        # Bouton de simulation
        if st.button(t["simulate"], type="primary"):
            with st.spinner("Simulation en cours..." if language == "fr" else "Simulation running..."):
                try:
                    # Création du modèle de sol avec les paramètres configurés
                    soil = PhysicalBucket(**soil_params)
                    
                    # Charger ERA5-Land si sélectionné
                    data_source, data_path, era5_land_cfg = _get_weather_source_cfg()
                    external_weather = None
                    effective_T = season_length
                    if data_source == "era5_land" and data_path:
                        try:
                            external_weather = load_data_for_simulation(
                                data_source="era5_land",
                                file_path=data_path,
                                resample_freq=era5_land_cfg.get("resample_freq", "1D") if era5_land_cfg else "1D",
                            )
                            if external_weather and "rain" in external_weather:
                                effective_T = len(np.asarray(external_weather["rain"]))
                        except Exception as exc:
                            st.warning(f"⚠️ Impossible de charger ERA5-Land, utilisation météo synthétique : {exc}")
                            external_weather = None
                    
                    # Simulation
                    sim_result = simulate_scenario1(
                        T=effective_T,
                        seed=seed,
                        I_max=max_irrigation,
                        soil=soil,
                        rule_fn=rule_fn,
                        rule_kwargs=rule_kwargs,
                        weather_params=weather_params,
                        external_weather=external_weather,
                    )
                    
                    # Sauvegarde des résultats
                    st.session_state.scenario1_result = sim_result
                    # Expose a rollout-compatible dict so validation tab can pick it up
                    et0_arr = sim_result.get("ET0")
                    if et0_arr is None:
                        et0_arr = sim_result.get("et0")
                    st.session_state.scenario1_rollout = {
                        "R": sim_result.get("rain"),
                        "et0": et0_arr,
                        "ETc": sim_result.get("ETc"),
                        "psi": sim_result.get("psi"),
                        "S": sim_result.get("S"),
                        "fluxes": {"runoff": sim_result.get("D")},
                        # provide a bucket soil moisture proxy
                        "soil_moisture_layers": {"bucket_total_mm": sim_result.get("S")},
                    }
                    st.success("✅ Simulation terminée !" if language == "fr" else "✅ Simulation complete!")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la simulation: {str(e)}" if language == "fr" else f"❌ Error during simulation: {str(e)}")
        
        # Affichage des métriques (toujours visible si résultats disponibles)
        if "scenario1_result" in st.session_state and st.session_state.scenario1_result is not None:
            sim_result = st.session_state.scenario1_result
            
            metrics_title = {
                "fr": "### 📊 Métriques de la simulation",
                "en": "### 📊 Simulation metrics",
            }[language]
            st.markdown(metrics_title)
            
            # Utiliser un layout 2x2 pour plus d'espace
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            with col1:
                st.metric(
                    "💧 Eau totale irriguée" if language == "fr" else "💧 Total irrigation",
                    f"{sim_result['I'].sum():.1f} mm",
                    help="Quantité totale d'eau d'irrigation appliquée sur toute la saison" if language == "fr" else "Total irrigation water applied over the season"
                )
            with col2:
                st.metric(
                    "🌧️ Pluie totale" if language == "fr" else "🌧️ Total rain",
                    f"{sim_result['rain'].sum():.1f} mm",
                    help="Quantité totale de pluie reçue sur toute la saison" if language == "fr" else "Total rainfall received over the season"
                )
            with col3:
                st.metric(
                    "💨 Drainage total" if language == "fr" else "💨 Total drainage",
                    f"{sim_result['D'].sum():.1f} mm",
                    help="Quantité totale d'eau drainée (perdue) sur toute la saison" if language == "fr" else "Total drained (lost) water over the season"
                )
            with col4:
                psi_mean = sim_result['psi'].mean()
                psi_min = sim_result['psi'].min()
                psi_max = sim_result['psi'].max()
                st.metric(
                    "📊 Tension moyenne" if language == "fr" else "📊 Average tension",
                    f"{psi_mean:.1f} cbar",
                    delta=(f"Min: {psi_min:.1f} | Max: {psi_max:.1f}" if language == "fr" else f"Min: {psi_min:.1f} | Max: {psi_max:.1f}"),
                    help="Tension matricielle moyenne, avec min et max" if language == "fr" else "Average matric tension with min and max"
                )
            
            # Métriques supplémentaires en une ligne
            st.markdown("#### 📈 Métriques supplémentaires" if language == "fr" else "#### 📈 Additional metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_water = sim_result['I'].sum() + sim_result['rain'].sum()
                st.metric("💦 Eau totale (I+R)" if language == "fr" else "💦 Total water (I+R)", f"{total_water:.1f} mm")
            with col2:
                et0_total = sim_result['ET0'].sum()
                st.metric("☀️ ET0 totale" if language == "fr" else "☀️ ET0 total", f"{et0_total:.1f} mm")
            with col3:
                etc_total = sim_result['ETc'].sum()
                st.metric("🌱 ETc totale" if language == "fr" else "🌱 ETc total", f"{etc_total:.1f} mm")
            with col4:
                water_balance = total_water - etc_total - sim_result['D'].sum()
                st.metric("⚖️ Bilan hydrique" if language == "fr" else "⚖️ Water balance", f"{water_balance:.1f} mm")
            
            st.info("✅ Résultats disponibles. Consultez l'onglet 'Visualisation' pour voir les graphiques détaillés." if language == "fr" else "✅ Results available. Check the 'Visualization' tab for detailed charts.")
        else:
            st.info("👆 Cliquez sur 'Simuler le scénario 1' pour lancer une simulation." if language == "fr" else "👆 Click on 'Run Scenario 1' to start a simulation.")
    
    # ========================================================================
    # ONGLET 2 : SCÉNARIO 2 - ENTRAÎNEMENT PPO
    # ========================================================================
    with tab2:
        language = st.session_state.get("ui_language", "fr")

        t2 = {
            "fr": {
                "header": "🎓 Scénario 2 — Entraînement du modèle PPO",
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
                "header": "🎓 Scenario 2 — PPO training",
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
            },
        }[language]

        st.markdown(f'<h2 class="section-header">{t2["header"]}</h2>', unsafe_allow_html=True)
        
        if not PPO_AVAILABLE:
            import sys
            st.error("⚠️ Les bibliothèques RL ne sont pas disponibles. Vérifiez que `gymnasium` et `stable-baselines3` sont installés dans votre environnement Python.")
            with st.expander("🔍 Détails et solution"):
                st.write(f"**Environnement Python actuel :** `{sys.executable}`")
                st.write(f"**Version Python :** `{sys.version.split()[0]}`")
                
                # Vérifier si on est dans un environnement conda
                conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
                conda_prefix = os.environ.get('CONDA_PREFIX', None)
                if conda_env:
                    st.info(f"✅ Environnement Conda détecté : `{conda_env}`")
                    if conda_prefix:
                        st.write(f"   Chemin : `{conda_prefix}`")
                else:
                    st.warning("⚠️ Aucun environnement Conda détecté")
                
                st.code("""
SOLUTION 1 - Activer l'environnement Conda avant de lancer Streamlit :
    conda activate RL_IntIrrEnv
    streamlit run src/rl_intelli_irrig_streamlit_config.py

SOLUTION 2 - Installer les dépendances dans l'environnement actuel :
    pip install gymnasium stable-baselines3
    
    Ou avec conda :
    conda install -c conda-forge gymnasium stable-baselines3

Note: Assurez-vous que Streamlit utilise le même environnement Python
que celui où les bibliothèques sont installées.
                """)
        else:
            # ====================================================================
            # PARAMÈTRES D'ENTRAÎNEMENT PPO
            # ====================================================================
            total_timesteps, policy_type, ppo_kwargs = render_ppo_training_section(language)
            policy_kwargs = render_mlp_policy_config(policy_type, language, key_prefix="scenario2")
            
            # Bouton d'entraînement
            start_label = {
                "fr": "🚀 Démarrer l'entraînement PPO (Scénario 2)",
                "en": "🚀 Start PPO training (Scenario 2)",
            }[language]
            if st.button(start_label, type="primary", key="train_ppo_btn"):
                with st.spinner("Entraînement en cours..." if language == "fr" else "Training in progress..."):
                    try:
                        # Récupérer les paramètres depuis session_state
                        soil_params = st.session_state.get("soil_params", {})
                        weather_params = st.session_state.get("weather_params", {})
                        season_length = st.session_state.get("season_length", 120)
                        max_irrigation = st.session_state.get("max_irrigation", 20.0)
                        seed = st.session_state.get("seed", 123)
                        data_source, data_path, era5_land_cfg = _get_weather_source_cfg()
                        
                        # Création de l'environnement vectorisé
                        base_env_factory = make_env(
                            seed=seed,
                            season_length=season_length,
                            max_irrigation=max_irrigation,
                            soil_params=soil_params,
                            weather_params=weather_params,
                            weather_shift_cfg=st.session_state.get("proposal_a_config"),
                            data_source=data_source,
                            data_path=data_path,
                            era5_land_cfg=era5_land_cfg,
                        )

                        def _init_env():
                            env = base_env_factory()
                            return env

                        vec_env = DummyVecEnv([_init_env])
                        
                        # Création du modèle PPO
                        model = PPO(
                            policy=policy_type,
                            env=vec_env,
                            seed=seed,
                            policy_kwargs=policy_kwargs if policy_kwargs else None,
                            **ppo_kwargs
                        )
                        
                        # Section de progression 
                        progress_title = {
                            "fr": "### 📊 Progression de l'entraînement",
                            "en": "### 📊 Training progress",
                        }[language]
                        st.markdown(progress_title)
                        
                        # Barre de progression 
                        progress_bar = st.progress(0)
                        
                        # Zone de statut 
                        status_container = st.container()
                        with status_container:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                status_text = st.empty()
                            with col2:
                                time_elapsed = st.empty()
                            with col3:
                                eta_text = st.empty()
                        
                        # Callback personnalisé pour suivre la progression
                        import time
                        start_time = time.time()
                        
                        # Callbacks pour suivre la progression et collecter les métriques
                        callbacks_list = []
                        training_metrics = {}
                        
                        # Callback pour collecter les métriques d'entraînement
                        if BaseCallback is not None:
                            class MetricsCallback(BaseCallback):
                                def __init__(self):
                                    super().__init__()
                                    self.metrics_history = []
                                
                                def _on_step(self) -> bool:
                                    # Collecter les métriques à chaque log
                                    if self.logger is not None:
                                        metrics = {}
                                        if hasattr(self.logger, 'name_to_value'):
                                            for key, value in self.logger.name_to_value.items():
                                                if isinstance(value, (int, float)):
                                                    metrics[key] = value
                                        if metrics:
                                            self.metrics_history.append(metrics)
                                    return True
                                
                                def get_final_metrics(self):
                                    """Retourne les métriques finales (dernières valeurs enregistrées)"""
                                    if not self.metrics_history:
                                        return {}
                                    for metrics in reversed(self.metrics_history):
                                        if metrics:
                                            return metrics
                                    return {}
                            
                            metrics_callback = MetricsCallback()
                            callbacks_list.append(metrics_callback)
                        
                        # Callback personnalisé pour suivre la progression de l'entraînement
                        if BaseCallback is not None:
                            class ProgressCallback(BaseCallback):
                                """
                                Callback pour suivre la progression de l'entraînement dans Streamlit.
                                
                                PRINCIPE :
                                Hérite de BaseCallback de stable-baselines3. Appelé à chaque pas
                                de simulation pour mettre à jour l'interface utilisateur avec :
                                - Barre de progression
                                - Nombre de pas effectués
                                - Temps écoulé et temps restant estimé
                                
                                MÉTHODE _on_step :
                                Appelée automatiquement par stable-baselines3 à chaque pas.
                                Retourne True pour continuer l'entraînement.
                                """
                                def __init__(self, progress_bar, status_text, time_elapsed, eta_text, total_timesteps, start_time, language: str):
                                    super().__init__()
                                    self.progress_bar = progress_bar      # Widget Streamlit pour la barre
                                    self.status_text = status_text        # Widget pour le texte de statut
                                    self.time_elapsed = time_elapsed     # Widget pour le temps écoulé
                                    self.eta_text = eta_text             # Widget pour le temps restant
                                    self.total_timesteps = total_timesteps
                                    self.start_time = start_time
                                    self.last_update = start_time
                                    self.language = language
                                
                                def _on_step(self) -> bool:
                                    """
                                    Appelée à chaque pas de simulation.
                                    
                                    Calcule et affiche :
                                    - Progression en pourcentage
                                    - Temps écoulé depuis le début
                                    - Temps restant estimé (ETA) basé sur la vitesse actuelle
                                    """
                                    # Mise à jour de la barre de progression (clipper entre 0.0 et 1.0)
                                    progress = min(1.0, max(0.0, self.num_timesteps / self.total_timesteps))
                                    self.progress_bar.progress(progress)
                                    
                                    # Calcul du temps écoulé et estimé
                                    current_time = time.time()
                                    elapsed = current_time - self.start_time
                                    
                                    if progress > 0:
                                        # Estimation du temps total basée sur la progression actuelle
                                        # ETA = (temps écoulé / progression) - temps écoulé
                                        estimated_total = elapsed / progress
                                        remaining = estimated_total - elapsed
                                        
                                        # Mise à jour du texte de statut avec formatage
                                        progress_label = "Progression" if self.language == "fr" else "Progress"
                                        steps_label = "pas" if self.language == "fr" else "steps"
                                        self.status_text.markdown(
                                            f"<div class='progress-status'>"
                                            f"<strong>{progress_label}:</strong> {self.num_timesteps:,} / {self.total_timesteps:,} {steps_label} "
                                            f"<strong>({progress*100:.1f}%)</strong>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Formatage du temps écoulé (heures:minutes:secondes)
                                        hours = int(elapsed // 3600)
                                        minutes = int((elapsed % 3600) // 60)
                                        seconds = int(elapsed % 60)
                                        if hours > 0:
                                            time_str = f"{hours}h {minutes}m {seconds}s"
                                        elif minutes > 0:
                                            time_str = f"{minutes}m {seconds}s"
                                        else:
                                            time_str = f"{seconds}s"
                                        
                                        elapsed_label = "⏱️ Temps écoulé" if self.language == "fr" else "⏱️ Elapsed time"
                                        self.time_elapsed.metric(elapsed_label, time_str)
                                        
                                        # Formatage du temps restant estimé (ETA)
                                        if remaining > 0:
                                            hours = int(remaining // 3600)
                                            minutes = int((remaining % 3600) // 60)
                                            seconds = int(remaining % 60)
                                            if hours > 0:
                                                eta_str = f"{hours}h {minutes}m {seconds}s"
                                            elif minutes > 0:
                                                eta_str = f"{minutes}m {seconds}s"
                                            else:
                                                eta_str = f"{seconds}s"
                                            eta_label = "⏳ Temps restant" if self.language == "fr" else "⏳ Time remaining"
                                            self.eta_text.metric(eta_label, eta_str)
                                    
                                    return True  # Continue l'entraînement
                            
                            # Création du callback de progression
                            progress_callback = ProgressCallback(progress_bar, status_text, time_elapsed, eta_text, total_timesteps, start_time, language)
                            callbacks_list.append(progress_callback)
                        else:
                            # Fallback si BaseCallback n'est pas disponible
                            callbacks_list = None
                            status_text = st.empty()
                            time_elapsed = st.empty()
                            eta_text = st.empty()
                        
                        # Entraînement (sans progress_bar=True pour éviter l'erreur de dépendances tqdm/rich)
                        model.learn(
                            total_timesteps=total_timesteps,
                            callback=callbacks_list
                        )
                        
                        # Récupérer les métriques finales
                        if BaseCallback is not None and 'metrics_callback' in locals():
                            training_metrics = metrics_callback.get_final_metrics()
                        
                        # Sauvegarde du modèle
                        st.session_state.ppo_model = model
                        st.session_state.scenario2_training_metrics = training_metrics
                        # Note: soil_params, weather_params, season_length, max_irrigation sont déjà sauvegardés dans la sidebar
                        
                        # Message de succès
                        final_time = time.time() - start_time
                        hours = int(final_time // 3600)
                        minutes = int((final_time % 3600) // 60)
                        seconds = int(final_time % 60)
                        if hours > 0:
                            final_time_str = f"{hours}h {minutes}m {seconds}s"
                        elif minutes > 0:
                            final_time_str = f"{minutes}m {seconds}s"
                        else:
                            final_time_str = f"{seconds}s"
                        
                        st.success("✅ Entraînement terminé en {time} ! Modèle sauvegardé.".format(time=final_time_str) if language == "fr" else "✅ Training finished in {time}! Model saved.".format(time=final_time_str))
                        progress_bar.progress(1.0)
                        status_done = {
                            "fr": f"<div class='progress-status'><strong>✅ Entraînement terminé:</strong> {total_timesteps:,} pas</div>",
                            "en": f"<div class='progress-status'><strong>✅ Training finished:</strong> {total_timesteps:,} steps</div>",
                        }[language]
                        status_text.markdown(status_done, unsafe_allow_html=True)
                        
                        # Afficher les métriques d'entraînement
                        metrics_title = {
                            "fr": "### 📊 Métriques d'entraînement",
                            "en": "### 📊 Training metrics",
                        }[language]
                        metric_labels = {
                            "reward": "Récompense moyenne" if language == "fr" else "Average reward",
                            "ep_len": "Longueur moyenne épisode" if language == "fr" else "Average episode length",
                            "policy_loss": "Perte de politique" if language == "fr" else "Policy loss",
                            "detailed": "📈 Métriques détaillées" if language == "fr" else "📈 Detailed metrics",
                            "no_metrics": "⚠️ Aucune métrique disponible" if language == "fr" else "⚠️ No metrics available",
                        }
                        st.markdown(metrics_title)
                        if training_metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                ep_rew = training_metrics.get("rollout/ep_rew_mean", "N/A")
                                if isinstance(ep_rew, (int, float)):
                                    st.metric(metric_labels["reward"], f"{ep_rew:.2f}")
                                else:
                                    st.metric(metric_labels["reward"], ep_rew)
                            with col2:
                                ep_len = training_metrics.get("rollout/ep_len_mean", "N/A")
                                if isinstance(ep_len, (int, float)):
                                    st.metric(metric_labels["ep_len"], f"{ep_len:.1f}")
                                else:
                                    st.metric(metric_labels["ep_len"], ep_len)
                            with col3:
                                policy_loss = training_metrics.get("train/policy_loss", "N/A")
                                if isinstance(policy_loss, (int, float)):
                                    st.metric(metric_labels["policy_loss"], f"{policy_loss:.4f}")
                                else:
                                    st.metric(metric_labels["policy_loss"], policy_loss)
                            
                            # Métriques supplémentaires dans un expander
                            with st.expander(metric_labels["detailed"]):
                                metrics_text = ""
                                for key, value in sorted(training_metrics.items()):
                                    if isinstance(value, (int, float)):
                                        metrics_text += f"- **{key}**: {value:.6f}\n"
                                    else:
                                        metrics_text += f"- **{key}**: {value}\n"
                                st.markdown(metrics_text)
                        else:
                            st.info(metric_labels["no_metrics"])
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            
            # Statut du modèle
            if st.session_state.ppo_model is not None:
                st.info("✅ Un modèle est déjà entraîné. Vous pouvez l'utiliser pour l'évaluation dans l'onglet 'Évaluation'." if language == "fr" else "✅ A model is already trained. You can evaluate it in the 'Evaluation' tab.")
            else:
                st.info("👆 Cliquez sur 'Démarrer l'entraînement' pour entraîner un nouveau modèle avec les paramètres configurés dans la sidebar." if language == "fr" else "👆 Click 'Start training' to train a new model with the parameters set in the sidebar.")
    
    # ========================================================================
    # ONGLET 3 : SCÉNARIO 3 - NEURAL ODE
    # ========================================================================
    with tab3:
        language = st.session_state.get("ui_language", "fr")

        t3 = {
            "fr": {
                "header": "🔬 Scénario 3 — RL sur modèle hybride Physique + Neural ODE",
                "desc": "Le **Scénario 3** combine un modèle physique (bucket) avec une correction neuronale apprise (Neural ODE). Cette approche hybride permet de corriger les biais du modèle physique tout en conservant sa structure.",
                "pretrain_title": "### 🔧 Étape 1 : Pré-entraîner le modèle Neural ODE",
                "pretrain_principle": """
            <div style="background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                <strong>PRINCIPE :</strong> Le modèle Neural ODE apprend une correction Δψ à partir de données simulées.
                Cette correction sera ajoutée à la prédiction physique pour améliorer la précision du modèle.
            </div>
            """,
                "pretrain_traj": "Nombre de trajectoires",
                "pretrain_traj_help": "Nombre de trajectoires simulées pour générer les données d'entraînement",
                "pretrain_epochs": "Nombre d'epochs",
                "pretrain_epochs_help": "Nombre d'epochs d'entraînement du Neural ODE",
                "pretrain_lr": "Taux d'apprentissage",
                "pretrain_lr_help": "Taux d'apprentissage pour l'optimiseur Adam",
                "pretrain_batch": "Taille des batches",
                "pretrain_batch_help": "Taille des batches pour l'entraînement",
                "pretrain_btn": "🔧 Pré-entraîner le Neural ODE",
                "pretrain_done": "✅ Pré-entraînement terminé !",
                "pretrain_ready": "✅ Modèle Neural ODE pré-entraîné et prêt pour l'entraînement PPO",
                "pretrain_ready_score": "✅ Modèle Neural ODE pré-entraîné et prêt pour l'entraînement PPO\n📊 Score final (Loss): {score:.6f}",
                "ppo_title": "### 🚀 Étape 2 : Entraînement PPO sur modèle hybride",
                "ppo_warn": "⚠️ Veuillez d'abord pré-entraîner le modèle Neural ODE (Étape 1)",
                "ppo_principle": """
                <div style="background-color: #fff3cd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                    <strong>PRINCIPE :</strong> L'agent PPO apprend une politique d'irrigation optimale sur l'environnement
                    hybride (physique + Neural ODE). Cette approche combine la structure physique avec la flexibilité neuronale.
                </div>
                """,
                "ppo_steps": "Nombre de pas d'entraînement PPO",
                "ppo_steps_help": "Nombre total de pas de simulation pour l'entraînement PPO",
                "ppo_policy": "Type de politique",
                "ppo_policy_help": "Type de politique pour PPO",
                "ppo_adv": "Hyperparamètres PPO avancés",
                "lr_label": "Learning rate",
            },
            "en": {
                "header": "🔬 Scenario 3 — RL on hybrid Physical + Neural ODE model",
                "desc": "Scenario 3 combines a physical bucket model with a learned neural correction (Neural ODE). This hybrid approach corrects biases of the physical model while keeping its structure.",
                "pretrain_title": "### 🔧 Step 1: Pre-train the Neural ODE model",
                "pretrain_principle": """
            <div style="background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                <strong>PRINCIPLE:</strong> The Neural ODE learns a Δψ correction from simulated data.
                This correction is added to the physical prediction to improve accuracy.
            </div>
            """,
                "pretrain_traj": "Number of trajectories",
                "pretrain_traj_help": "Number of simulated trajectories to generate training data",
                "pretrain_epochs": "Number of epochs",
                "pretrain_epochs_help": "Training epochs for the Neural ODE",
                "pretrain_lr": "Learning rate",
                "pretrain_lr_help": "Learning rate for the Adam optimizer",
                "pretrain_batch": "Batch size",
                "pretrain_batch_help": "Batch size for training",
                "pretrain_btn": "🔧 Pre-train Neural ODE",
                "pretrain_done": "✅ Pre-training complete!",
                "pretrain_ready": "✅ Neural ODE pre-trained and ready for PPO training",
                "pretrain_ready_score": "✅ Neural ODE pre-trained and ready for PPO training\n📊 Final loss: {score:.6f}",
                "ppo_title": "### 🚀 Step 2: PPO training on hybrid model",
                "ppo_warn": "⚠️ Please pre-train the Neural ODE first (Step 1)",
                "ppo_principle": """
                <div style="background-color: #fff3cd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                    <strong>PRINCIPLE:</strong> The PPO agent learns an optimal irrigation policy on the hybrid
                    (physical + Neural ODE) environment, combining physical structure with neural flexibility.
                </div>
                """,
                "ppo_steps": "PPO training steps",
                "ppo_steps_help": "Total simulation steps for PPO training",
                "ppo_policy": "Policy type",
                "ppo_policy_help": "Policy type for PPO",
                "ppo_adv": "Advanced PPO hyperparameters",
                "lr_label": "Learning rate",
            },
        }[language]

        st.markdown(f'<h2 class="section-header">{t3["header"]}</h2>', unsafe_allow_html=True)
        
        st.markdown(t3["desc"])
        
        if not TORCH_AVAILABLE:
            st.error("⚠️ PyTorch n'est pas installé. Installez-le avec: `pip install torch`")
        elif not PPO_AVAILABLE:
            st.error("⚠️ Les bibliothèques RL ne sont pas disponibles. Vérifiez que `gymnasium` et `stable-baselines3` sont installés dans votre environnement Python.")
            with st.expander("🔍 Détails et solution"):
                st.code("""
Pour installer les dépendances nécessaires :
pip install gymnasium stable-baselines3

Ou avec conda :
conda install -c conda-forge gymnasium stable-baselines3

Note: Assurez-vous d'utiliser le même environnement Python que Streamlit.
                """)
        else:
            # Initialisation de l'état de session pour le scénario 3
            if "scenario3_residual_model" not in st.session_state:
                st.session_state.scenario3_residual_model = None
            if "scenario3_ppo_model" not in st.session_state:
                st.session_state.scenario3_ppo_model = None
            if "scenario3_pretrain_complete" not in st.session_state:
                st.session_state.scenario3_pretrain_complete = False
            
            # ====================================================================
            # ÉTAPE 1 : PRÉ-ENTRAÎNEMENT DU NEURAL ODE
            # ====================================================================
            st.markdown(t3["pretrain_title"])
            st.markdown(t3["pretrain_principle"], unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                pretrain_n_traj = st.number_input(
                    t3["pretrain_traj"],
                    min_value=10,
                    max_value=100,
                    value=32,
                    step=5,
                    help=t3["pretrain_traj_help"]
                )
            with col2:
                pretrain_n_epochs = st.number_input(
                    t3["pretrain_epochs"],
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help=t3["pretrain_epochs_help"]
                )
            with col3:
                pretrain_lr = st.number_input(
                    t3["pretrain_lr"],
                    min_value=1e-5,
                    max_value=1e-2,
                    value=1e-3,
                    step=1e-4,
                    format="%.4f",
                    help=t3["pretrain_lr_help"]
                )
            
            pretrain_batch_size = st.number_input(
                t3["pretrain_batch"],
                min_value=32,
                max_value=512,
                value=256,
                step=32,
                help=t3["pretrain_batch_help"]
            )
            
            if st.button(t3["pretrain_btn"], key="pretrain_ode_btn"):
                spinner_text = "Pré-entraînement en cours..." if language == "fr" else "Pre-training in progress..."
                with st.spinner(spinner_text):
                    try:
                        # Récupérer les paramètres du sol depuis la session
                        soil_params = st.session_state.get("soil_params", {})
                        soil = PhysicalBucket(**soil_params) if soil_params else PhysicalBucket()
                        
                        # Barre de progression
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def progress_callback(epoch, total_epochs, loss):
                            progress = min(1.0, max(0.0, epoch / total_epochs))  # Clipper entre 0.0 et 1.0
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
                        
                        # Pré-entraîner le modèle
                        residual_model, pretrain_score = pretrain_residual_ode(
                            soil=soil,
                            max_irrigation=st.session_state.get("max_irrigation", 20.0),
                            T=st.session_state.get("season_length", 120),
                            N_traj=pretrain_n_traj,
                            n_epochs=pretrain_n_epochs,
                            batch_size=pretrain_batch_size,
                            lr=pretrain_lr,
                            seed=st.session_state.get("seed", 123),  # Use seed from sidebar for consistency
                            device="cpu",
                            progress_callback=progress_callback
                        )
                        
                        st.session_state.scenario3_residual_model = residual_model
                        st.session_state.scenario3_pretrain_complete = True
                        st.session_state.scenario3_pretrain_score = pretrain_score
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        st.success(t3["pretrain_done"])
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors du pré-entraînement : {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Afficher le statut du pré-entraînement
            if st.session_state.scenario3_pretrain_complete:
                pretrain_score = st.session_state.get("scenario3_pretrain_score", None)
                if pretrain_score is not None:
                    st.info(t3["pretrain_ready_score"].format(score=pretrain_score))
                else:
                    st.info(t3["pretrain_ready"])
            
            st.markdown("---")
            
            # ====================================================================
            # ÉTAPE 2 : ENTRAÎNEMENT PPO SUR MODÈLE HYBRIDE
            # ====================================================================
            st.markdown(t3["ppo_title"])
            
            if not st.session_state.scenario3_pretrain_complete:
                st.warning(t3["ppo_warn"])
            else:
                st.markdown(t3["ppo_principle"], unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    ppo_timesteps = st.number_input(
                        t3["ppo_steps"],
                        min_value=10000,
                        max_value=500000,
                        value=50000,
                        step=10000,
                        key="scenario3_ppo_timesteps",
                        help=t3["ppo_steps_help"]
                    )
                with col2:
                    ppo_policy_type = st.selectbox(
                        t3["ppo_policy"],
                        options=["MlpPolicy"],
                        index=0,
                        key="scenario3_ppo_policy",
                        help=t3["ppo_policy_help"]
                    )
                
                # Hyperparamètres PPO (simplifiés pour le scénario 3)
                with st.expander(t3["ppo_adv"], expanded=False):
                    ppo_lr = st.number_input(
                        t3["lr_label"],
                        min_value=1e-5,
                        max_value=1e-2,
                        value=3e-4,
                        step=1e-5,
                        format="%.5f",
                        key="scenario3_ppo_lr"
                    )
                    ppo_gamma = st.number_input(
                        "Gamma (discount factor)",
                        min_value=0.9,
                        max_value=0.999,
                        value=0.99,
                        step=0.01,
                        key="scenario3_ppo_gamma"
                    )
                
                # Configuration MLP de la politique (comme scénario 2)
                policy_kwargs = render_mlp_policy_config(ppo_policy_type, language, key_prefix="scenario3")
                
                start_label = {
                    "fr": "🚀 Démarrer l'entraînement PPO (Scénario 3)",
                    "en": "🚀 Start PPO training (Scenario 3)",
                }[language]
                if st.button(start_label, type="primary", key="train_ode_btn"):
                    with st.spinner("Entraînement PPO en cours..." if language == "fr" else "PPO training in progress..."):
                        try:
                            # Section de progression agrandie
                            progress_title = {
                                "fr": "### 📊 Progression de l'entraînement",
                                "en": "### 📊 Training progress",
                            }[language]
                            st.markdown(progress_title)
                            
                            # Barre de progression agrandie
                            progress_bar = st.progress(0)
                            
                            # Zone de statut agrandie
                            status_container = st.container()
                            with status_container:
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    status_text = st.empty()
                                with col2:
                                    time_elapsed = st.empty()
                                with col3:
                                    eta_text = st.empty()
                            
                            # Callback personnalisé pour suivre la progression
                            import time
                            start_time = time.time()
                            data_source, data_path, era5_land_cfg = _get_weather_source_cfg()
                            external_weather = None
                            effective_season_length = st.session_state.get("season_length", 120)
                            if data_source == "era5_land" and data_path:
                                try:
                                    external_weather = load_data_for_simulation(
                                        data_source="era5_land",
                                        file_path=data_path,
                                        resample_freq=era5_land_cfg.get("resample_freq", "1D") if era5_land_cfg else "1D",
                                    )
                                    if external_weather and "rain" in external_weather:
                                        effective_season_length = len(np.asarray(external_weather["rain"]))
                                except Exception as exc:
                                    st.error(f"❌ Chargement ERA5-Land impossible : {exc}")
                                    raise

                            def ppo_progress_callback(current, total):
                                progress = min(1.0, max(0.0, current / total))  # Clipper entre 0.0 et 1.0
                                progress_bar.progress(progress)
                                
                                # Calcul du temps écoulé et estimé
                                current_time = time.time()
                                elapsed = current_time - start_time
                                
                                if progress > 0:
                                    # Estimation du temps total basée sur la progression actuelle
                                    estimated_total = elapsed / progress
                                    remaining = estimated_total - elapsed
                                    
                                    # Mise à jour du texte de statut avec formatage
                                    progress_label = "Progression" if language == "fr" else "Progress"
                                    steps_label = "pas" if language == "fr" else "steps"
                                    status_text.markdown(
                                        f"<div class='progress-status'>"
                                        f"<strong>{progress_label}:</strong> {current:,} / {total:,} {steps_label} "
                                        f"<strong>({progress*100:.1f}%)</strong>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Formatage du temps écoulé (heures:minutes:secondes)
                                    hours = int(elapsed // 3600)
                                    minutes = int((elapsed % 3600) // 60)
                                    seconds = int(elapsed % 60)
                                    if hours > 0:
                                        time_str = f"{hours}h {minutes}m {seconds}s"
                                    elif minutes > 0:
                                        time_str = f"{minutes}m {seconds}s"
                                    else:
                                        time_str = f"{seconds}s"
                                    
                                    time_elapsed.metric("⏱️ Temps écoulé" if language == "fr" else "⏱️ Elapsed time", time_str)
                                    
                                    # Formatage du temps restant estimé (ETA)
                                    if remaining > 0:
                                        hours = int(remaining // 3600)
                                        minutes = int((remaining % 3600) // 60)
                                        seconds = int(remaining % 60)
                                        if hours > 0:
                                            eta_str = f"{hours}h {minutes}m {seconds}s"
                                        elif minutes > 0:
                                            eta_str = f"{minutes}m {seconds}s"
                                        else:
                                            eta_str = f"{seconds}s"
                                        eta_text.metric("⏳ Temps restant" if language == "fr" else "⏳ Remaining time", eta_str)
                            
                            # Entraîner PPO
                            ppo_model, training_metrics = train_ppo_hybrid_ode(
                                residual_ode_model=st.session_state.scenario3_residual_model,
                                season_length=effective_season_length,
                                max_irrigation=st.session_state.get("max_irrigation", 20.0),
                                total_timesteps=ppo_timesteps,
                                seed=st.session_state.get("seed", 123),
                                soil_params=st.session_state.get("soil_params"),
                                weather_params=st.session_state.get("weather_params"),
                                weather_shift_cfg=st.session_state.get("proposal_a_config"),
                                external_weather=external_weather,
                                ppo_kwargs={
                                    "learning_rate": ppo_lr,
                                    "gamma": ppo_gamma,
                                    **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
                                },
                                progress_callback=ppo_progress_callback
                            )
                            
                            st.session_state.scenario3_ppo_model = ppo_model
                            st.session_state.scenario3_training_metrics = training_metrics
                            
                            progress_bar.progress(1.0)
                            status_done = {
                                "fr": f"<div class='progress-status'><strong>✅ Entraînement terminé:</strong> {ppo_timesteps:,} pas</div>",
                                "en": f"<div class='progress-status'><strong>✅ Training finished:</strong> {ppo_timesteps:,} steps</div>",
                            }[language]
                            status_text.markdown(status_done, unsafe_allow_html=True)
                            
                            # Message de succès avec temps d'entraînement
                            final_time = time.time() - start_time
                            hours = int(final_time // 3600)
                            minutes = int((final_time % 3600) // 60)
                            seconds = int(final_time % 60)
                            if hours > 0:
                                final_time_str = f"{hours}h {minutes}m {seconds}s"
                            elif minutes > 0:
                                final_time_str = f"{minutes}m {seconds}s"
                            else:
                                final_time_str = f"{seconds}s"
                            
                            st.success("✅ Entraînement terminé en {time} ! Modèle sauvegardé.".format(time=final_time_str) if language == "fr" else "✅ Training finished in {time}! Model saved.".format(time=final_time_str))
                            
                            # Afficher les métriques d'entraînement
                            metrics_title = {
                                "fr": "### 📊 Métriques d'entraînement",
                                "en": "### 📊 Training metrics",
                            }[language]
                            metric_labels = {
                                "reward": "Récompense moyenne" if language == "fr" else "Average reward",
                                "ep_len": "Longueur moyenne épisode" if language == "fr" else "Average episode length",
                                "policy_loss": "Perte de politique" if language == "fr" else "Policy loss",
                                "detailed": "📈 Métriques détaillées" if language == "fr" else "📈 Detailed metrics",
                                "no_metrics": "⚠️ Aucune métrique disponible" if language == "fr" else "⚠️ No metrics available",
                            }
                            st.markdown(metrics_title)
                            if training_metrics:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    ep_rew = training_metrics.get("rollout/ep_rew_mean", "N/A")
                                    if isinstance(ep_rew, (int, float)):
                                        st.metric(metric_labels["reward"], f"{ep_rew:.2f}")
                                    else:
                                        st.metric(metric_labels["reward"], ep_rew)
                                with col2:
                                    ep_len = training_metrics.get("rollout/ep_len_mean", "N/A")
                                    if isinstance(ep_len, (int, float)):
                                        st.metric(metric_labels["ep_len"], f"{ep_len:.1f}")
                                    else:
                                        st.metric(metric_labels["ep_len"], ep_len)
                                with col3:
                                    policy_loss = training_metrics.get("train/policy_loss", "N/A")
                                    if isinstance(policy_loss, (int, float)):
                                        st.metric(metric_labels["policy_loss"], f"{policy_loss:.4f}")
                                    else:
                                        st.metric(metric_labels["policy_loss"], policy_loss)
                                
                                # Métriques supplémentaires dans un expander
                                with st.expander(metric_labels["detailed"]):
                                    metrics_text = ""
                                    for key, value in sorted(training_metrics.items()):
                                        if isinstance(value, (int, float)):
                                            metrics_text += f"- **{key}**: {value:.6f}\n"
                                        else:
                                            metrics_text += f"- **{key}**: {value}\n"
                                    st.markdown(metrics_text)
                            else:
                                st.info(metric_labels["no_metrics"])
                            
                            status_title = {
                                "fr": "### ✅ Modèle entraîné",
                                "en": "### ✅ Model trained",
                            }[language]
                            status_msg = {
                                "fr": "✅ Le modèle PPO hybride est prêt. Vous pouvez l'évaluer dans l'onglet 'Évaluation'.",
                                "en": "✅ PPO hybrid model is ready. You can evaluate it in the 'Evaluation' tab.",
                            }[language]
                            st.markdown(status_title)
                            st.info(status_msg)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'entraînement PPO : {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Afficher le statut
                if st.session_state.scenario3_ppo_model is not None:
                    st.info({
                        "fr": "✅ Modèle PPO hybride entraîné et disponible pour l'évaluation",
                        "en": "✅ PPO hybrid model trained and available for evaluation",
                    }[language])

    # ========================================================================
    # ONGLET 3B : SCÉNARIO 3B - NEURAL ODE CONTINU
    # ========================================================================
    with tab3b:
        language = st.session_state.get("ui_language", "fr")

        t3b = {
            "fr": {
                "header": "🧠 Scénario 3b — RL + Neural ODE continu",
                "desc": "Variante continue du scénario 3 : la dérivée dψ/dt est apprise puis intégrée (torchdiffeq si dispo, sinon Euler) pour corriger la prédiction physique.",
                "pretrain_title": "### 🔧 Étape 1 : Pré-entraîner le Neural ODE continu",
                "pretrain_principle": """
            <div style="background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                <strong>PRINCIPE :</strong> On apprend dψ/dt à partir de données simulées, puis on intègre sur un pas (1 jour).
                Cela permet une correction plus lisse que la version discrète.
            </div>
            """,
                "pretrain_traj": "Nombre de trajectoires",
                "pretrain_traj_help": "Nombre de trajectoires simulées pour générer les données d'entraînement",
                "pretrain_epochs": "Nombre d'epochs",
                "pretrain_epochs_help": "Nombre d'epochs d'entraînement du Neural ODE continu",
                "pretrain_lr": "Taux d'apprentissage",
                "pretrain_lr_help": "Taux d'apprentissage pour l'optimiseur Adam",
                "pretrain_batch": "Taille des batches",
                "pretrain_batch_help": "Taille des batches pour l'entraînement",
                "pretrain_btn": "🔧 Pré-entraîner le Neural ODE continu",
                "pretrain_done": "✅ Pré-entraînement terminé !",
                "pretrain_ready": "✅ Modèle Neural ODE continu prêt pour l'entraînement PPO",
                "pretrain_ready_score": "✅ Neural ODE continu prêt pour PPO\n📊 Score final (Loss): {score:.6f}",
                "ppo_title": "### 🚀 Étape 2 : Entraînement PPO (hybride continu)",
                "ppo_warn": "⚠️ Pré-entraînez d'abord le Neural ODE continu (Étape 1)",
                "ppo_principle": """
                <div style="background-color: #fff3cd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                    <strong>PRINCIPE :</strong> L'agent PPO apprend une politique d'irrigation sur l'environnement
                    physique + correction continue (Neural ODE continu).
                </div>
                """,
                "ppo_steps": "Nombre de pas d'entraînement PPO",
                "ppo_steps_help": "Nombre total de pas de simulation pour l'entraînement PPO",
                "ppo_policy": "Type de politique",
                "ppo_policy_help": "Type de politique pour PPO",
                "ppo_adv": "Hyperparamètres PPO avancés",
                "lr_label": "Learning rate",
            },
            "en": {
                "header": "🧠 Scenario 3b — RL + Continuous Neural ODE",
                "desc": "Continuous variant of Scenario 3: learn dψ/dt and integrate it (torchdiffeq when available, else Euler) to correct the physical prediction.",
                "pretrain_title": "### 🔧 Step 1: Pre-train the continuous Neural ODE",
                "pretrain_principle": """
            <div style="background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                <strong>PRINCIPLE:</strong> Learn dψ/dt from simulated data and integrate over one step (1 day).
                This yields a smoother correction than the discrete version.
            </div>
            """,
                "pretrain_traj": "Number of trajectories",
                "pretrain_traj_help": "Number of simulated trajectories to generate training data",
                "pretrain_epochs": "Number of epochs",
                "pretrain_epochs_help": "Training epochs for the continuous Neural ODE",
                "pretrain_lr": "Learning rate",
                "pretrain_lr_help": "Learning rate for Adam",
                "pretrain_batch": "Batch size",
                "pretrain_batch_help": "Batch size for training",
                "pretrain_btn": "🔧 Pre-train continuous Neural ODE",
                "pretrain_done": "✅ Pre-training complete!",
                "pretrain_ready": "✅ Continuous Neural ODE ready for PPO training",
                "pretrain_ready_score": "✅ Continuous Neural ODE ready for PPO training\n📊 Final loss: {score:.6f}",
                "ppo_title": "### 🚀 Step 2: PPO training (continuous hybrid)",
                "ppo_warn": "⚠️ Please pre-train the continuous Neural ODE first (Step 1)",
                "ppo_principle": """
                <div style="background-color: #fff3cd; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                    <strong>PRINCIPLE:</strong> PPO learns an irrigation policy on the physical environment plus the
                    continuous correction (Neural ODE).
                </div>
                """,
                "ppo_steps": "PPO training steps",
                "ppo_steps_help": "Total simulation steps for PPO training",
                "ppo_policy": "Policy type",
                "ppo_policy_help": "Policy type for PPO",
                "ppo_adv": "Advanced PPO hyperparameters",
                "lr_label": "Learning rate",
            },
        }[language]

        st.markdown(f'<h2 class="section-header">{t3b["header"]}</h2>', unsafe_allow_html=True)
        st.markdown(t3b["desc"])

        if not TORCH_AVAILABLE:
            st.error("⚠️ PyTorch n'est pas installé. Installez-le avec: `pip install torch`")
        elif not PPO_AVAILABLE:
            st.error("⚠️ Les bibliothèques RL ne sont pas disponibles. Vérifiez que `gymnasium` et `stable-baselines3` sont installés.")
        else:
            if "scenario3b_residual_model" not in st.session_state:
                st.session_state.scenario3b_residual_model = None
            if "scenario3b_ppo_model" not in st.session_state:
                st.session_state.scenario3b_ppo_model = None
            if "scenario3b_pretrain_complete" not in st.session_state:
                st.session_state.scenario3b_pretrain_complete = False

            st.markdown(t3b["pretrain_title"])
            st.markdown(t3b["pretrain_principle"], unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                pretrain_n_traj = st.number_input(
                    t3b["pretrain_traj"],
                    min_value=10,
                    max_value=100,
                    value=32,
                    step=5,
                    help=t3b["pretrain_traj_help"],
                    key="scenario3b_pretrain_traj"
                )
            with col2:
                pretrain_n_epochs = st.number_input(
                    t3b["pretrain_epochs"],
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help=t3b["pretrain_epochs_help"],
                    key="scenario3b_pretrain_epochs"
                )
            with col3:
                pretrain_lr = st.number_input(
                    t3b["pretrain_lr"],
                    min_value=1e-5,
                    max_value=1e-2,
                    value=1e-3,
                    step=1e-4,
                    format="%.4f",
                    help=t3b["pretrain_lr_help"],
                    key="scenario3b_pretrain_lr"
                )

            pretrain_batch_size = st.number_input(
                t3b["pretrain_batch"],
                min_value=32,
                max_value=512,
                value=256,
                step=32,
                help=t3b["pretrain_batch_help"],
                key="scenario3b_pretrain_batch_size"
            )

            if st.button(t3b["pretrain_btn"], key="pretrain_ode_cont_btn"):
                spinner_text = "Pré-entraînement en cours..." if language == "fr" else "Pre-training in progress..."
                with st.spinner(spinner_text):
                    try:
                        soil_params = st.session_state.get("soil_params", {})
                        soil = PhysicalBucket(**soil_params) if soil_params else PhysicalBucket()

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(epoch, total_epochs, loss):
                            progress = min(1.0, max(0.0, epoch / total_epochs))
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")

                        residual_model, pretrain_score = pretrain_continuous_residual_ode(
                            soil=soil,
                            max_irrigation=st.session_state.get("max_irrigation", 20.0),
                            T=st.session_state.get("season_length", 120),
                            N_traj=pretrain_n_traj,
                            n_epochs=pretrain_n_epochs,
                            batch_size=pretrain_batch_size,
                            lr=pretrain_lr,
                            seed=st.session_state.get("seed", 123),
                            device="cpu",
                            progress_callback=progress_callback
                        )

                        st.session_state.scenario3b_residual_model = residual_model
                        st.session_state.scenario3b_pretrain_complete = True
                        st.session_state.scenario3b_pretrain_score = pretrain_score

                        progress_bar.progress(1.0)
                        status_text.empty()
                        st.success(t3b["pretrain_done"])
                    except Exception as e:
                        st.error(f"❌ Erreur lors du pré-entraînement : {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            if st.session_state.scenario3b_pretrain_complete:
                pretrain_score = st.session_state.get("scenario3b_pretrain_score", None)
                if pretrain_score is not None:
                    st.info(t3b["pretrain_ready_score"].format(score=pretrain_score))
                else:
                    st.info(t3b["pretrain_ready"])

            st.markdown("---")
            st.markdown(t3b["ppo_title"])

            if not st.session_state.scenario3b_pretrain_complete:
                st.warning(t3b["ppo_warn"])
            else:
                st.markdown(t3b["ppo_principle"], unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    ppo_timesteps = st.number_input(
                        t3b["ppo_steps"],
                        min_value=10000,
                        max_value=500000,
                        value=50000,
                        step=10000,
                        key="scenario3b_ppo_timesteps",
                        help=t3b["ppo_steps_help"]
                    )
                with col2:
                    ppo_policy_type = st.selectbox(
                        t3b["ppo_policy"],
                        options=["MlpPolicy"],
                        index=0,
                        key="scenario3b_ppo_policy",
                        help=t3b["ppo_policy_help"]
                    )

                with st.expander(t3b["ppo_adv"], expanded=False):
                    ppo_lr = st.number_input(
                        t3b["lr_label"],
                        min_value=1e-5,
                        max_value=1e-2,
                        value=3e-4,
                        step=1e-5,
                        format="%.5f",
                        key="scenario3b_ppo_lr"
                    )
                    ppo_gamma = st.number_input(
                        "Gamma (discount factor)",
                        min_value=0.9,
                        max_value=0.999,
                        value=0.99,
                        step=0.01,
                        key="scenario3b_ppo_gamma"
                    )
                
                # Configuration MLP de la politique
                policy_kwargs = render_mlp_policy_config(ppo_policy_type, language, key_prefix="scenario3b")

                start_label = {
                    "fr": "🚀 Démarrer l'entraînement PPO (Scénario 3b)",
                    "en": "🚀 Start PPO training (Scenario 3b)",
                }[language]
                if st.button(start_label, type="primary", key="train_ode_cont_btn"):
                    with st.spinner("Entraînement PPO en cours..." if language == "fr" else "PPO training in progress..."):
                        try:
                            # Charger ERA5-Land si sélectionné
                            data_source, data_path, era5_land_cfg = _get_weather_source_cfg()
                            external_weather = None
                            effective_season_length = st.session_state.get("season_length", 120)
                            if data_source == "era5_land" and data_path:
                                try:
                                    external_weather = load_data_for_simulation(
                                        data_source="era5_land",
                                        file_path=data_path,
                                        resample_freq=era5_land_cfg.get("resample_freq", "1D") if era5_land_cfg else "1D",
                                    )
                                    if external_weather and "rain" in external_weather:
                                        effective_season_length = len(np.asarray(external_weather["rain"]))
                                except Exception as exc:
                                    st.warning(f"⚠️ Impossible de charger ERA5-Land, utilisation météo synthétique : {exc}")
                                    external_weather = None

                            import time
                            progress_bar = st.progress(0)
                            status_container = st.container()
                            with status_container:
                                col1s, col2s, col3s = st.columns([2, 1, 1])
                                with col1s:
                                    status_text = st.empty()
                                with col2s:
                                    time_elapsed = st.empty()
                                with col3s:
                                    eta_text = st.empty()

                            start_time = time.perf_counter()

                            def progress_callback(current: int, total_timesteps: int):
                                total_timesteps = max(1, total_timesteps)
                                progress = min(1.0, current / total_timesteps)
                                elapsed = time.perf_counter() - start_time
                                eta = (elapsed / progress - elapsed) if progress > 0 else float("nan")
                                progress_bar.progress(progress)

                                # Texte de progression aligné sur le style des autres scénarios
                                progress_label = "Progression" if language == "fr" else "Progress"
                                steps_label = "pas" if language == "fr" else "steps"
                                status_text.markdown(
                                    f"<div class='progress-status'>"
                                    f"<strong>{progress_label}:</strong> {current:,} / {total_timesteps:,} {steps_label} "
                                    f"<strong>({progress*100:.1f}%)</strong>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                                # Temps écoulé formaté
                                hours = int(elapsed // 3600)
                                minutes = int((elapsed % 3600) // 60)
                                seconds = int(elapsed % 60)
                                if hours > 0:
                                    time_str = f"{hours}h {minutes}m {seconds}s"
                                elif minutes > 0:
                                    time_str = f"{minutes}m {seconds}s"
                                else:
                                    time_str = f"{seconds}s"
                                elapsed_label = "⏱️ Temps écoulé" if language == "fr" else "⏱️ Elapsed time"
                                time_elapsed.metric(elapsed_label, time_str)

                                # ETA formaté
                                if eta > 0:
                                    hours_eta = int(eta // 3600)
                                    minutes_eta = int((eta % 3600) // 60)
                                    seconds_eta = int(eta % 60)
                                    if hours_eta > 0:
                                        eta_str = f"{hours_eta}h {minutes_eta}m {seconds_eta}s"
                                    elif minutes_eta > 0:
                                        eta_str = f"{minutes_eta}m {seconds_eta}s"
                                    else:
                                        eta_str = f"{seconds_eta}s"
                                    eta_label = "⏳ Temps restant" if language == "fr" else "⏳ Time remaining"
                                    eta_text.metric(eta_label, eta_str)

                            ppo_config = {
                                "policy": ppo_policy_type,
                                "learning_rate": ppo_lr,
                                "gamma": ppo_gamma,
                                **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
                            }

                            ppo_model, training_metrics = train_ppo_hybrid_ode_cont(
                                residual_ode_model=st.session_state.scenario3b_residual_model,
                                season_length=effective_season_length,
                                max_irrigation=st.session_state.get("max_irrigation", 20.0),
                                total_timesteps=ppo_timesteps,
                                seed=st.session_state.get("seed", 123),
                                soil_params=st.session_state.get("soil_params"),
                                weather_params=st.session_state.get("weather_params"),
                                external_weather=external_weather,
                                ppo_kwargs=ppo_config,
                                weather_shift_cfg=st.session_state.get("proposal_a_config"),
                                progress_callback=progress_callback
                            )

                            st.session_state.scenario3b_ppo_model = ppo_model
                            st.session_state.scenario3b_training_metrics = training_metrics

                            progress_bar.progress(1.0)
                            status_text.text("✅ Entraînement terminé" if language == "fr" else "✅ Training finished")
                            st.success("✅ Modèle PPO entraîné pour le Scénario 3b" if language == "fr" else "✅ PPO model trained for Scenario 3b")

                            if training_metrics:
                                metrics_title = {"fr": "### 📊 Métriques d'entraînement", "en": "### 📊 Training metrics"}[language]
                                metric_labels = {
                                    "reward": "Récompense moyenne" if language == "fr" else "Average reward",
                                    "ep_len": "Longueur épisode" if language == "fr" else "Episode length",
                                    "policy_loss": "Perte de politique" if language == "fr" else "Policy loss",
                                    "detailed": "📈 Métriques détaillées" if language == "fr" else "📈 Detailed metrics",
                                    "no_metrics": "⚠️ Aucune métrique disponible" if language == "fr" else "⚠️ No metrics available",
                                }
                                st.markdown(metrics_title)
                                col1m, col2m, col3m = st.columns(3)
                                with col1m:
                                    if "rollout/ep_rew_mean" in training_metrics:
                                        st.metric(metric_labels["reward"], f"{training_metrics['rollout/ep_rew_mean']:.2f}")
                                with col2m:
                                    if "rollout/ep_len_mean" in training_metrics:
                                        st.metric(metric_labels["ep_len"], f"{training_metrics['rollout/ep_len_mean']:.1f}")
                                with col3m:
                                    if "train/policy_loss" in training_metrics:
                                        st.metric(metric_labels["policy_loss"], f"{training_metrics['train/policy_loss']:.4f}")
                                
                                with st.expander(metric_labels["detailed"]):
                                    st.json(training_metrics)
                            else:
                                st.info("⚠️ Aucune métrique disponible" if language == "fr" else "⚠️ No metrics available")
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

            if st.session_state.scenario3b_ppo_model is not None:
                st.info("✅ Modèle disponible pour l'évaluation dans l'onglet 'Évaluation'." if language == "fr" else "✅ Model ready for evaluation in the 'Evaluation' tab.")

    # ========================================================================
    # ONGLET 7 : ÉVALUATION
    # ========================================================================
    with tab5:
        # Sélecteur de langue (synchronisé)
        language = st.session_state.get("ui_language", "fr")

        eval_text = {
            "fr": {
                "header": "📈 Évaluation du modèle",
                "desc": "**Évaluation :** Testez le modèle entraîné sur une nouvelle saison (avec une graine différente) pour évaluer sa performance et sa capacité de généralisation.",
                "model_label": "Modèle à évaluer",
                "eval_btn": "🔍 Évaluer le modèle",
                "seed_section": "### ⚙️ Configuration de l'évaluation",
                "seed_label": "Graine pour l'évaluation",
                "seed_help": "Utilisez une graine différente de l'entraînement pour tester la généralisation",
                "no_model": {
                    "scenario2": "⚠️ Aucun modèle du scénario 2 entraîné. Veuillez d'abord entraîner un modèle dans l'onglet 'Scénario 2 (Entraînement PPO)'.",
                    "scenario3": "⚠️ Aucun modèle du scénario 3 entraîné. Veuillez d'abord entraîner un modèle dans l'onglet 'Scénario 3 (Neural ODE)'.",
                    "scenario3b": "⚠️ Aucun modèle du scénario 3b entraîné. Veuillez d'abord entraîner un modèle dans l'onglet 'Scénario 3b (Neural ODE continu)'.",
                },
                "eval_running": "Évaluation en cours...",
                "eval_done": "✅ Évaluation terminée !",
            },
            "en": {
                "header": "📈 Model evaluation",
                "desc": "**Evaluation:** Test the trained model on a new season (with a different seed) to assess performance and generalization.",
                "model_label": "Model to evaluate",
                "eval_btn": "🔍 Evaluate model",
                "seed_section": "### ⚙️ Evaluation settings",
                "seed_label": "Seed for evaluation",
                "seed_help": "Use a different seed from training to test generalization",
                "no_model": {
                    "scenario2": "⚠️ No Scenario 2 model trained. Please train one in the 'Scenario 2 (PPO Training)' tab first.",
                    "scenario3": "⚠️ No Scenario 3 model trained. Please train one in the 'Scenario 3 (Neural ODE)' tab first.",
                    "scenario3b": "⚠️ No Scenario 3b model trained. Please train one in the 'Scenario 3b (Continuous Neural ODE)' tab first.",
                },
                "eval_running": "Evaluation in progress...",
                "eval_done": "✅ Evaluation complete!",
            },
        }[language]

        st.markdown(f'<h2 class="section-header">{eval_text["header"]}</h2>', unsafe_allow_html=True)
        st.markdown(eval_text["desc"])
        
        scenario_labels = {
            "scenario2": {"fr": "Scénario 2 (PPO physique)", "en": "Scenario 2 (PPO physical)"},
            "scenario3": {"fr": "Scénario 3 (PPO hybride Neural ODE)", "en": "Scenario 3 (PPO hybrid Neural ODE)"},
            "scenario3b": {"fr": "Scénario 3b (Neural ODE continu)", "en": "Scenario 3b (Continuous Neural ODE)"},
        }
        
        # Sélection du modèle à évaluer (valeurs = clés internes, affichage traduit)
        model_to_evaluate = st.radio(
            eval_text["model_label"],
            options=list(scenario_labels.keys()),
            index=0,
            horizontal=True,
            format_func=lambda k: scenario_labels[k][language]
        )
        
        # Vérifier la disponibilité des modèles
        model = None
        use_residual = False
        use_residual_cont = False
        residual_model = None
        
        if model_to_evaluate == "scenario2":
            if st.session_state.ppo_model is None:
                st.warning("⚠️ Aucun modèle du scénario 2 entraîné. Veuillez d'abord entraîner un modèle dans l'onglet 'Scénario 2 (Entraînement PPO)'.")
            else:
                model = st.session_state.ppo_model
                use_residual = False
        elif model_to_evaluate == "scenario3":
            if st.session_state.scenario3_ppo_model is None:
                st.warning("⚠️ Aucun modèle du scénario 3 entraîné. Veuillez d'abord entraîner un modèle dans l'onglet 'Scénario 3 (Neural ODE)'.")
            else:
                model = st.session_state.scenario3_ppo_model
                use_residual = True
                residual_model = st.session_state.scenario3_residual_model
        elif model_to_evaluate == "scenario3b":
            if st.session_state.scenario3b_ppo_model is None:
                st.warning("⚠️ Aucun modèle du scénario 3b entraîné. Veuillez d'abord entraîner un modèle dans l'onglet 'Scénario 3b (Neural ODE continu)'.")
            else:
                model = st.session_state.scenario3b_ppo_model
                use_residual = True
                use_residual_cont = True
                residual_model = st.session_state.scenario3b_residual_model
        
        if (model_to_evaluate == "scenario2" and st.session_state.ppo_model is not None) or \
           (model_to_evaluate == "scenario3" and st.session_state.scenario3_ppo_model is not None) or \
           (model_to_evaluate == "scenario3b" and st.session_state.scenario3b_ppo_model is not None):
            # Section de configuration de la graine
            st.markdown(eval_text["seed_section"])
            
            eval_seed = st.number_input(
                eval_text["seed_label"],
                min_value=0,
                max_value=10000,
                value=123,
                step=1,
                help=eval_text["seed_help"]
            )
                        
            # Bouton d'évaluation
            if st.button(eval_text["eval_btn"], type="primary", key="eval_model_btn"):
                # Validation des ressources nécessaires
                if model is None:
                    st.error({
                        "fr": "❌ Aucun modèle entraîné n'est disponible pour ce scénario.",
                        "en": "❌ No trained model is available for this scenario.",
                    }[language])
                    st.stop()
                if use_residual and residual_model is None:
                    st.error("❌ Modèle résiduel (Neural ODE) manquant." if language == "fr" else "❌ Residual model (Neural ODE) is missing.")
                    st.stop()
                    st.stop()
                    st.stop()
                    st.stop()
                    st.stop()
                    st.stop()

                with st.spinner(eval_text["eval_running"]):
                    try:
                        # Préparer les paramètres selon le modèle choisi
                        eval_kwargs = {
                            "model": model,
                            "season_length": st.session_state.get("season_length", 120),
                            "max_irrigation": st.session_state.get("max_irrigation", 20.0),
                            "seed": eval_seed,
                            "soil_params": st.session_state.get("soil_params"),
                            "weather_params": st.session_state.get("weather_params")
                        }
                        data_source, data_path, era5_land_cfg = _get_weather_source_cfg()
                        eval_kwargs["data_source"] = data_source
                        eval_kwargs["data_path"] = data_path
                        eval_kwargs["era5_land_cfg"] = era5_land_cfg
                        if data_source == "era5_land" and data_path:
                            try:
                                weather_bundle = load_data_for_simulation(
                                    data_source="era5_land",
                                    file_path=data_path,
                                    resample_freq=era5_land_cfg.get("resample_freq", "1D") if era5_land_cfg else "1D",
                                )
                                if weather_bundle and "rain" in weather_bundle:
                                    eval_kwargs["season_length"] = len(np.asarray(weather_bundle["rain"]))
                            except Exception as exc:
                                st.warning(f"⚠️ Impossible de charger ERA5-Land pour l'évaluation : {exc}")
                        
                        if use_residual and residual_model is not None:
                            eval_kwargs["residual_ode"] = residual_model
                        rollout = evaluate_episode(**eval_kwargs)
                        
                        # Ajouter un identifiant du scénario au rollout
                        if use_residual:
                            scenario_id = "scenario3b" if use_residual_cont else "scenario3"
                        else:
                            scenario_id = "scenario2"
                        rollout["scenario"] = scenario_id
                        
                        # Stocker le rollout dans la variable appropriée selon le scénario
                        if use_residual:
                            if use_residual_cont:
                                st.session_state.scenario3b_rollout = rollout
                            else:
                                st.session_state.scenario3_rollout = rollout
                        else:
                            st.session_state.scenario2_rollout = rollout
                        
                        # Garder aussi evaluation_rollout pour rétrocompatibilité
                        st.session_state.evaluation_rollout = rollout
                        st.success("✅ Évaluation terminée !")
                        
                        # Métriques résumées
                        st.markdown("### 📊 Métriques de l'épisode")
                        
                        # Utiliser un layout 2x2 pour plus d'espace
                        col1, col2 = st.columns(2)
                        col3, col4 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "💧 Eau totale irriguée",
                                f"{rollout['I'].sum():.1f} mm",
                                help="Quantité totale d'eau d'irrigation appliquée"
                            )
                        with col2:
                            st.metric(
                                "🌧️ Pluie totale",
                                f"{rollout['R'].sum():.1f} mm",
                                help="Quantité totale de pluie reçue"
                            )
                        with col3:
                            st.metric(
                                "💨 Drainage total",
                                f"{rollout['D'].sum():.1f} mm",
                                help="Quantité totale d'eau drainée"
                            )
                        with col4:
                            psi_mean = rollout['psi'].mean()
                            psi_min = rollout['psi'].min()
                            psi_max = rollout['psi'].max()
                            st.metric(
                                "📊 Tension moyenne",
                                f"{psi_mean:.1f} cbar",
                                delta=f"Min: {psi_min:.1f} | Max: {psi_max:.1f}",
                                help="Tension matricielle moyenne"
                            )
                        
                        # Métriques supplémentaires
                        st.markdown("#### 📈 Métriques supplémentaires")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_water = rollout['I'].sum() + rollout['R'].sum()
                            st.metric("💦 Eau totale (I+R)", f"{total_water:.1f} mm")
                        with col2:
                            etc_total = rollout['ETc'].sum()
                            st.metric("🌱 ETc totale", f"{etc_total:.1f} mm")
                        with col3:
                            water_balance = total_water - etc_total - rollout['D'].sum()
                            st.metric("⚖️ Bilan hydrique", f"{water_balance:.1f} mm")
                        with col4:
                            days_stress = np.sum((rollout['psi'] < 20) | (rollout['psi'] > 60))
                            st.metric("⚠️ Jours hors confort", f"{days_stress} / {len(rollout['psi'])}")
                        
                        st.info("✅ Résultats disponibles. Consultez l'onglet 'Visualisation' pour voir les graphiques détaillés.")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'évaluation: {str(e)}")
            
            # Affichage des résultats si disponibles
            # Afficher les métriques du dernier rollout évalué (rétrocompatibilité)
            # On essaie d'abord les rollouts séparés, puis l'ancien format
            rollout = None
            if st.session_state.scenario3b_rollout is not None:
                rollout = st.session_state.scenario3b_rollout
            elif st.session_state.scenario3_rollout is not None:
                rollout = st.session_state.scenario3_rollout
            elif st.session_state.scenario2_rollout is not None:
                rollout = st.session_state.scenario2_rollout
            elif "evaluation_rollout" in st.session_state and st.session_state.evaluation_rollout is not None:
                rollout = st.session_state.evaluation_rollout
            
            if rollout is not None:
                
                st.markdown("### 📊 Métriques de l'épisode (dernière évaluation)")
                
                # Utiliser un layout 2x2 pour plus d'espace
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                with col1:
                    st.metric(
                        "💧 Eau totale irriguée",
                        f"{rollout['I'].sum():.1f} mm",
                        help="Quantité totale d'eau d'irrigation appliquée"
                    )
                with col2:
                    st.metric(
                        "🌧️ Pluie totale",
                        f"{rollout['R'].sum():.1f} mm",
                        help="Quantité totale de pluie reçue"
                    )
                with col3:
                    st.metric(
                        "💨 Drainage total",
                        f"{rollout['D'].sum():.1f} mm",
                        help="Quantité totale d'eau drainée"
                    )
                with col4:
                    psi_mean = rollout['psi'].mean()
                    psi_min = rollout['psi'].min()
                    psi_max = rollout['psi'].max()
                    st.metric(
                        "📊 Tension moyenne",
                        f"{psi_mean:.1f} cbar",
                        delta=f"Min: {psi_min:.1f} | Max: {psi_max:.1f}",
                        help="Tension matricielle moyenne"
                    )
                
                # Métriques supplémentaires
                st.markdown("#### 📈 Métriques supplémentaires")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_water = rollout['I'].sum() + rollout['R'].sum()
                    st.metric("💦 Eau totale (I+R)", f"{total_water:.1f} mm")
                with col2:
                    etc_total = rollout['ETc'].sum()
                    st.metric("🌱 ETc totale", f"{etc_total:.1f} mm")
                with col3:
                    water_balance = total_water - etc_total - rollout['D'].sum()
                    st.metric("⚖️ Bilan hydrique", f"{water_balance:.1f} mm")
                with col4:
                    days_stress = np.sum((rollout['psi'] < 20) | (rollout['psi'] > 60))
                    st.metric("⚠️ Jours hors confort", f"{days_stress} / {len(rollout['psi'])}")
                
                st.info("✅ Consultez l'onglet 'Visualisation' pour voir les graphiques détaillés.")

    # --------------------------------------------------------------------
    # ROBUSTESSE (LEVEL 3): test splits et sols multiples
    # --------------------------------------------------------------------
    robust_language = st.session_state.get("ui_language", "fr")
    tr_robust = lambda fr, en: fr if robust_language == "fr" else en
    st.markdown(
        f"### {tr_robust('🛡️ Évaluation de robustesse', '🛡️ Robustness evaluation')}"
    )
    st.markdown(
        "<div style='height:4px; width:100%; background:#2f80ed; border-radius:4px; margin:0 0 14px 0;'></div>",
        unsafe_allow_html=True,
    )
    with st.container():
        st.info(tr_robust(
            "ℹ️ Testez le modèle sur d'autres années/fichiers ERA5-Land et classes de sols.",
            "ℹ️ Test the model on other ERA5-Land years/files and soil classes.",
        ))
        default_test_files = "data/era5_land_fr_spring2025_all_nc3.nc"
        test_files_raw = st.text_area(
            tr_robust("Fichiers ERA5-Land de test (un chemin par ligne)", "Test ERA5-Land files (one path per line)"),
            value=default_test_files,
            help=tr_robust(
                "Chaque ligne = un fichier ERA5-Land (NetCDF) pour la validation hors période.",
                "Each line = one ERA5-Land (NetCDF) file for out-of-period validation.",
            ),
        )
        # Soil presets tuned closer to ERA5 swvl ranges
        soil_presets = {
            # Slightly drier FC and lower drainage to tighten runoff/soil moisture
            "sandy": dict(theta_fc=0.12, theta_wp=0.06, theta_s=0.42, Z_r=240.0, k_d=0.09),
            "loam": dict(theta_fc=0.15, theta_wp=0.08, theta_s=0.45, Z_r=260.0, k_d=0.11),
            "clay": dict(theta_fc=0.22, theta_wp=0.12, theta_s=0.50, Z_r=350.0, k_d=0.09),
        }
        selected_soils = st.multiselect(
            tr_robust("Classes de sol à tester", "Soil classes to test"),
            options=list(soil_presets.keys()),
            default=list(soil_presets.keys()),
        )
        has_model = model is not None
        if not has_model:
            st.info(tr_robust(
                "ℹ️ Chargez/entraînez un modèle (Scénario 2/3) avant de lancer la robustesse.",
                "ℹ️ Load/train a model (Scenario 2/3) before running robustness.",
            ))
        run_robust = st.button(
            tr_robust("Lancer l'évaluation robustesse (années/sols)", "Run robustness evaluation (years/soils)"),
            disabled=not has_model,
        )
        if run_robust and has_model:
            rows = []
            from pathlib import Path
            for fpath in [l.strip() for l in test_files_raw.splitlines() if l.strip()]:
                try:
                    era_bundle = load_data_for_simulation(
                        data_source="era5_land",
                        file_path=fpath,
                        resample_freq="1D",
                    )
                except Exception as exc:
                    st.warning(tr_robust(f"Impossible de charger {fpath}: {exc}", f"Unable to load {fpath}: {exc}"))
                    continue
                for soil_name in selected_soils:
                    soil_override = soil_presets.get(soil_name, {})
                    eval_kwargs = {
                        "model": model,
                        "season_length": len(np.asarray(era_bundle.get("rain", []))) or st.session_state.get("season_length", 120),
                        "max_irrigation": st.session_state.get("max_irrigation", 20.0),
                        "seed": eval_seed,
                        "soil_params": {**(st.session_state.get("soil_params") or {}), **soil_override},
                        "weather_params": st.session_state.get("weather_params"),
                        "data_source": "era5_land",
                        "data_path": fpath,
                        "era5_land_cfg": {"use_era5_land": True, "data_path": fpath, "resample_freq": "1D"},
                    }
                    if use_residual and st.session_state.scenario3_residual_model is not None:
                        eval_kwargs["residual_ode"] = st.session_state.scenario3_residual_model
                    rollout_rb = evaluate_episode(**eval_kwargs)
                    sim_outputs_rb = rollout_to_sim_outputs(rollout_rb)
                    results_rb = summarize_era5_land_validation(sim_outputs_rb, era_bundle)
                    for metric, vals in results_rb.items():
                        rows.append({
                            "test_file": Path(fpath).name,
                            "soil": soil_name,
                            "metric": metric,
                            "bias": vals.get("bias"),
                            "rmse": vals.get("rmse"),
                            "ubrmse": vals.get("ubrmse"),
                            "kge": vals.get("kge"),
                        })
            if rows:
                df_rb = pd.DataFrame(rows)
                # Pivot for quick scan on key metrics
                key_metrics = df_rb[df_rb["metric"].isin(["ETc", "fluxes/runoff", "soil_moisture/bucket_total_mm", "rain", "et0"])]
                pivot = key_metrics.pivot_table(
                    index=["test_file", "soil"],
                    columns="metric",
                    values=["bias", "rmse", "kge"],
                    aggfunc="first",
                )
                st.dataframe(pivot)
                csv_rb = df_rb.to_csv(index=False).encode("utf-8")
                st.download_button(
                    tr_robust("📥 Télécharger résultats robustesse (CSV)", "📥 Download robustness results (CSV)"),
                    csv_rb,
                    file_name="robustness_metrics.csv",
                    mime="text/csv",
                )
            else:
                st.info(tr_robust("Aucun résultat généré.", "No results generated."))

    # ========================================================================
    # ONGLET 8 : VISUALISATION
    # ========================================================================
    with tab6:
        language = st.session_state.get("ui_language", "fr")
        tr = lambda fr, en: fr if language == "fr" else en

        st.markdown(f'<h2 class="section-header">{tr("📊 Visualisation des résultats", "📊 Results visualization")}</h2>', unsafe_allow_html=True)

        viz_options = {
            "scenario1": tr("Scénario 1 (Règles simples)", "Scenario 1 (Simple rules)"),
            "scenario2": tr("Scénario 2 (PPO physique)", "Scenario 2 (PPO physical)"),
            "scenario3": tr("Scénario 3 (PPO hybride Neural ODE)", "Scenario 3 (PPO hybrid Neural ODE)"),
            "scenario3b": tr("Scénario 3b (Neural ODE continu)", "Scenario 3b (Continuous Neural ODE)"),
        }
        selected = st.radio(
            tr("Choisir le scénario", "Choose scenario"),
            options=list(viz_options.keys()),
            format_func=lambda k: viz_options[k],
            horizontal=True,
        )

        if selected == "scenario1":
            sim = st.session_state.get("scenario1_result")
            if sim is None:
                st.warning(tr("⚠️ Exécutez d'abord le Scénario 1.", "⚠️ Run Scenario 1 first."))
            else:
                fig = plot_scenario1(sim, language=language)
                st.pyplot(fig)
                df = pd.DataFrame({
                    "day": np.arange(len(sim["I"])),
                    "psi": sim["psi"][1:],
                    "S": sim["S"][1:],
                    "I": sim["I"],
                    "rain": sim["rain"],
                    "ETc": sim["ETc"],
                    "D": sim["D"],
                })
                st.dataframe(df, width='stretch', height=350)
        else:
            rollout_map = {
                "scenario2": st.session_state.get("scenario2_rollout"),
                "scenario3": st.session_state.get("scenario3_rollout"),
                "scenario3b": st.session_state.get("scenario3b_rollout"),
            }
            rollout = rollout_map[selected]
            if rollout is None:
                st.warning(tr("⚠️ Évaluez d'abord ce scénario.", "⚠️ Evaluate this scenario first."))
            else:
                fig = plot_episode_rollout(rollout, title=viz_options[selected], language=language)
                st.pyplot(fig)
                df = pd.DataFrame({
                    "day": np.arange(len(rollout["I"])),
                    "psi": rollout["psi"][1:],
                    "S": rollout["S"][1:],
                    "I": rollout["I"],
                    "R": rollout["R"],
                    "ETc": rollout["ETc"],
                    "D": rollout["D"],
                })
                st.dataframe(df, width='stretch', height=350)

    with tab7:
        language = st.session_state.get("ui_language", "fr")
        tr = lambda fr, en: fr if language == "fr" else en

        st.markdown(f'<h2 class="section-header">{tr("⚖️ Comparaison des scénarios", "⚖️ Scenario comparison")}</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            c1 = st.checkbox(tr("Scénario 1", "Scenario 1"), value=False)
        with col2:
            c2 = st.checkbox(tr("Scénario 2", "Scenario 2"), value=False)
        with col3:
            c3 = st.checkbox(tr("Scénario 3", "Scenario 3"), value=False)
        with col4:
            c3b = st.checkbox(tr("Scénario 3b", "Scenario 3b"), value=False)

        scenarios = []
        if c1 and st.session_state.get("scenario1_result") is not None:
            s = st.session_state["scenario1_result"]
            scenarios.append(("scenario1", s["I"], s["rain"], s["ETc"], s["D"], s["psi"][1:]))
        if c2 and st.session_state.get("scenario2_rollout") is not None:
            s = st.session_state["scenario2_rollout"]
            scenarios.append(("scenario2", s["I"], s["R"], s["ETc"], s["D"], s["psi"][1:]))
        if c3 and st.session_state.get("scenario3_rollout") is not None:
            s = st.session_state["scenario3_rollout"]
            scenarios.append(("scenario3", s["I"], s["R"], s["ETc"], s["D"], s["psi"][1:]))
        if c3b and st.session_state.get("scenario3b_rollout") is not None:
            s = st.session_state["scenario3b_rollout"]
            scenarios.append(("scenario3b", s["I"], s["R"], s["ETc"], s["D"], s["psi"][1:]))

        if len(scenarios) < 2:
            st.info(tr("Sélectionnez au moins 2 scénarios avec résultats.", "Select at least 2 scenarios with results."))
        else:
            scenario_labels = {
                "scenario1": tr("Scénario 1", "Scenario 1"),
                "scenario2": tr("Scénario 2", "Scenario 2"),
                "scenario3": tr("Scénario 3", "Scenario 3"),
                "scenario3b": tr("Scénario 3b", "Scenario 3b"),
            }

            rows = []
            series_by_scenario = {}
            for name, I, R, ETc, D, psi in scenarios:
                I = np.asarray(I, dtype=float)
                R = np.asarray(R, dtype=float)
                ETc = np.asarray(ETc, dtype=float)
                D = np.asarray(D, dtype=float)
                psi = np.asarray(psi, dtype=float)
                S = None
                if name == "scenario1":
                    s = st.session_state.get("scenario1_result")
                    if s is not None and "S" in s:
                        S = np.asarray(s["S"][1:], dtype=float)
                elif name == "scenario2":
                    s = st.session_state.get("scenario2_rollout")
                    if s is not None and "S" in s:
                        S = np.asarray(s["S"][1:], dtype=float)
                elif name == "scenario3":
                    s = st.session_state.get("scenario3_rollout")
                    if s is not None and "S" in s:
                        S = np.asarray(s["S"][1:], dtype=float)
                elif name == "scenario3b":
                    s = st.session_state.get("scenario3b_rollout")
                    if s is not None and "S" in s:
                        S = np.asarray(s["S"][1:], dtype=float)

                water_total = float(np.sum(I + R))
                etc_total = float(np.sum(ETc))
                irrig_total = float(np.sum(I))
                drainage_total = float(np.sum(D))
                eff = float(etc_total / water_total) if water_total > 0 else 0.0
                comfort = float(100.0 * np.mean((psi >= 20) & (psi <= 60)))
                psi_mean = float(np.mean(psi))
                rows.append({
                    tr("scénario", "scenario"): scenario_labels.get(name, name),
                    tr("irrigation_mm", "irrigation_mm"): irrig_total,
                    tr("pluie_mm", "rain_mm"): float(np.sum(R)),
                    tr("etc_mm", "etc_mm"): etc_total,
                    tr("drainage_mm", "drainage_mm"): drainage_total,
                    tr("psi_moyenne", "psi_mean"): psi_mean,
                    tr("jours_confort_pct", "comfort_days_pct"): comfort,
                    tr("efficacite_eau", "water_efficiency"): eff,
                })
                series_by_scenario[name] = {
                    "label": scenario_labels.get(name, name),
                    "I": I,
                    "R": R,
                    "ETc": ETc,
                    "D": D,
                    "psi": psi,
                    "S": S,
                    "water_total": water_total,
                    "etc_total": etc_total,
                    "irrig_total": irrig_total,
                    "drainage_total": drainage_total,
                    "eff": eff,
                    "comfort": comfort,
                    "psi_mean": psi_mean,
                }

            df_compare = pd.DataFrame(rows)
            st.dataframe(df_compare, width='stretch', hide_index=True)

            st.markdown(f"### {tr('📊 Graphiques comparatifs', '📊 Comparative charts')}")

            # 1) Tension matricielle
            st.markdown(f"#### 1️⃣ {tr('Tension matricielle ψ (cbar)', 'Matric tension ψ (cbar)')}")
            fig_psi, ax_psi = plt.subplots(figsize=(12, 5))
            for name, data in series_by_scenario.items():
                x = np.arange(len(data["psi"]))
                ax_psi.plot(x, data["psi"], linewidth=2, label=data["label"])
            ax_psi.axhspan(20, 60, color="green", alpha=0.12, label=tr("Zone optimale (20-60 cbar)", "Optimal zone (20-60 cbar)"))
            ax_psi.set_title(tr("Comparaison des tensions matricielles", "Matric tension comparison"), fontsize=16, fontweight="bold")
            ax_psi.set_xlabel(tr("Jour", "Day"))
            ax_psi.set_ylabel(tr("Tension ψ (cbar)", "Tension ψ (cbar)"))
            ax_psi.grid(True, alpha=0.25)
            ax_psi.legend()
            st.pyplot(fig_psi)

            # 2) Réserve en eau
            st.markdown(f"#### 2️⃣ {tr('Réserve en eau S (mm)', 'Soil water storage S (mm)')}")
            fig_s, ax_s = plt.subplots(figsize=(12, 5))
            plotted_s = False
            for name, data in series_by_scenario.items():
                if data["S"] is None:
                    continue
                x = np.arange(len(data["S"]))
                ax_s.plot(x, data["S"], linewidth=2, label=data["label"])
                plotted_s = True
            soil_params = st.session_state.get("soil_params", {}) or {}
            zr = float(soil_params.get("Z_r", 300.0))
            s_fc = float(soil_params.get("theta_fc", 0.22)) * zr
            s_wp = float(soil_params.get("theta_wp", 0.12)) * zr
            ax_s.axhline(s_fc, color="gray", linestyle="--", linewidth=1.8, label="S_fc")
            ax_s.axhline(s_wp, color="#b5651d", linestyle="--", linewidth=1.8, label="S_wp")
            ax_s.set_title(tr("Comparaison des réserves en eau", "Soil storage comparison"), fontsize=16, fontweight="bold")
            ax_s.set_xlabel(tr("Jour", "Day"))
            ax_s.set_ylabel(tr("Réserve S (mm)", "Storage S (mm)"))
            ax_s.grid(True, alpha=0.25)
            ax_s.legend()
            if plotted_s:
                st.pyplot(fig_s)
            else:
                st.info(tr("Aucune série S disponible pour les scénarios sélectionnés.", "No S series available for selected scenarios."))

            # 3) Irrigation et pluie
            st.markdown(f"#### 3️⃣ {tr('Irrigation et pluie (mm)', 'Irrigation and rain (mm)')}")
            summary_df = pd.DataFrame([
                {
                    tr("Scénario", "Scenario"): d["label"],
                    tr("Irrigation totale", "Total irrigation"): d["irrig_total"],
                    tr("Pluie totale", "Total rain"): float(np.sum(d["R"])),
                }
                for d in series_by_scenario.values()
            ])
            fig_ir, ax_ir = plt.subplots(figsize=(12, 5))
            x = np.arange(len(summary_df))
            w = 0.4
            irr_col = tr("Irrigation totale", "Total irrigation")
            rain_col = tr("Pluie totale", "Total rain")
            ax_ir.bar(x - w / 2, summary_df[irr_col], width=w, color="#4c9fd1", label=irr_col)
            ax_ir.bar(x + w / 2, summary_df[rain_col], width=w, color="#58d1d8", label=rain_col)
            for i, v in enumerate(summary_df[irr_col]):
                ax_ir.text(i - w / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
            for i, v in enumerate(summary_df[rain_col]):
                ax_ir.text(i + w / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
            ax_ir.set_xticks(x)
            ax_ir.set_xticklabels(summary_df[tr("Scénario", "Scenario")], rotation=20)
            ax_ir.set_ylabel(tr("Volume d'eau (mm)", "Water volume (mm)"))
            ax_ir.set_title(tr("Comparaison des volumes d'eau (irrigation + pluie)", "Water volumes comparison (irrigation + rain)"), fontsize=16, fontweight="bold")
            ax_ir.grid(True, axis="y", alpha=0.25)
            ax_ir.legend()
            st.pyplot(fig_ir)

            # 4) Métriques de performance
            st.markdown(f"#### 4️⃣ {tr('Métriques de performance', 'Performance metrics')}")
            labels = [d["label"] for d in series_by_scenario.values()]
            drainage_vals = [d["drainage_total"] for d in series_by_scenario.values()]
            comfort_vals = [d["comfort"] for d in series_by_scenario.values()]
            psi_mean_vals = [d["psi_mean"] for d in series_by_scenario.values()]
            eff_vals = [d["eff"] for d in series_by_scenario.values()]
            fig_perf, axs = plt.subplots(2, 2, figsize=(12, 8))

            axs[0, 0].bar(labels, drainage_vals, color="#a78ac3")
            axs[0, 0].set_title(tr("Drainage total", "Total drainage"), fontweight="bold")
            axs[0, 0].set_ylabel("mm")
            axs[0, 0].grid(True, axis="y", alpha=0.25)
            for i, v in enumerate(drainage_vals):
                axs[0, 0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

            axs[0, 1].bar(labels, comfort_vals, color="#69b96b")
            axs[0, 1].axhline(80, color="orange", linestyle="--", label=tr("Objectif 80%", "80% target"))
            axs[0, 1].set_title(tr("Jours en zone optimale (20-60 cbar)", "Days in optimal zone (20-60 cbar)"), fontweight="bold")
            axs[0, 1].set_ylabel("%")
            axs[0, 1].grid(True, axis="y", alpha=0.25)
            axs[0, 1].legend()
            for i, v in enumerate(comfort_vals):
                axs[0, 1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

            axs[1, 0].bar(labels, psi_mean_vals, color="#dd6666")
            axs[1, 0].axhspan(20, 60, color="green", alpha=0.2, label=tr("Zone optimale (20-60 cbar)", "Optimal zone (20-60 cbar)"))
            axs[1, 0].set_title(tr("Tension matricielle moyenne", "Average matric tension"), fontweight="bold")
            axs[1, 0].set_ylabel("cbar")
            axs[1, 0].grid(True, axis="y", alpha=0.25)
            axs[1, 0].legend()
            for i, v in enumerate(psi_mean_vals):
                axs[1, 0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

            axs[1, 1].bar(labels, eff_vals, color="#f2a04f")
            axs[1, 1].set_title(tr("Efficacité de l'eau (ETc / (I+R))", "Water efficiency (ETc / (I+R))"), fontweight="bold")
            axs[1, 1].set_ylabel(tr("Efficacité", "Efficiency"))
            axs[1, 1].grid(True, axis="y", alpha=0.25)
            for i, v in enumerate(eff_vals):
                axs[1, 1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

            for ax in axs.flat:
                ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            st.pyplot(fig_perf)

            st.markdown("---")
            best_comfort = max(series_by_scenario.values(), key=lambda d: d["comfort"])
            best_eff = max(series_by_scenario.values(), key=lambda d: d["eff"])
            best_drain = min(series_by_scenario.values(), key=lambda d: d["drainage_total"])
            best_irrig = min(series_by_scenario.values(), key=lambda d: d["irrig_total"])
            lbl_best_comfort = tr("Meilleur maintien en zone de confort", "Best comfort-zone performance")
            lbl_best_eff = tr("Meilleure efficacité de l'eau", "Best water efficiency")
            lbl_best_drain = tr("Drainage le plus faible", "Lowest drainage")
            lbl_best_irrig = tr("Consommation d'eau la plus faible", "Lowest irrigation use")
            st.markdown(f"### {tr('📝 Analyse comparative', '📝 Comparative analysis')}")
            st.markdown(
                "\n".join(
                    [
                        f"- **{lbl_best_comfort}**: {best_comfort['label']} ({best_comfort['comfort']:.1f}%)",
                        f"- **{lbl_best_eff}**: {best_eff['label']} ({best_eff['eff']:.3f})",
                        f"- **{lbl_best_drain}**: {best_drain['label']} ({best_drain['drainage_total']:.1f} mm)",
                        f"- **{lbl_best_irrig}**: {best_irrig['label']} ({best_irrig['irrig_total']:.1f} mm)",
                    ]
                )
            )

            csv = df_compare.to_csv(index=False)
            st.download_button(
                label=tr("📥 Télécharger le tableau comparatif (CSV)", "📥 Download comparison table (CSV)"),
                data=csv,
                file_name="scenario_comparison.csv",
                mime="text/csv",
            )

    with tab8:
        language = st.session_state.get("ui_language", "fr")
        tr = lambda fr, en: fr if language == "fr" else en

        st.markdown(f'<h2 class="section-header">{tr("🧪 Validation ERA5-Land (avancée)", "🧪 Advanced ERA5-Land validation")}</h2>', unsafe_allow_html=True)
        st.markdown(
            "<div style='height:4px; width:100%; background:#2f80ed; border-radius:4px; margin:0 0 14px 0;'></div>",
            unsafe_allow_html=True,
        )
        st.info(
            tr(
                "ℹ️ Cette section sert à comparer un scénario déjà simulé/évalué avec un fichier ERA5-Land de référence afin d'obtenir des métriques (bias, RMSE, KGE).",
                "ℹ️ Use this section to compare an already simulated/evaluated scenario against an ERA5-Land reference file and compute validation metrics (bias, RMSE, KGE).",
            )
        )

        model_option = st.radio(
            "",
            options=["scenario1", "scenario2", "scenario3", "scenario3b"],
            horizontal=True,
            label_visibility="collapsed",
        )
        era5_path = st.text_input(
            tr("Chemin fichier ERA5-Land (.nc)", "ERA5-Land file path (.nc)"),
            value=st.session_state.get("era5_path", ""),
        )

        if st.button(tr("Lancer validation", "Run validation"), type="primary"):
            if not era5_path:
                st.error(tr("Veuillez renseigner un fichier ERA5-Land.", "Please provide an ERA5-Land file."))
            else:
                try:
                    era_bundle = load_data_for_simulation(
                        data_source="era5_land",
                        file_path=era5_path,
                        resample_freq="1D",
                    )

                    if model_option == "scenario1":
                        sim = st.session_state.get("scenario1_result")
                        if sim is None:
                            st.error(tr("Scénario 1 non disponible.", "Scenario 1 not available."))
                            st.stop()
                        sim_outputs = {
                            "rain": sim.get("rain"),
                            "ETc": sim.get("ETc"),
                            "psi": sim.get("psi"),
                            "S": sim.get("S"),
                            "fluxes": {"runoff": sim.get("D")},
                            "soil_moisture_layers": {"bucket_total_mm": sim.get("S")},
                        }
                    else:
                        key = {
                            "scenario2": "scenario2_rollout",
                            "scenario3": "scenario3_rollout",
                            "scenario3b": "scenario3b_rollout",
                        }[model_option]
                        rollout = st.session_state.get(key)
                        if rollout is None:
                            st.error(tr("Aucun rollout disponible pour ce scénario.", "No rollout available for this scenario."))
                            st.stop()
                        sim_outputs = rollout_to_sim_outputs(rollout)

                    results = summarize_era5_land_validation(sim_outputs, era_bundle)
                    st.markdown(format_validation_table(results))
                    st.json(results)
                except Exception as exc:
                    st.error(f"Validation ERA5-Land échouée: {exc}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#6b7280; font-size:0.9rem;'>"
        "EIPEx - Efficient Irrigation Policy Explorer • "
        "© 2025 Raymond houé Ngouna (raymond.houe-ngouna@uttop.fr) • Powered by Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()
