"""
Utilitaires pour la génération de graphiques matplotlib/seaborn.

Ce module centralise les fonctions répétitives pour :
- Création de figures standardisées
- Graphiques d'épisodes (tension, réserve, irrigation, flux)
- Styles et couleurs communs
- Fonctions helper pour les sous-graphiques
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple


# ============================================================================
# TRADUCTIONS
# ============================================================================

_LABELS = {
    "fr": {
        "day": "Jour",
        "comfort_band": "Bande confort ({low:.0f}–{high:.0f})",
        "tension_label": "ψ (cbar)",
        "tension_ylabel": "ψ (cbar)",
        "reserve_label": "S (mm)",
        "reserve_ylabel": "S (mm)",
        "irrigation_label": "Irrigation (mm)",
        "rain_label": "Pluie (mm)",
        "irrigation_rain_ylabel": "I / R (mm)",
        "ETc_label": "ETc (mm)",
        "drainage_label": "Drainage (mm)",
        "flux_ylabel": "Flux (mm)",
        "ppo_eval_title": "Évaluation du modèle PPO",
        "scenario1_title": "Scénario 1 — Modèle physique + règles simples",
        "threshold_label": "Seuil ({threshold:.0f} cbar)",
        "tension_target": "Tension cible ({target:.0f} cbar)",
        "general_opt_zone": "Zone optimale générale",
        "comparison_tension_title": "Comparaison de la tension matricielle",
        "comparison_reserve_title": "Comparaison de la réserve en eau",
        "comparison_irrigation_title": "Comparaison de l'irrigation",
    },
    "en": {
        "day": "Day",
        "comfort_band": "Comfort band ({low:.0f}–{high:.0f})",
        "tension_label": "ψ (cbar)",
        "tension_ylabel": "ψ (cbar)",
        "reserve_label": "S (mm)",
        "reserve_ylabel": "S (mm)",
        "irrigation_label": "Irrigation (mm)",
        "rain_label": "Rain (mm)",
        "irrigation_rain_ylabel": "I / R (mm)",
        "ETc_label": "ETc (mm)",
        "drainage_label": "Drainage (mm)",
        "flux_ylabel": "Flows (mm)",
        "ppo_eval_title": "PPO model evaluation",
        "scenario1_title": "Scenario 1 — Physical model + simple rules",
        "threshold_label": "Threshold ({threshold:.0f} cbar)",
        "tension_target": "Target tension ({target:.0f} cbar)",
        "general_opt_zone": "General optimal zone",
        "comparison_tension_title": "Tension comparison",
        "comparison_reserve_title": "Soil moisture comparison",
        "comparison_irrigation_title": "Irrigation comparison",
    },
}


def _t(key: str, language: str = "fr", **fmt) -> str:
    """Helper de traduction avec fallback."""
    lang = language.lower()
    if lang not in _LABELS:
        lang = "fr"
    template = _LABELS[lang].get(key, _LABELS["fr"].get(key, key))
    try:
        return template.format(**fmt)
    except Exception:
        return template


# ============================================================================
# CONFIGURATION MATPLOTLIB
# ============================================================================

def configure_matplotlib():
    """Configure matplotlib pour des graphiques de meilleure qualité."""
    plt.rcParams['figure.max_open_warning'] = 0
    plt.rcParams['figure.dpi'] = 100


# ============================================================================
# CRÉATION DE FIGURES STANDARDISÉES
# ============================================================================

def create_4panel_figure(title: str, figsize: Tuple[int, int] = (16, 12)) -> Tuple[plt.Figure, np.ndarray]:
    """
    Crée une figure avec 4 sous-graphiques empilés verticalement.
    
    Args:
        title: Titre de la figure
        figsize: Taille de la figure (largeur, hauteur)
        
    Returns:
        tuple: (figure, array d'axes)
    """
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, y=1.02, fontsize=16)
    return fig, axs


def create_2x2_figure(title: str, figsize: Tuple[int, int] = (14, 10)) -> Tuple[plt.Figure, np.ndarray]:
    """
    Crée une figure avec 4 sous-graphiques en grille 2x2.
    
    Args:
        title: Titre de la figure
        figsize: Taille de la figure (largeur, hauteur)
        
    Returns:
        tuple: (figure, array d'axes)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig, axes


# ============================================================================
# GRAPHIQUES STANDARDISÉS
# ============================================================================

def plot_tension_subplot(
    ax: plt.Axes,
    t: np.ndarray,
    psi: np.ndarray,
    psi_low: float = 20.0,
    psi_high: float = 60.0,
    label: Optional[str] = None,
    color: str = "tab:red",
    show_zone: bool = True,
    language: str = "fr"
):
    """
    Trace la tension matricielle ψ avec zone de confort.
    
    Args:
        ax: Axe matplotlib
        t: Axe temporel
        psi: Valeurs de tension
        psi_low: Limite inférieure de la zone de confort
        psi_high: Limite supérieure de la zone de confort
        label: Label pour la courbe
        color: Couleur de la courbe
        show_zone: Afficher la zone de confort
        language: Langue pour les labels ("fr" ou "en")
    """
    label = label or _t("tension_label", language)
    ax.plot(t, psi, label=label, color=color, linewidth=2)
    
    if show_zone:
        ax.axhspan(psi_low, psi_high, color="tab:green", alpha=0.1, 
                   label=_t("comfort_band", language, low=psi_low, high=psi_high))
    
    ax.set_ylabel(_t("tension_ylabel", language), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_reserve_subplot(
    ax: plt.Axes,
    t: np.ndarray,
    S: np.ndarray,
    S_fc: float,
    S_wp: float,
    label: Optional[str] = None,
    color: str = "tab:blue",
    language: str = "fr"
):
    """
    Trace la réserve en eau S avec seuils caractéristiques.
    
    Args:
        ax: Axe matplotlib
        t: Axe temporel
        S: Valeurs de réserve
        S_fc: Capacité au champ
        S_wp: Point de flétrissement
        label: Label pour la courbe
        color: Couleur de la courbe
        language: Langue pour les labels ("fr" ou "en")
    """
    label = label or _t("reserve_label", language)
    ax.plot(t, S, label=label, color=color, linewidth=2)
    ax.axhline(S_fc, ls="--", color="gray", alpha=0.7, label="S_fc")
    ax.axhline(S_wp, ls="--", color="brown", alpha=0.7, label="S_wp")
    ax.set_ylabel(_t("reserve_ylabel", language), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_irrigation_rain_subplot(
    ax: plt.Axes,
    t: np.ndarray,
    I: np.ndarray,
    R: np.ndarray,
    I_label: Optional[str] = None,
    R_label: Optional[str] = None,
    I_color: str = "tab:blue",
    R_color: str = "tab:cyan",
    language: str = "fr"
):
    """
    Trace l'irrigation et la pluie en barres empilées.
    
    Args:
        ax: Axe matplotlib
        t: Axe temporel
        I: Valeurs d'irrigation
        R: Valeurs de pluie
        I_label: Label pour l'irrigation
        R_label: Label pour la pluie
        I_color: Couleur pour l'irrigation
        R_color: Couleur pour la pluie
        language: Langue pour les labels ("fr" ou "en")
    """
    I_label = I_label or _t("irrigation_label", language)
    R_label = R_label or _t("rain_label", language)
    ax.bar(t, I, width=0.8, label=I_label, color=I_color, alpha=0.7)
    ax.bar(t, R, width=0.8, label=R_label, color=R_color, alpha=0.5, bottom=I)
    ax.set_ylabel(_t("irrigation_rain_ylabel", language), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_flux_subplot(
    ax: plt.Axes,
    t: np.ndarray,
    ETc: np.ndarray,
    D: np.ndarray,
    ETc_label: Optional[str] = None,
    D_label: Optional[str] = None,
    ETc_color: str = "tab:orange",
    D_color: str = "tab:purple",
    language: str = "fr"
):
    """
    Trace les flux ETc et drainage.
    
    Args:
        ax: Axe matplotlib
        t: Axe temporel
        ETc: Valeurs d'évapotranspiration
        D: Valeurs de drainage
        ETc_label: Label pour ETc
        D_label: Label pour le drainage
        ETc_color: Couleur pour ETc
        D_color: Couleur pour le drainage
        language: Langue pour les labels ("fr" ou "en")
    """
    ETc_label = ETc_label or _t("ETc_label", language)
    D_label = D_label or _t("drainage_label", language)
    ax.plot(t, ETc, label=ETc_label, color=ETc_color, linewidth=2)
    ax.plot(t, D, label=D_label, color=D_color, linewidth=2)
    ax.set_ylabel(_t("flux_ylabel", language), fontsize=12)
    ax.set_xlabel(_t("day", language), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()


# ============================================================================
# FONCTION COMPLÈTE POUR PLOT D'ÉPISODE
# ============================================================================

def plot_episode_rollout(
    rollout: Dict[str, Any],
    title: Optional[str] = None,
    language: str = "fr"
) -> plt.Figure:
    """
    Visualise les résultats d'un épisode d'évaluation du modèle PPO.
    
    GRAPHIQUES :
    1. Évolution de la tension matricielle ψ avec zone de confort
    2. Évolution de la réserve en eau S avec seuils caractéristiques
    3. Irrigation et pluie (barres empilées)
    4. Évapotranspiration ETc et drainage D
    
    Args:
        rollout: Dictionnaire retourné par evaluate_episode()
        title: Titre du graphique principal
        language: Langue pour les labels ("fr" ou "en")
        
    Returns:
        matplotlib.figure.Figure: Figure contenant les 4 graphiques
    """
    psi = rollout["psi"]
    S = rollout["S"]
    I = rollout["I"]
    R = rollout["R"]
    ETc = rollout["ETc"]
    D = rollout["D"]
    season_length = rollout["env_summary"]["season_length"]
    S_fc = rollout["env_summary"]["S_fc"]
    S_wp = rollout["env_summary"]["S_wp"]

    # Préparation des axes temporels
    t_Spsi = np.arange(season_length + 1)  # Pour ψ et S (état initial inclus)
    t_flux = np.arange(season_length)      # Pour I, R, ETc, D (pas d'état initial)

    # Création de la figure avec 4 sous-graphiques empilés verticalement
    fig_title = title or _t("ppo_eval_title", language)
    fig, axs = create_4panel_figure(fig_title)

    # Graphique 1 : Tension matricielle ψ
    plot_tension_subplot(axs[0], t_Spsi, psi, language=language)

    # Graphique 2 : Réserve en eau S
    plot_reserve_subplot(axs[1], t_Spsi, S, S_fc, S_wp, language=language)

    # Graphique 3 : Irrigation / Pluie
    plot_irrigation_rain_subplot(axs[2], t_flux, I, R, language=language)

    # Graphique 4 : Évapotranspiration ETc et drainage D
    plot_flux_subplot(axs[3], t_flux, ETc, D, language=language)

    plt.tight_layout()
    return fig


def plot_scenario1(
    sim: Dict[str, Any],
    title: Optional[str] = None,
    language: str = "fr"
) -> plt.Figure:
    """
    Visualise les résultats du scénario 1 sur 4 graphiques.
    
    Args:
        sim: Dictionnaire retourné par simulate_scenario1()
        title: Titre du graphique principal
        language: Langue pour les labels ("fr" ou "en")
        
    Returns:
        matplotlib.figure.Figure: Figure contenant les 4 graphiques
    """
    T = sim["params"]["T"]
    t = np.arange(T + 1)  # Pour ψ et S (état initial inclus)
    tt = np.arange(T)     # Pour I, R, ETc, D (pas d'état initial)
    soil = sim["soil"]

    # Création de la figure avec 4 sous-graphiques empilés verticalement
    fig_title = title or _t("scenario1_title", language)
    fig, axs = create_4panel_figure(fig_title)

    # Graphique 1 : Tension matricielle ψ
    # Adaptation de l'affichage selon la règle utilisée
    rule_name = sim.get("params", {}).get("rule_fn", "")
    rule_kwargs = sim.get("params", {}).get("rule_kwargs", {})
    
    if rule_name == "rule_bande_confort":
        psi_low = rule_kwargs.get("psi_low", 20.0)
        psi_high = rule_kwargs.get("psi_high", 60.0)
        plot_tension_subplot(axs[0], t, sim["psi"], psi_low, psi_high, language=language)
    elif rule_name == "rule_seuil_unique":
        threshold = rule_kwargs.get("threshold_cbar", 80.0)
        axs[0].plot(t, sim["psi"], label=_t("tension_label", language), color="tab:red", linewidth=2)
        axs[0].axhline(threshold, color="tab:orange", linestyle="--", linewidth=2, 
                       label=_t("threshold_label", language, threshold=threshold))
        axs[0].axhspan(20, 60, color="lightgray", alpha=0.05, label=_t("general_opt_zone", language))
        axs[0].set_ylabel(_t("tension_ylabel", language), fontsize=12)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
    elif rule_name == "rule_proportionnelle":
        psi_target = rule_kwargs.get("psi_target", 40.0)
        axs[0].plot(t, sim["psi"], label=_t("tension_label", language), color="tab:red", linewidth=2)
        axs[0].axhline(psi_target, color="tab:blue", linestyle="--", linewidth=2, 
                       label=_t("tension_target", language, target=psi_target))
        axs[0].axhspan(20, 60, color="lightgray", alpha=0.05, label=_t("general_opt_zone", language))
        axs[0].set_ylabel(_t("tension_ylabel", language), fontsize=12)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
    else:
        plot_tension_subplot(axs[0], t, sim["psi"], language=language)

    # Graphique 2 : Réserve en eau S
    plot_reserve_subplot(axs[1], t, sim["S"], soil.S_fc, soil.S_wp, language=language)

    # Graphique 3 : Irrigation / Pluie
    plot_irrigation_rain_subplot(axs[2], tt, sim["I"], sim["rain"], language=language)

    # Graphique 4 : Évapotranspiration ETc et drainage D
    plot_flux_subplot(axs[3], tt, sim["ETc"], sim["D"], language=language)

    plt.tight_layout()
    return fig


# ============================================================================
# GRAPHIQUES POUR COMPARAISON
# ============================================================================

def plot_comparison_tension(
    ax: plt.Axes,
    scenarios_data: Dict[str, Dict[str, np.ndarray]],
    season_length: int,
    language: str = "fr"
):
    """
    Trace la comparaison de la tension pour plusieurs scénarios.
    
    Args:
        ax: Axe matplotlib
        scenarios_data: Dictionnaire {nom_scenario: {psi: array, ...}}
        season_length: Longueur de la saison
        language: Langue pour les labels ("fr" ou "en")
    """
    t = np.arange(season_length + 1)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    
    for idx, (scenario_name, data) in enumerate(scenarios_data.items()):
        if "psi" in data:
            ax.plot(t, data["psi"], label=scenario_name, color=colors[idx % len(colors)], linewidth=2)
    
    ax.axhspan(20, 60, color="tab:green", alpha=0.1, label=_t("comfort_band", language, low=20, high=60))
    ax.set_ylabel(_t("tension_ylabel", language), fontsize=12)
    ax.set_title(_t("comparison_tension_title", language), fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_comparison_reserve(
    ax: plt.Axes,
    scenarios_data: Dict[str, Dict[str, np.ndarray]],
    season_length: int,
    S_fc: float,
    S_wp: float,
    language: str = "fr"
):
    """
    Trace la comparaison de la réserve pour plusieurs scénarios.
    
    Args:
        ax: Axe matplotlib
        scenarios_data: Dictionnaire {nom_scenario: {S: array, ...}}
        season_length: Longueur de la saison
        S_fc: Capacité au champ
        S_wp: Point de flétrissement
        language: Langue pour les labels ("fr" ou "en")
    """
    t = np.arange(season_length + 1)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    
    for idx, (scenario_name, data) in enumerate(scenarios_data.items()):
        if "S" in data:
            ax.plot(t, data["S"], label=scenario_name, color=colors[idx % len(colors)], linewidth=2)
    
    ax.axhline(S_fc, ls="--", color="gray", alpha=0.7, label="S_fc")
    ax.axhline(S_wp, ls="--", color="brown", alpha=0.7, label="S_wp")
    ax.set_ylabel(_t("reserve_ylabel", language), fontsize=12)
    ax.set_title(_t("comparison_reserve_title", language), fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_comparison_irrigation(
    ax: plt.Axes,
    scenarios_data: Dict[str, Dict[str, np.ndarray]],
    season_length: int,
    language: str = "fr"
):
    """
    Trace la comparaison de l'irrigation pour plusieurs scénarios.
    
    Args:
        ax: Axe matplotlib
        scenarios_data: Dictionnaire {nom_scenario: {I: array, ...}}
        season_length: Longueur de la saison
        language: Langue pour les labels ("fr" ou "en")
    """
    t = np.arange(season_length)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    
    for idx, (scenario_name, data) in enumerate(scenarios_data.items()):
        if "I" in data:
            ax.plot(t, data["I"], label=scenario_name, color=colors[idx % len(colors)], 
                   linewidth=2, marker='o', markersize=3, alpha=0.7)
    
    ax.set_ylabel(_t("irrigation_label", language), fontsize=12)
    ax.set_xlabel(_t("day", language), fontsize=12)
    ax.set_title(_t("comparison_irrigation_title", language), fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
