"""
Utilitaires pour l'interface utilisateur Streamlit.

Ce module centralise les fonctions répétitives pour :
- Styles CSS/HTML
- Barres de progression avec callbacks
- Messages formatés (info, success, warning, error)
- Formatage de temps
- Sections d'entraînement
"""

import streamlit as st
import time
from typing import Optional, Callable, Any

# Import conditionnel pour BaseCallback
try:
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
    BASE_CALLBACK_AVAILABLE = True
except ImportError:
    BASE_CALLBACK_AVAILABLE = False
    BaseCallback = None  # type: ignore


# ============================================================================
# TRADUCTIONS UI
# ============================================================================

_UI_TEXT = {
    "fr": {
        "progress_label": "Progression",
        "steps_label": "pas",
        "elapsed": "⏱️ Temps écoulé",
        "eta": "⏳ Temps restant",
        "train_done_status": "✅ Entraînement terminé:",
        "train_done_msg": "✅ Entraînement terminé en {time} ! Modèle sauvegardé.",
        "metrics_title": "### 📊 Métriques d'entraînement",
        "metric_reward": "Récompense moyenne",
        "metric_len": "Longueur moyenne épisode",
        "metric_loss": "Perte de politique",
        "metrics_empty": "Aucune métrique disponible pour cet entraînement.",
        "no_metrics": "⚠️ Aucune métrique disponible",
    },
    "en": {
        "progress_label": "Progress",
        "steps_label": "steps",
        "elapsed": "⏱️ Elapsed time",
        "eta": "⏳ Remaining time",
        "train_done_status": "✅ Training completed:",
        "train_done_msg": "✅ Training finished in {time}! Model saved.",
        "metrics_title": "### 📊 Training metrics",
        "metric_reward": "Average reward",
        "metric_len": "Average episode length",
        "metric_loss": "Policy loss",
        "metrics_empty": "No metrics available for this training.",
        "no_metrics": "⚠️ No metrics available",
    },
}


def _ui_t(key: str, language: str = "fr") -> str:
    """Small helper to fetch translated UI strings with fallback."""
    lang = language.lower()
    if lang not in _UI_TEXT:
        lang = "fr"
    return _UI_TEXT[lang].get(key, _UI_TEXT["fr"].get(key, key))


# ============================================================================
# STYLES CSS
# ============================================================================

def get_custom_css() -> str:
    """
    Retourne le CSS personnalisé pour l'application Streamlit.
    
    Returns:
        str: CSS personnalisé
    """
    return """
    <style>
    /* Forcer une police globale pour Streamlit */
    html, body, [class*="css"]  {
        font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif !important;
        font-size: 17px !important;
        line-height: 1.55;
    }
    /* Maximiser la largeur du contenu principal */
    .main .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Réduire la largeur de la sidebar pour plus d'espace */
    .css-1d391kg {
        width: 20rem;
    }
    
    /* Améliorer l'espacement */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Agrandir les colonnes */
    .stColumn {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* Améliorer l'affichage des graphiques */
    .stPlotlyChart, .stImage {
        width: 100%;
    }
    
    /* Agrandir les métriques */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    /* Espacement des colonnes de métriques */
    .stMetric {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Agrandir la barre de progression */
    .stProgress > div > div > div {
        height: 30px !important;
        background-color: #e0e0e0 !important;
    }
    
    .stProgress > div > div > div > div {
        height: 30px !important;
        background: linear-gradient(90deg, #1f77b4 0%, #3498db 100%) !important;
        transition: width 0.3s ease !important;
    }
    
    /* Améliorer l'affichage du texte de progression */
    .progress-status {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Zone d'entraînement */
    .training-section {
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    </style>
    """


def apply_custom_css():
    """Applique le CSS personnalisé à l'application Streamlit."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


# ============================================================================
# FORMATAGE DE TEMPS
# ============================================================================

def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes en une chaîne lisible (heures:minutes:secondes).
    
    Args:
        seconds: Temps en secondes
        
    Returns:
        str: Temps formaté (ex: "1h 23m 45s", "23m 45s", "45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ============================================================================
# MESSAGES FORMATÉS
# ============================================================================

def display_info_box(title: str, content: str, color: str = "#2196F3"):
    """
    Affiche une boîte d'information stylisée.
    
    Args:
        title: Titre de la boîte
        content: Contenu de la boîte
        color: Couleur de la bordure (hex)
    """
    st.markdown(
        f"""
        <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 4px solid {color}; margin: 15px 0;">
        <h4 style="margin-top: 0; color: {color};">{title}</h4>
        <p style="margin-bottom: 0;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_warning_box(message: str):
    """
    Affiche une boîte d'avertissement stylisée.
    
    Args:
        message: Message d'avertissement
    """
    st.markdown(
        f"""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #ffc107;">
        <strong>⚠️ Attention:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_info_small(message: str):
    """
    Affiche un petit message d'information stylisé.
    
    Args:
        message: Message d'information
    """
    st.markdown(
        f"""
        <div style="background-color: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
        {message}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# BARRES DE PROGRESSION
# ============================================================================

class ProgressCallback:
    """
    Callback pour suivre la progression de l'entraînement dans Streamlit.
    
    PRINCIPE :
    Hérite de BaseCallback de stable-baselines3. Appelé à chaque pas
    de simulation pour mettre à jour l'interface utilisateur avec :
    - Barre de progression
    - Nombre de pas effectués
    - Temps écoulé et temps restant estimé
    """
    
    def __init__(
        self,
        progress_bar: st.progress,
        status_text: st.empty,
        time_elapsed: st.empty,
        eta_text: st.empty,
        total_timesteps: int,
        start_time: float,
        language: str = "fr"
    ):
        """
        Initialise le callback de progression.
        
        Args:
            progress_bar: Widget Streamlit pour la barre de progression
            status_text: Widget pour le texte de statut
            time_elapsed: Widget pour le temps écoulé
            eta_text: Widget pour le temps restant
            total_timesteps: Nombre total de pas d'entraînement
            start_time: Temps de début (timestamp)
            language: Langue pour l'affichage ("fr" ou "en")
        """
        if BASE_CALLBACK_AVAILABLE and BaseCallback is not None:
            super().__init__()
        self.num_timesteps = 0
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.time_elapsed = time_elapsed
        self.eta_text = eta_text
        self.total_timesteps = total_timesteps
        self.start_time = start_time
        self.language = language
    
    def _on_step(self) -> bool:
        """
        Appelée à chaque pas de simulation.
        
        Calcule et affiche :
        - Progression en pourcentage
        - Temps écoulé depuis le début
        - Temps restant estimé (ETA) basé sur la vitesse actuelle
        
        Returns:
            bool: True pour continuer l'entraînement
        """
        # Mise à jour de la barre de progression (clipper entre 0.0 et 1.0)
        progress = min(1.0, max(0.0, self.num_timesteps / self.total_timesteps))
        self.progress_bar.progress(progress)
        
        # Calcul du temps écoulé et estimé
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if progress > 0:
            # Estimation du temps total basée sur la progression actuelle
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            
            # Mise à jour du texte de statut avec formatage
            self.status_text.markdown(
                f"<div class='progress-status'>"
                f"<strong>{_ui_t('progress_label', self.language)}:</strong> {self.num_timesteps:,} / {self.total_timesteps:,} {_ui_t('steps_label', self.language)} "
                f"<strong>({progress*100:.1f}%)</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Formatage du temps écoulé
            elapsed_str = format_time(elapsed)
            self.time_elapsed.metric(_ui_t("elapsed", self.language), elapsed_str)
            
            # Formatage du temps restant estimé (ETA)
            if remaining > 0:
                eta_str = format_time(remaining)
                self.eta_text.metric(_ui_t("eta", self.language), eta_str)
        
        return True  # Continue l'entraînement


class MetricsCallback:
    """
    Callback pour collecter les métriques d'entraînement.
    """
    
    def __init__(self):
        if BASE_CALLBACK_AVAILABLE and BaseCallback is not None:
            super().__init__()
        self.metrics_history = []
    
    def _on_step(self) -> bool:
        """Collecte les métriques à chaque log."""
        if self.logger is not None:
            metrics = {}
            if hasattr(self.logger, 'name_to_value'):
                for key, value in self.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
            if metrics:
                self.metrics_history.append(metrics)
        return True
    
    def get_final_metrics(self) -> dict:
        """
        Retourne les métriques finales (dernières valeurs enregistrées).
        
        Returns:
            dict: Dictionnaire des métriques finales
        """
        if not self.metrics_history:
            return {}
        for metrics in reversed(self.metrics_history):
            if metrics:
                return metrics
        return {}


def create_progress_ui(total_timesteps: int) -> tuple:
    """
    Crée l'interface utilisateur pour suivre la progression de l'entraînement.
    
    Args:
        total_timesteps: Nombre total de pas d'entraînement
        
    Returns:
        tuple: (progress_bar, status_text, time_elapsed, eta_text, start_time)
    """
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
    
    start_time = time.time()
    
    return progress_bar, status_text, time_elapsed, eta_text, start_time


def create_training_callbacks(
    progress_bar: st.progress,
    status_text: st.empty,
    time_elapsed: st.empty,
    eta_text: st.empty,
    total_timesteps: int,
    start_time: float,
    BaseCallback: Optional[type] = None,
    language: str = "fr"
) -> list:
    """
    Crée les callbacks pour l'entraînement.
    
    Args:
        progress_bar: Widget Streamlit pour la barre de progression
        status_text: Widget pour le texte de statut
        time_elapsed: Widget pour le temps écoulé
        eta_text: Widget pour le temps restant
        total_timesteps: Nombre total de pas d'entraînement
        start_time: Temps de début (timestamp)
        BaseCallback: Classe BaseCallback de stable-baselines3 (optionnel)
        language: Langue pour l'affichage ("fr" ou "en")
        
    Returns:
        list: Liste des callbacks à utiliser pour l'entraînement
    """
    callbacks_list = []
    
    if BaseCallback is not None:
        # Callback pour collecter les métriques
        metrics_callback = MetricsCallback()
        callbacks_list.append(metrics_callback)
        
        # Callback pour suivre la progression
        progress_callback = ProgressCallback(
            progress_bar, status_text, time_elapsed, eta_text,
            total_timesteps, start_time, language=language
        )
        callbacks_list.append(progress_callback)
    
    return callbacks_list


def finalize_training_ui(
    progress_bar: st.progress,
    status_text: st.empty,
    total_timesteps: int,
    start_time: float,
    metrics_callback: Optional[MetricsCallback] = None,
    language: str = "fr"
) -> dict:
    """
    Finalise l'interface utilisateur après l'entraînement.
    
    Args:
        progress_bar: Widget Streamlit pour la barre de progression
        status_text: Widget pour le texte de statut
        total_timesteps: Nombre total de pas d'entraînement
        start_time: Temps de début (timestamp)
        metrics_callback: Callback de métriques (optionnel)
        
    Returns:
        dict: Dictionnaire des métriques finales
    """
    # Finaliser la barre de progression
    progress_bar.progress(1.0)
    
    # Calculer le temps total
    final_time = time.time() - start_time
    final_time_str = format_time(final_time)
    
    # Message de succès
    st.success(_ui_t("train_done_msg", language).format(time=final_time_str))
    
    # Mise à jour du statut
    status_text.markdown(
        f"<div class='progress-status'>"
        f"<strong>{_ui_t('train_done_status', language)}</strong> {total_timesteps:,} {_ui_t('steps_label', language)}"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Récupérer les métriques finales
    training_metrics = {}
    if metrics_callback is not None:
        training_metrics = metrics_callback.get_final_metrics()
    
    return training_metrics


def display_training_metrics(training_metrics: dict, language: str = "fr"):
    """
    Affiche les métriques d'entraînement dans l'interface.
    
    Args:
        training_metrics: Dictionnaire des métriques d'entraînement
        language: Langue pour l'affichage ("fr" ou "en")
    """
    st.markdown(_ui_t("metrics_title", language))
    if training_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            ep_rew = training_metrics.get("rollout/ep_rew_mean", "N/A")
            if isinstance(ep_rew, (int, float)):
                st.metric(_ui_t("metric_reward", language), f"{ep_rew:.2f}")
            else:
                st.metric(_ui_t("metric_reward", language), ep_rew)
        with col2:
            ep_len = training_metrics.get("rollout/ep_len_mean", "N/A")
            if isinstance(ep_len, (int, float)):
                st.metric(_ui_t("metric_len", language), f"{ep_len:.1f}")
            else:
                st.metric(_ui_t("metric_len", language), ep_len)
        with col3:
            loss = training_metrics.get("train/policy_loss", "N/A")
            if isinstance(loss, (int, float)):
                st.metric(_ui_t("metric_loss", language), f"{loss:.4f}")
            else:
                st.metric(_ui_t("metric_loss", language), loss)
    else:
        st.info(_ui_t("metrics_empty", language))
