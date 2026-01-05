"""
Utilitaires communs pour l'entraînement PPO.

Ce module contient :
- MetricsCallback : Callback pour collecter les métriques d'entraînement PPO
- ProgressCallback : Callback pour suivre la progression de l'entraînement
- get_default_ppo_config : Fonction pour obtenir la configuration PPO par défaut
- create_ppo_callbacks : Fonction pour créer les callbacks PPO (métriques + progression)
"""

from typing import Dict, Optional, List, Callable

# Imports stable-baselines3
try:
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    BaseCallback = None


class MetricsCallback(BaseCallback):
    """
    Callback pour collecter les métriques d'entraînement PPO.
    
    Collecte automatiquement les métriques enregistrées par stable-baselines3
    (reward, policy_loss, value_loss, etc.) et permet de récupérer les métriques finales.
    """
    def __init__(self):
        if not PPO_AVAILABLE:
            raise ImportError("stable-baselines3 n'est pas disponible")
        super().__init__()
        self.metrics_history = []
        self.ep_rewards = []
        self.ep_lengths = []
    
    def _on_step(self) -> bool:
        """
        Collecte les métriques à chaque pas d'entraînement.
        
        Returns:
            True pour continuer l'entraînement
        """
        # Collecter les métriques à chaque log (généralement tous les n_steps)
        if self.logger is not None:
            metrics = {}
            if hasattr(self.logger, 'name_to_value'):
                for key, value in self.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
            if metrics:
                self.metrics_history.append(metrics)
        # Capturer les épisodes terminés via Monitor
        infos = self.locals.get("infos") if hasattr(self, "locals") else None
        dones = self.locals.get("dones") if hasattr(self, "locals") else None
        if infos is not None and dones is not None:
            for info, done in zip(infos, dones):
                if not done or not isinstance(info, dict):
                    continue
                episode = info.get("episode")
                if isinstance(episode, dict):
                    r = episode.get("r")
                    l = episode.get("l")
                    if isinstance(r, (int, float)) and isinstance(l, (int, float)):
                        self.ep_rewards.append(float(r))
                        self.ep_lengths.append(float(l))
        return True
    
    def get_final_metrics(self) -> Dict:
        """
        Retourne les métriques finales (dernières valeurs enregistrées).
        
        Returns:
            Dictionnaire des métriques finales (reward, policy_loss, value_loss, etc.)
        """
        if not self.metrics_history:
            return {}
        # Agréger les dernières valeurs disponibles pour chaque métrique.
        merged: Dict[str, float] = {}
        for metrics in self.metrics_history:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged[key] = value
        if self.ep_rewards:
            merged.setdefault("rollout/ep_rew_mean", sum(self.ep_rewards) / len(self.ep_rewards))
        if self.ep_lengths:
            merged.setdefault("rollout/ep_len_mean", sum(self.ep_lengths) / len(self.ep_lengths))
        return merged


class ProgressCallback(BaseCallback):
    """
    Callback pour suivre la progression de l'entraînement PPO.
    
    Appelle une fonction de callback personnalisée à intervalles réguliers
    pour mettre à jour l'interface utilisateur (barre de progression, etc.).
    """
    def __init__(self, callback_fn: Callable, total_timesteps: int):
        """
        Args:
            callback_fn: Fonction appelée avec (current_timesteps, total_timesteps)
            total_timesteps: Nombre total de pas d'entraînement
        """
        if not PPO_AVAILABLE:
            raise ImportError("stable-baselines3 n'est pas disponible")
        super().__init__()
        self.callback_fn = callback_fn
        self.total_timesteps = float(total_timesteps)
    
    def _on_step(self) -> bool:
        """
        Appelle la fonction de callback à chaque pas d'entraînement.
        
        Returns:
            True pour continuer l'entraînement
        """
        if self.callback_fn:
            try:
                current = int(self.num_timesteps) if hasattr(self, 'num_timesteps') else 0
                total = int(self.total_timesteps)
                if total > 0:
                    # Le callback attend (current, total) et calcule progress lui-même
                    self.callback_fn(current, total)
            except (TypeError, ValueError, ZeroDivisionError):
                # En cas d'erreur, ne pas bloquer l'entraînement
                pass
        return True


def get_default_ppo_config(**kwargs) -> Dict:
    """
    Retourne la configuration PPO par défaut.
    
    Args:
        **kwargs: Paramètres PPO personnalisés pour surcharger les valeurs par défaut
    
    Returns:
        Dictionnaire de configuration PPO
    """
    default_config = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "verbose": 0,
    }
    # Mettre à jour avec les paramètres personnalisés
    default_config.update(kwargs)
    return default_config


# ============================================================================
# HYPERPARAMÈTRES PPO POUR L'UI (bornes et valeurs par défaut homogènes)
# ============================================================================
DEFAULT_PPO_HYPERPARAMS_UI = {
    "total_timesteps": 50000,
    "policy": "MlpPolicy",
    "n_steps": 2048,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
}

PPO_HYPERPARAMS_UI_RANGES = {
    "total_timesteps": {"min": 10000, "max": 500000, "step": 10000},
    "n_steps": {"min": 512, "max": 4096, "step": 256},
    "batch_size": {"min": 32, "max": 256, "step": 32},
    "learning_rate": {"min": 1e-5, "max": 1e-2, "step": 1e-4},
    "gamma": {"min": 0.90, "max": 0.999, "step": 0.01},
    "gae_lambda": {"min": 0.90, "max": 0.99, "step": 0.01},
    "clip_range": {"min": 0.1, "max": 0.3, "step": 0.05},
    "ent_coef": {"min": 0.0, "max": 0.1, "step": 0.01},
    "vf_coef": {"min": 0.1, "max": 1.0, "step": 0.1},
}


def get_default_ppo_hyperparams_ui(**kwargs) -> Dict:
    """
    Retourne les hyperparamètres PPO par défaut pour l'interface utilisateur.
    """
    config = DEFAULT_PPO_HYPERPARAMS_UI.copy()
    config.update(kwargs)
    return config


def get_ppo_hyperparams_ui_ranges() -> Dict:
    """
    Retourne les bornes (min, max, step) pour les hyperparamètres PPO dans l'UI.
    """
    return PPO_HYPERPARAMS_UI_RANGES.copy()


def create_ppo_callbacks(
    progress_callback: Optional[Callable] = None,
    total_timesteps: int = 10000
) -> tuple:
    """
    Crée les callbacks PPO (métriques + progression optionnelle).
    
    Args:
        progress_callback: Fonction optionnelle pour suivre la progression
        total_timesteps: Nombre total de pas d'entraînement (pour le callback de progression)
    
    Returns:
        Tuple (liste des callbacks, MetricsCallback pour récupérer les métriques finales)
    """
    if not PPO_AVAILABLE:
        raise ImportError("stable-baselines3 n'est pas disponible")
    
    callbacks: List[BaseCallback] = []
    
    # Callback pour collecter les métriques
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)
    
    # Callback pour la progression (optionnel)
    if progress_callback:
        progress_cb = ProgressCallback(progress_callback, total_timesteps)
        callbacks.append(progress_cb)
    
    return callbacks, metrics_callback
