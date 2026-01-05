"""
Configuration par défaut pour les paramètres physiques et d'entraînement.

Ce module centralise les paramètres par défaut utilisés dans les fonctions d'entraînement
(train_ppo_hybrid_ode, train_ppo_hybrid_cde, etc.) pour garantir
la reproductibilité et la cohérence entre les différents scénarios.

Paramètres définis :
- season_length : Longueur de la saison de culture (en jours)
- max_irrigation : Dose maximale d'irrigation par jour (en mm)
- total_timesteps : Nombre total de pas d'entraînement pour PPO
- seed : Graine aléatoire pour la reproductibilité
- soil_params : Paramètres du sol (Z_r, theta_s, theta_fc, theta_wp, psi_sat, psi_fc, psi_wp, k_d, eta_I, psi_ET_crit)
- weather_params : Paramètres météorologiques (et0_base, et0_amp, et0_noise, p_rain_early, p_rain_mid, p_rain_late, rain_min, rain_max)
"""

from typing import Dict


# ============================================================================
# PARAMÈTRES D'ENVIRONNEMENT
# ============================================================================
DEFAULT_SEASON_LENGTH = 120  # Longueur typique d'une saison de culture (4 mois)
DEFAULT_MAX_IRRIGATION = 20.0  # Dose maximale d'irrigation par jour (mm)
DEFAULT_SEED = 123  # Graine aléatoire par défaut pour reproductibilité

# Limites et pas pour les widgets Streamlit - Paramètres d'environnement
ENV_PARAMS_RANGES = {
    "season_length": {"min": 60, "max": 180, "step": 10},
    "max_irrigation": {"min": 10.0, "max": 50.0, "step": 5.0},
    "seed": {"min": 0, "max": 10000, "step": 1}
}

# Paramètres par défaut pour l'entraînement PPO
DEFAULT_TOTAL_TIMESTEPS = 50000  # Nombre total de pas d'entraînement

# ============================================================================
# PARAMÈTRES DU SOL
# ============================================================================
DEFAULT_SOIL_PARAMS = {
    "Z_r": 600.0,           # Profondeur zone racinaire (mm)
    "theta_s": 0.45,        # Teneur en eau volumique à saturation
    "theta_fc": 0.30,       # Teneur en eau à la capacité au champ
    "theta_wp": 0.15,       # Teneur en eau au point de flétrissement
    "psi_sat": 10.0,        # Tension à saturation (cbar)
    "psi_fc": 33.0,         # Tension à la capacité au champ (cbar)
    "psi_wp": 1500.0,       # Tension au point de flétrissement (cbar)
    "k_d": 0.3,             # Coefficient de drainage
    "eta_I": 0.85,          # Efficacité d'irrigation (0-1)
    "psi_ET_crit": 80.0     # Seuil de stress ET (cbar)
}

# Limites et pas pour les widgets Streamlit - Paramètres du sol
SOIL_PARAMS_RANGES = {
    "Z_r": {"min": 300.0, "max": 1000.0, "step": 50.0},
    "theta_s": {"min": 0.30, "max": 0.60, "step": 0.05},
    "theta_fc": {"min": 0.20, "max": 0.40, "step": 0.05},
    "theta_wp": {"min": 0.10, "max": 0.25, "step": 0.05},
    "psi_sat": {"min": 5.0, "max": 20.0, "step": 1.0},
    "psi_fc": {"min": 20.0, "max": 50.0, "step": 1.0},
    "psi_wp": {"min": 1000.0, "max": 2000.0, "step": 100.0},
    "k_d": {"min": 0.1, "max": 0.5, "step": 0.05},
    "eta_I": {"min": 0.70, "max": 0.95, "step": 0.05},
    "psi_ET_crit": {"min": 50.0, "max": 150.0, "step": 10.0}
}

# ============================================================================
# PARAMÈTRES MÉTÉOROLOGIQUES
# ============================================================================
DEFAULT_WEATHER_PARAMS = {
    "et0_base": 4.0,        # ET0 base (mm/j)
    "et0_amp": 2.0,         # ET0 amplitude
    "et0_noise": 0.3,       # ET0 bruit
    "p_rain_early": 0.25,   # Probabilité pluie début saison
    "p_rain_mid": 0.15,     # Probabilité pluie milieu saison
    "p_rain_late": 0.20,    # Probabilité pluie fin saison
    "rain_min": 3.0,        # Pluie minimale (mm)
    "rain_max": 25.0        # Pluie maximale (mm)
}

# Limites et pas pour les widgets Streamlit - Paramètres météorologiques
WEATHER_PARAMS_RANGES = {
    "et0_base": {"min": 2.0, "max": 6.0, "step": 0.5},
    "et0_amp": {"min": 1.0, "max": 3.0, "step": 0.5},
    "et0_noise": {"min": 0.1, "max": 0.5, "step": 0.1},
    "p_rain_early": {"min": 0.10, "max": 0.40, "step": 0.05},
    "p_rain_mid": {"min": 0.05, "max": 0.30, "step": 0.05},
    "p_rain_late": {"min": 0.10, "max": 0.40, "step": 0.05},
    "rain_min": {"min": 1.0, "max": 10.0, "step": 1.0},
    "rain_max": {"min": 15.0, "max": 50.0, "step": 5.0}
}


def get_default_physics_config(**kwargs) -> Dict:
    """
    Retourne la configuration physique par défaut.
    
    Args:
        **kwargs: Paramètres personnalisés pour surcharger les valeurs par défaut
    
    Returns:
        Dictionnaire de configuration avec les clés :
        - season_length : Longueur de la saison (défaut: 120 jours)
        - max_irrigation : Dose maximale d'irrigation (défaut: 20.0 mm/jour)
        - seed : Graine aléatoire (défaut: 0)
    
    Exemple:
        >>> config = get_default_physics_config(season_length=150, seed=42)
        >>> config['season_length']  # 150
        >>> config['max_irrigation']  # 20.0 (valeur par défaut)
    """
    default_config = {
        "season_length": DEFAULT_SEASON_LENGTH,
        "max_irrigation": DEFAULT_MAX_IRRIGATION,
        "seed": DEFAULT_SEED,
    }
    # Mettre à jour avec les paramètres personnalisés
    default_config.update(kwargs)
    return default_config


def get_default_training_config(**kwargs) -> Dict:
    """
    Retourne la configuration d'entraînement par défaut.
    
    Args:
        **kwargs: Paramètres personnalisés pour surcharger les valeurs par défaut
    
    Returns:
        Dictionnaire de configuration avec les clés :
        - total_timesteps : Nombre total de pas d'entraînement (défaut: 50000)
    
    Exemple:
        >>> config = get_default_training_config(total_timesteps=100000)
        >>> config['total_timesteps']  # 100000
    """
    default_config = {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS,
    }
    # Mettre à jour avec les paramètres personnalisés
    default_config.update(kwargs)
    return default_config


def get_default_soil_config(**kwargs) -> Dict:
    """
    Retourne la configuration des paramètres du sol par défaut.
    
    Args:
        **kwargs: Paramètres personnalisés pour surcharger les valeurs par défaut
    
    Returns:
        Dictionnaire de configuration avec les clés :
        - Z_r : Profondeur zone racinaire (défaut: 600.0 mm)
        - theta_s : Teneur en eau à saturation (défaut: 0.45)
        - theta_fc : Teneur en eau à la capacité au champ (défaut: 0.30)
        - theta_wp : Teneur en eau au point de flétrissement (défaut: 0.15)
        - psi_sat : Tension à saturation (défaut: 10.0 cbar)
        - psi_fc : Tension à la capacité au champ (défaut: 33.0 cbar)
        - psi_wp : Tension au point de flétrissement (défaut: 1500.0 cbar)
        - k_d : Coefficient de drainage (défaut: 0.3)
        - eta_I : Efficacité d'irrigation (défaut: 0.85)
        - psi_ET_crit : Seuil de stress ET (défaut: 80.0 cbar)
    
    Exemple:
        >>> config = get_default_soil_config(Z_r=800.0, theta_s=0.50)
        >>> config['Z_r']  # 800.0
        >>> config['theta_fc']  # 0.30 (valeur par défaut)
    """
    default_config = DEFAULT_SOIL_PARAMS.copy()
    # Mettre à jour avec les paramètres personnalisés
    default_config.update(kwargs)
    return default_config


def get_soil_params_ranges() -> Dict:
    """
    Retourne les limites (min, max) et pas pour les widgets Streamlit des paramètres du sol.
    
    Returns:
        Dictionnaire avec les clés des paramètres du sol, chaque valeur étant un dict
        contenant 'min', 'max', et 'step'
    """
    return SOIL_PARAMS_RANGES.copy()


def get_default_weather_config(**kwargs) -> Dict:
    """
    Retourne la configuration des paramètres météorologiques par défaut.
    
    Args:
        **kwargs: Paramètres personnalisés pour surcharger les valeurs par défaut
    
    Returns:
        Dictionnaire de configuration avec les clés :
        - et0_base : ET0 base (défaut: 4.0 mm/j)
        - et0_amp : ET0 amplitude (défaut: 2.0)
        - et0_noise : ET0 bruit (défaut: 0.3)
        - p_rain_early : Probabilité pluie début saison (défaut: 0.25)
        - p_rain_mid : Probabilité pluie milieu saison (défaut: 0.15)
        - p_rain_late : Probabilité pluie fin saison (défaut: 0.20)
        - rain_min : Pluie minimale (défaut: 3.0 mm)
        - rain_max : Pluie maximale (défaut: 25.0 mm)
    
    Exemple:
        >>> config = get_default_weather_config(et0_base=5.0, rain_max=30.0)
        >>> config['et0_base']  # 5.0
        >>> config['p_rain_early']  # 0.25 (valeur par défaut)
    """
    default_config = DEFAULT_WEATHER_PARAMS.copy()
    # Mettre à jour avec les paramètres personnalisés
    default_config.update(kwargs)
    return default_config


def get_weather_params_ranges() -> Dict:
    """
    Retourne les limites (min, max) et pas pour les widgets Streamlit des paramètres météorologiques.
    
    Returns:
        Dictionnaire avec les clés des paramètres météorologiques, chaque valeur étant un dict
        contenant 'min', 'max', et 'step'
    """
    return WEATHER_PARAMS_RANGES.copy()


def get_default_config(**kwargs) -> Dict:
    """
    Retourne la configuration complète par défaut (physique + entraînement).
    
    Args:
        **kwargs: Paramètres personnalisés pour surcharger les valeurs par défaut
    
    Returns:
        Dictionnaire de configuration avec les clés :
        - season_length : Longueur de la saison (défaut: 120 jours)
        - max_irrigation : Dose maximale d'irrigation (défaut: 20.0 mm/jour)
        - total_timesteps : Nombre total de pas d'entraînement (défaut: 50000)
        - seed : Graine aléatoire (défaut: 0)
    
    Exemple:
        >>> config = get_default_config()
        >>> config['season_length']  # 120
        >>> config['max_irrigation']  # 20.0
        >>> config['total_timesteps']  # 50000
        >>> config['seed']  # 0
    """
    default_config = {
        "season_length": DEFAULT_SEASON_LENGTH,
        "max_irrigation": DEFAULT_MAX_IRRIGATION,
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS,
        "seed": DEFAULT_SEED,
    }
    # Mettre à jour avec les paramètres personnalisés
    default_config.update(kwargs)
    return default_config


def get_env_params_ranges() -> Dict:
    """
    Retourne les limites (min, max) et pas pour les widgets Streamlit des paramètres d'environnement.
    
    Returns:
        Dictionnaire avec les clés des paramètres d'environnement, chaque valeur étant un dict
        contenant 'min', 'max', et 'step'
    """
    return ENV_PARAMS_RANGES.copy()


# ============================================================================
# RÈGLE "SEUIL UNIQUE" (SCÉNARIO 1 - Règles simples)
# ============================================================================
# Valeurs par défaut des paramètres de la règle seuil unique
DEFAULT_RULE_SEUIL_UNIQUE = {
    "threshold_cbar": 80.0,   # Seuil de tension (cbar)
    "dose_mm": 15.0,          # Dose d'irrigation (mm)
    "rain_threshold_mm": 5.0, # Seuil de pluie prévue (mm)
    "reduce_factor": 0.5      # Facteur de réduction si pluie
}

# Limites et pas pour les widgets associés à la règle seuil unique
RULE_SEUIL_UNIQUE_RANGES = {
    "threshold_cbar": {"min": 30.0, "max": 150.0, "step": 5.0},
    "dose_mm": {"min": 5.0, "max": 50.0, "step": 1.0},
    "rain_threshold_mm": {"min": 0.0, "max": 20.0, "step": 1.0},
    "reduce_factor": {"min": 0.0, "max": 1.0, "step": 0.1},
}


def get_rule_seuil_unique_config(**kwargs) -> Dict:
    """
    Retourne la configuration par défaut de la règle "Seuil unique" (scénario 1).
    """
    config = DEFAULT_RULE_SEUIL_UNIQUE.copy()
    config.update(kwargs)
    return config


def get_rule_seuil_unique_ranges() -> Dict:
    """
    Retourne les limites (min, max, step) pour les widgets de la règle "Seuil unique".
    """
    return RULE_SEUIL_UNIQUE_RANGES.copy()


# ============================================================================
# RÈGLE "BANDE DE CONFORT" (SCÉNARIO 1 - Règles simples)
# ============================================================================
DEFAULT_RULE_BANDE_CONFORT = {
    "psi_low": 20.0,   # ψ bas (cbar)
    "psi_high": 60.0,  # ψ haut (cbar)
    "dose_mm": 12.0,   # Dose d'irrigation (mm)
}

RULE_BANDE_CONFORT_RANGES = {
    "psi_low": {"min": 10.0, "max": 50.0, "step": 5.0},
    "psi_high": {"min": 40.0, "max": 100.0, "step": 5.0},
    "dose_mm": {"min": 5.0, "max": 50.0, "step": 1.0},
}


def get_rule_bande_confort_config(**kwargs) -> Dict:
    """
    Retourne la configuration par défaut de la règle "Bande de confort".
    """
    config = DEFAULT_RULE_BANDE_CONFORT.copy()
    config.update(kwargs)
    return config


def get_rule_bande_confort_ranges() -> Dict:
    """
    Retourne les limites (min, max, step) pour les widgets de la règle "Bande de confort".
    """
    return RULE_BANDE_CONFORT_RANGES.copy()


# ============================================================================
# RÈGLE "PROPORTIONNELLE" (SCÉNARIO 1 - Règles simples)
# ============================================================================
DEFAULT_RULE_PROPORTIONNELLE = {
    "psi_target": 40.0,  # ψ cible (cbar)
    "k_I": 0.1,          # Coefficient proportionnel
}

RULE_PROPORTIONNELLE_RANGES = {
    "psi_target": {"min": 20.0, "max": 80.0, "step": 5.0},
    "k_I": {"min": 0.01, "max": 0.5, "step": 0.01},
}


def get_rule_proportionnelle_config(**kwargs) -> Dict:
    """
    Retourne la configuration par défaut de la règle proportionnelle.
    """
    config = DEFAULT_RULE_PROPORTIONNELLE.copy()
    config.update(kwargs)
    return config


def get_rule_proportionnelle_ranges() -> Dict:
    """
    Retourne les limites (min, max, step) pour les widgets de la règle proportionnelle.
    """
    return RULE_PROPORTIONNELLE_RANGES.copy()
