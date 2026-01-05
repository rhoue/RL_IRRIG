"""
Utilitaires pour le modèle physique et les règles d'irrigation.

Ce module contient :
- PhysicalBucket : Modèle physique de type "bucket" pour le bilan hydrique
- Règles d'irrigation : rule_seuil_unique, rule_bande_confort, rule_proportionnelle
- Fonctions de simulation : simulate_scenario1, make_env, evaluate_episode
"""

import numpy as np
from typing import Optional, Dict, Any

# Import de generate_weather depuis utils_weather
try:
    from src.utils_weather import generate_weather
except ImportError:
    # Fallback si l'import ne fonctionne pas
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils_weather import generate_weather


# ============================================================================
# MODÈLE PHYSIQUE
# ============================================================================

class PhysicalBucket:
    """
    Modèle physique de type "bucket" pour simuler le bilan hydrique du sol.
    
    PRINCIPE DU MODÈLE BUCKET :
    Le sol est représenté comme un réservoir (bucket) d'eau dans la zone racinaire.
    L'eau entre (irrigation, pluie) et sort (évapotranspiration, drainage) selon
    les lois de la physique du sol.
    
    VARIABLES CLÉS :
    - S (mm) : Réserve en eau du sol (variable interne, non directement mesurée)
    - ψ (cbar) : Tension matricielle (variable observée par les tensiomètres)
    - Relation S ↔ ψ : Courbe de rétention du sol (fonction non-linéaire)
    
    BILAN HYDRIQUE :
    S_{t+1} = S_t + η_I × I_t + R_t - ETc_t - D(S_t)
    où :
    - η_I : efficacité d'irrigation (fraction de l'eau qui atteint la zone racinaire)
    - I_t : dose d'irrigation appliquée (mm)
    - R_t : pluie (mm)
    - ETc_t : évapotranspiration culturelle = Kc_t × ET0_t × f_ET(ψ_t)
    - D(S_t) : drainage (se produit si S > S_fc)
    
    PARAMÈTRES PÉDOPHYSIQUES :
    - Z_r : profondeur de la zone racinaire (mm)
    - θ_s, θ_fc, θ_wp : teneurs en eau volumiques (0-1)
    - S_max, S_fc, S_wp : réserves correspondantes (mm) = θ × Z_r
    - ψ_sat, ψ_fc, ψ_wp : tensions aux points caractéristiques (cbar)
    """
    def __init__(
        self,
        Z_r=600.0,          # Profondeur zone racinaire (mm) - typiquement 400-800 mm
        theta_s=0.45,       # Teneur volumique à saturation (0-1) - sol argileux ~0.5, sableux ~0.3
        theta_fc=0.30,      # Teneur à capacité au champ (0-1) - point optimal pour les plantes
        theta_wp=0.15,      # Teneur au point de flétrissement (0-1) - limite inférieure
        psi_sat=10.0,       # Tension à saturation (cbar) - proche de 0, eau facilement disponible
        psi_fc=33.0,        # Tension à capacité au champ (cbar) - zone optimale 20-60 cbar
        psi_wp=1500.0,      # Tension au point de flétrissement (cbar) - très élevée, eau difficile à extraire
        k_d=0.3,            # Coefficient de drainage (mm par mm de dépassement) - vitesse de percolation
        eta_I=0.85,         # Efficacité d'irrigation (0-1) - pertes par évaporation/ruissellement
        psi_ET_crit=80.0    # Seuil critique pour réduction ET (cbar) - au-delà, stress hydrique
    ):
        # Paramètres géométriques du sol
        self.Z_r = Z_r
        self.theta_s = theta_s
        self.theta_fc = theta_fc
        self.theta_wp = theta_wp
        
        # Conversion des teneurs volumiques en réserves (mm)
        # La réserve = teneur volumique × profondeur de la zone racinaire
        self.S_max = theta_s * Z_r  # Réserve maximale (saturation)
        self.S_fc = theta_fc * Z_r  # Réserve à capacité au champ (optimal)
        self.S_wp = theta_wp * Z_r  # Réserve au point de flétrissement (minimum)
        
        # Paramètres de la courbe de rétention (relation S ↔ ψ)
        # Cette courbe décrit comment la tension augmente quand la réserve diminue
        self.psi_sat = psi_sat
        self.psi_fc = psi_fc
        self.psi_wp = psi_wp
        
        # Paramètres de dynamique
        self.k_d = k_d              # Coefficient de drainage (proportionnel au dépassement de S_fc)
        self.eta_I = eta_I          # Efficacité d'irrigation (fraction efficace)
        self.psi_ET_crit = psi_ET_crit  # Seuil de stress pour l'évapotranspiration

    def S_to_psi(self, S):
        """
        Convertit la réserve en eau S (mm) en tension matricielle ψ (cbar).
        
        PRINCIPE : Courbe de rétention par morceaux (approximation linéaire)
        La relation réelle S(ψ) est non-linéaire (loi de van Genuchten), mais on utilise
        une approximation linéaire par morceaux pour simplifier.
        
        ZONES :
        1. Zone saturée (S >= S_fc) : interpolation entre ψ_sat et ψ_fc
        2. Zone de disponibilité (S_wp <= S < S_fc) : interpolation entre ψ_fc et ψ_wp
        3. Zone sèche (S < S_wp) : retourne ψ_wp (point de flétrissement)
        
        Args:
            S (float): Réserve en eau du sol (mm)
            
        Returns:
            float: Tension matricielle ψ (cbar)
        """
        S = float(S)
        # Zone saturée à capacité au champ : interpolation linéaire
        if S >= self.S_fc:
            if self.S_max == self.S_fc:
                return self.psi_sat
            # Fraction de la zone entre S_fc et S_max
            frac = (self.S_max - S) / (self.S_max - self.S_fc)
            # Interpolation linéaire entre ψ_sat et ψ_fc
            psi = self.psi_sat + (self.psi_fc - self.psi_sat) * frac
            return float(np.clip(psi, self.psi_sat, self.psi_fc))
        # Zone de disponibilité : entre capacité au champ et point de flétrissement
        elif self.S_wp <= S < self.S_fc:
            # Fraction de la zone entre S_wp et S_fc
            frac = (self.S_fc - S) / (self.S_fc - self.S_wp)
            # Interpolation linéaire entre ψ_fc et ψ_wp
            psi = self.psi_fc + (self.psi_wp - self.psi_fc) * frac
            return float(np.clip(psi, self.psi_fc, self.psi_wp))
        # Zone sèche : au-delà du point de flétrissement
        else:
            return float(self.psi_wp)

    def psi_to_S(self, psi):
        """
        Convertit la tension matricielle ψ (cbar) en réserve en eau S (mm).
        
        FONCTION INVERSE de S_to_psi. Utilisée pour initialiser S à partir
        d'une mesure de tension (par exemple au début d'un épisode).
        
        Args:
            psi (float): Tension matricielle (cbar)
            
        Returns:
            float: Réserve en eau du sol (mm)
        """
        psi = float(psi)
        # Zone saturée : réserve maximale
        if psi <= self.psi_sat:
            return float(self.S_max)
        # Zone entre saturation et capacité au champ
        if self.psi_sat < psi <= self.psi_fc:
            frac = (psi - self.psi_sat) / (self.psi_fc - self.psi_sat)
            S = self.S_max - frac * (self.S_max - self.S_fc)
            return float(np.clip(S, self.S_fc, self.S_max))
        # Zone entre capacité au champ et point de flétrissement
        if self.psi_fc < psi < self.psi_wp:
            frac = (psi - self.psi_fc) / (self.psi_wp - self.psi_fc)
            S = self.S_fc - frac * (self.S_fc - self.S_wp)
            return float(np.clip(S, self.S_wp, self.S_fc))
        # Au-delà du point de flétrissement
        return float(self.S_wp)

    def drainage(self, S):
        """
        Calcule le drainage (percolation) lorsque la réserve dépasse la capacité au champ.
        
        PRINCIPE PHYSIQUE :
        Quand S > S_fc, le sol est au-delà de sa capacité de rétention. L'eau en excès
        percole vers les couches profondes (drainage). Le drainage est proportionnel
        au dépassement de S_fc selon un coefficient k_d.
        
        FORMULE : D(S) = k_d × max(0, S - S_fc)
        
        Args:
            S (float): Réserve en eau actuelle (mm)
            
        Returns:
            float: Volume d'eau drainé (mm/jour)
        """
        return 0.0 if S <= self.S_fc else self.k_d * (S - self.S_fc)

    def f_ET(self, psi):
        """
        Calcule le facteur de réduction de l'évapotranspiration en fonction du stress hydrique.
        
        PRINCIPE :
        Quand le sol devient sec (ψ élevé), les plantes ont du mal à extraire l'eau.
        L'évapotranspiration (ET) est réduite, ce qui limite la croissance et peut
        causer du stress hydrique.
        
        FONCTION :
        - Si ψ <= ψ_ET_crit : pas de stress, f_ET = 1.0 (ET maximale)
        - Si ψ >= ψ_wp : stress maximal, f_ET = 0.0 (pas d'ET, flétrissement)
        - Entre les deux : interpolation linéaire (réduction progressive)
        
        FORMULE : f_ET(ψ) = (ψ_wp - ψ) / (ψ_wp - ψ_ET_crit) pour ψ_ET_crit < ψ < ψ_wp
        
        Args:
            psi (float): Tension matricielle (cbar)
            
        Returns:
            float: Facteur de réduction de l'ET (0-1)
        """
        psi = float(psi)
        # Pas de stress : ET maximale
        if psi <= self.psi_ET_crit:
            return 1.0
        # Stress maximal : pas d'ET
        if psi >= self.psi_wp:
            return 0.0
        # Stress intermédiaire : réduction linéaire entre ψ_ET_crit et ψ_wp
        return (self.psi_wp - psi) / (self.psi_wp - self.psi_ET_crit)


# ============================================================================
# RÈGLES D'IRRIGATION
# ============================================================================

def rule_seuil_unique(psi_t, threshold_cbar=80.0, dose_mm=15.0, I_max=20.0,
                      rain_forecast_mm=0.0, rain_threshold_mm=5.0, reduce_factor=0.5):
    """
    Règle d'irrigation à seuil unique (SCÉNARIO 1).
    
    PRINCIPE :
    Règle simple et couramment utilisée par les agriculteurs :
    "Si la tension dépasse un seuil, j'irrigue une dose fixe."
    
    LOGIQUE :
    - Si ψ_t > threshold_cbar : irriguer avec dose_mm (mm)
    - Si pluie prévue >= rain_threshold_mm : réduire l'irrigation (évite gaspillage)
    - Sinon : pas d'irrigation
    
    AVANTAGES :
    - Simple à comprendre et implémenter
    - Réactive (répond rapidement au stress)
    - Peut tenir compte des prévisions météo
    
    INCONVÉNIENTS :
    - Ne tient pas compte de l'historique
    - Peut être inefficace (irrigation excessive ou insuffisante)
    - Ne s'adapte pas aux conditions spécifiques
    
    Args:
        psi_t (float): Tension matricielle actuelle (cbar)
        threshold_cbar (float): Seuil de déclenchement (cbar) - typiquement 60-100 cbar
        dose_mm (float): Dose d'irrigation à appliquer si seuil dépassé (mm)
        I_max (float): Dose maximale autorisée (mm)
        rain_forecast_mm (float): Pluie prévue pour le jour suivant (mm)
        rain_threshold_mm (float): Seuil de pluie pour réduire l'irrigation (mm)
        reduce_factor (float): Facteur de réduction si pluie imminente (0-1)
        
    Returns:
        float: Dose d'irrigation à appliquer (mm)
    """
    # Vérification du seuil de tension
    if psi_t > threshold_cbar:
        # Dose de base
        I = min(dose_mm, I_max)
        # Réduction si pluie imminente (évite gaspillage)
        if rain_forecast_mm >= rain_threshold_mm:
            I *= reduce_factor
        return float(np.clip(I, 0.0, I_max))
    # Pas d'irrigation si sous le seuil
    return 0.0


def rule_bande_confort(psi_t, psi_low=20.0, psi_high=60.0, dose_mm=12.0, I_max=20.0, **_):
    """
    Règle d'irrigation basée sur une bande de confort (SCÉNARIO 1).
    
    PRINCIPE :
    Maintient la tension dans une bande optimale [psi_low, psi_high] pour la croissance.
    Irrigue uniquement si on sort par le haut de la bande (sol trop sec).
    
    LOGIQUE :
    - Si ψ_t > psi_high : irriguer avec dose_mm (ramener dans la bande)
    - Si ψ_t dans [psi_low, psi_high] : pas d'irrigation (zone optimale)
    - Si ψ_t < psi_low : pas d'irrigation (sol trop humide, risque de drainage)
    
    AVANTAGES :
    - Maintient la tension dans une zone optimale
    - Évite l'irrigation excessive (ne réagit que si nécessaire)
    - Plus sophistiquée que le seuil unique
    
    INCONVÉNIENTS :
    - Ne tient pas compte de la tendance (peut être trop réactive)
    - Dose fixe peut être sous-optimale
    
    BANDE DE CONFORT TYPIQUE :
    - psi_low = 20 cbar : limite basse (sol bien humide)
    - psi_high = 60 cbar : limite haute (début de stress)
    - Zone optimale pour la plupart des cultures : 20-60 cbar
    
    Args:
        psi_t (float): Tension matricielle actuelle (cbar)
        psi_low (float): Limite basse de la bande de confort (cbar)
        psi_high (float): Limite haute de la bande de confort (cbar)
        dose_mm (float): Dose d'irrigation à appliquer (mm)
        I_max (float): Dose maximale autorisée (mm)
        
    Returns:
        float: Dose d'irrigation à appliquer (mm)
    """
    # Irriguer uniquement si on sort par le haut de la bande de confort
    return float(np.clip(dose_mm if psi_t > psi_high else 0.0, 0.0, I_max))


def rule_proportionnelle(psi_t, psi_target=40.0, k_I=0.1, I_max=20.0, **_):
    """
    Règle d'irrigation proportionnelle (SCÉNARIO 1).
    
    PRINCIPE :
    La dose d'irrigation est proportionnelle à l'écart par rapport à une tension cible.
    Plus le sol est sec (ψ élevé), plus on irrigue.
    
    FORMULE :
    I_t = min(I_max, k_I × max(0, ψ_t - ψ_target))
    où :
    - ψ_target : tension cible à maintenir (cbar)
    - k_I : coefficient de proportionnalité (mm/cbar)
    - (ψ_t - ψ_target)_+ : partie positive de l'écart
    
    LOGIQUE :
    - Si ψ_t <= ψ_target : pas d'irrigation (sol suffisamment humide)
    - Si ψ_t > ψ_target : irrigation proportionnelle à l'écart
    - Plus l'écart est grand, plus la dose est importante
    
    AVANTAGES :
    - S'adapte à l'intensité du stress (dose variable)
    - Plus fine que les règles à seuil fixe
    - Peut être calibrée via k_I
    
    INCONVÉNIENTS :
    - Nécessite un réglage de k_I (peut être instable)
    - Peut réagir trop fortement si k_I trop élevé
    
    CALIBRAGE :
    - k_I = 0.1 signifie : 1 cbar d'écart → 0.1 mm d'irrigation
    - Pour un écart de 40 cbar (ψ=80, target=40) : I = 0.1 × 40 = 4 mm
    
    Args:
        psi_t (float): Tension matricielle actuelle (cbar)
        psi_target (float): Tension cible à maintenir (cbar) - typiquement 30-50 cbar
        k_I (float): Coefficient de proportionnalité (mm/cbar) - typiquement 0.05-0.2
        I_max (float): Dose maximale autorisée (mm)
        
    Returns:
        float: Dose d'irrigation à appliquer (mm)
    """
    # Calcul de l'écart par rapport à la cible (partie positive uniquement)
    excess = max(0.0, psi_t - psi_target)
    # Dose proportionnelle à l'écart
    I = k_I * excess
    # Clippage à la dose maximale
    return float(np.clip(I, 0.0, I_max))


# ============================================================================
# FONCTIONS DE SIMULATION
# ============================================================================

def simulate_scenario1(
    T=120,
    seed=123,
    I_max=20.0,
    soil: Optional[PhysicalBucket] = None,
    rule_fn=rule_seuil_unique,
    rule_kwargs=None,
    weather_params: Optional[Dict[str, Any]] = None
):
    """
    Simule le SCÉNARIO 1 : modèle physique + règles simples d'irrigation.
    
    PRINCIPE DU SCÉNARIO 1 :
    Utilise un modèle physique (bucket) pour simuler le bilan hydrique du sol
    et applique une règle d'irrigation fixe basée sur des seuils de tension ψ.
    
    BOUCLE DE SIMULATION :
    Pour chaque jour t de la saison :
    1. Mesure de la tension actuelle ψ_t (via courbe de rétention S → ψ)
    2. Décision d'irrigation I_t = rule_fn(ψ_t, ...) selon la règle choisie
    3. Calcul des flux :
       - ETc_t = Kc_t × ET0_t × f_ET(ψ_t)  (évapotranspiration)
       - D_t = drainage(S_t)               (drainage si S > S_fc)
    4. Mise à jour du bilan hydrique :
       S_{t+1} = S_t + η_I × I_t + R_t - ETc_t - D_t
    5. Conversion S_{t+1} → ψ_{t+1} pour le jour suivant
    
    DIFFÉRENCE AVEC SCÉNARIO 2 :
    - Scénario 1 : règle fixe (déterministe, pas d'apprentissage)
    - Scénario 2 : politique apprise par RL (s'adapte, optimise)
    
    Args:
        T (int): Longueur de la saison en jours
        seed (int): Graine aléatoire pour la météo
        I_max (float): Irrigation maximale (mm/jour)
        soil (PhysicalBucket, optional): Instance du modèle de sol (créée si None)
        rule_fn (callable): Fonction de règle d'irrigation (rule_seuil_unique, rule_bande_confort, etc.)
        rule_kwargs (dict, optional): Paramètres de la règle (seuil, dose, etc.)
        weather_params (dict, optional): Paramètres météorologiques (ET0, pluie, etc.)
        
    Returns:
        dict: Dictionnaire contenant :
            - S (np.ndarray): Historique des réserves (mm), shape (T+1,)
            - psi (np.ndarray): Historique des tensions (cbar), shape (T+1,)
            - I (np.ndarray): Historique des irrigations (mm), shape (T,)
            - rain (np.ndarray): Historique des pluies (mm), shape (T,)
            - ET0 (np.ndarray): Historique des ET0 (mm/j), shape (T,)
            - Kc (np.ndarray): Historique des Kc (-), shape (T,)
            - ETc (np.ndarray): Historique des ETc (mm), shape (T,)
            - D (np.ndarray): Historique des drainages (mm), shape (T,)
            - params (dict): Paramètres de la simulation
            - soil (PhysicalBucket): Instance du modèle de sol utilisée
    """
    # Initialisation du modèle de sol si non fourni
    if soil is None:
        soil = PhysicalBucket()
    if rule_kwargs is None:
        rule_kwargs = {}
    
    # Génération météo avec paramètres personnalisés
    # Génère rain, et0, Kc pour toute la saison (déterministe pour une graine donnée)
    weather_kwargs = weather_params if weather_params else {}
    rain, et0, Kc = generate_weather(T=T, seed=seed, **weather_kwargs)

    # Initialisation des tableaux pour stocker l'historique
    # T+1 pour S et psi (état initial + T jours)
    # T pour I, ETc, D (pas d'état initial pour les actions/flux)
    S = np.zeros(T + 1, dtype=np.float32)      # Réserve en eau (mm)
    psi = np.zeros(T + 1, dtype=np.float32)    # Tension matricielle (cbar)
    I = np.zeros(T, dtype=np.float32)          # Irrigation (mm)
    ETc = np.zeros(T, dtype=np.float32)        # Évapotranspiration culturelle (mm)
    D = np.zeros(T, dtype=np.float32)         # Drainage (mm)

    # État initial : sol à la capacité au champ
    # Condition initiale typique : sol à capacité au champ (optimal pour démarrer)
    # Cela correspond à une tension d'environ ψ_fc (typiquement 30-40 cbar)
    S[0] = soil.S_fc
    psi[0] = soil.S_to_psi(S[0])

    # BOUCLE PRINCIPALE DE SIMULATION
    # Pour chaque jour de la saison :
    for t in range(T):
        # Prévision pluie simple (t+1) pour la règle d'irrigation
        # Prévision naïve : on connaît la pluie du jour suivant (simulation)
        # En pratique, on utiliserait des prévisions météo réelles
        rain_fc = rain[t + 1] if (t + 1) < T else 0.0

        # DÉCISION D'IRRIGATION PAR RÈGLE
        # La règle choisit la dose I_t en fonction de :
        # - La tension actuelle ψ_t (variable observée)
        # - La pluie prévue (pour éviter gaspillage)
        # - Les paramètres de la règle (seuil, dose, etc.)
        I[t] = float(rule_fn(
            psi_t=psi[t],
            I_max=I_max,
            rain_forecast_mm=rain_fc,
            **{k: v for k, v in rule_kwargs.items()}
        ))

        # CALCUL DES FLUX PHYSIQUES
        # Évapotranspiration culturelle : ETc = Kc × ET0 × f_ET(ψ)
        # f_ET(ψ) : facteur de stress hydrique (réduction si sol sec)
        fET = soil.f_ET(psi[t])
        ETc[t] = Kc[t] * et0[t] * fET
        
        # Drainage : se produit si S > S_fc (sol au-delà de sa capacité de rétention)
        D[t] = soil.drainage(S[t])

        # MISE À JOUR DU BILAN HYDRIQUE
        # Bilan : S_{t+1} = S_t + η_I × I_t + R_t - ETc_t - D_t
        # où :
        # - η_I × I_t : irrigation efficace (pertes prises en compte)
        # - R_t : pluie (100% efficace, pas de pertes)
        # - ETc_t : évapotranspiration (sortie)
        # - D_t : drainage (sortie)
        S_next = S[t] + soil.eta_I * I[t] + rain[t] - ETc[t] - D[t]
        # Clippage pour maintenir S dans [0, S_max] (contraintes physiques)
        S_next = float(np.clip(S_next, 0.0, soil.S_max))
        S[t + 1] = S_next
        
        # Conversion de la réserve en tension pour le jour suivant
        # C'est cette valeur qui sera utilisée par la règle au pas suivant
        psi[t + 1] = soil.S_to_psi(S[t + 1])

    return {
        "S": S,
        "psi": psi,
        "I": I,
        "rain": rain,
        "ET0": et0,
        "Kc": Kc,
        "ETc": ETc,
        "D": D,
        "params": {
            "T": T,
            "seed": seed,
            "I_max": I_max,
            "rule_fn": rule_fn.__name__,
            "rule_kwargs": rule_kwargs
        },
        "soil": soil,
    }


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def make_env(seed=None, season_length=None, max_irrigation=None, soil_params=None, weather_params=None, goal_spec=None, weather_shift_cfg=None):
    """
    Fabrique une fonction d'initialisation d'environnement avec Monitor.
    
    Cette fonction utilise IrrigationEnvPhysical de utils_env_gymnasium.py pour le scénario 2
    (RL sur modèle physique simple), qui supporte weather_params.
    
    Pour les scénarios 3, 4, 5 (avec modèles résiduels), utiliser directement
    IrrigationEnvPhysical de utils_env_modeles.py.
    
    Args:
        seed (int, optional): Graine aléatoire (défaut depuis utils_physics_config)
        season_length (int, optional): Longueur de la saison (défaut depuis utils_physics_config)
        max_irrigation (float, optional): Dose maximale d'irrigation (défaut depuis utils_physics_config)
        soil_params (dict, optional): Paramètres du sol
        weather_params (dict, optional): Paramètres météorologiques
        goal_spec (dict, optional): Priorités/objets lexicographiques (Option A)
        weather_shift_cfg (dict, optional): Décalage météo (robustesse)
        
    Returns:
        callable: Fonction d'initialisation d'environnement
    """
    # Importer la configuration par défaut
    from src.utils_physics_config import get_default_physics_config
    
    # Obtenir les paramètres par défaut et surcharger avec les valeurs fournies
    default_config = get_default_physics_config(
        seed=seed,
        season_length=season_length,
        max_irrigation=max_irrigation
    )
    seed = default_config["seed"]
    season_length = default_config["season_length"]
    max_irrigation = default_config["max_irrigation"]
    
    # Import conditionnel de Monitor
    try:
        from stable_baselines3.common.monitor import Monitor  # type: ignore
    except ImportError:
        raise ImportError("stable-baselines3 n'est pas installé. Installez-le avec: pip install stable-baselines3")
    
    # Import de l'environnement depuis utils_env_gymnasium (scénario 2)
    # Cette version supporte weather_params et utilise PhysicalBucket
    try:
        from src.utils_env_gymnasium import IrrigationEnvPhysical
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils_env_gymnasium import IrrigationEnvPhysical
    
    def _init():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil_params=soil_params,  # utils_env_gymnasium utilise 'soil_params'
            weather_params=weather_params,  # Supporté par utils_env_gymnasium
            goal_spec=goal_spec,
            weather_shift_cfg=weather_shift_cfg,
        )
        return Monitor(env)
    return _init


def evaluate_episode(
    model,
    season_length=120,
    max_irrigation=20.0,
    seed=123,
    soil_params=None,
    weather_params=None,
    residual_ode=None,
    residual_cde=None,
    seq_len_cde=5,
):
    """
    Évalue un modèle PPO entraîné en exécutant un épisode complet (SCÉNARIO 2, 3 ou 4).
    
    PRINCIPE :
    Simule une saison complète en utilisant la politique apprise par le modèle PPO.
    Mode déterministe : pas d'exploration, choix de l'action la plus probable.
    Permet d'évaluer la performance du modèle sur une nouvelle saison.
    
    DIFFÉRENCE AVEC L'ENTRAÎNEMENT :
    - Entraînement : mode stochastique (exploration) + collecte de données pour apprendre
    - Évaluation : mode déterministe (exploitation) + enregistrement pour analyse
    
    Args:
        model (PPO): Modèle PPO entraîné
        season_length (int): Longueur de la saison en jours
        max_irrigation (float): Dose maximale d'irrigation (mm)
        seed (int): Graine pour générer la météo (différente de l'entraînement pour évaluation)
        soil_params (dict, optional): Paramètres du sol
        weather_params (dict, optional): Paramètres météorologiques
        residual_ode (nn.Module, optional): Modèle Neural ODE pour le scénario 3 (hybride)
        residual_cde (nn.Module, optional): Modèle Neural CDE pour le scénario 4 (hybride)
        seq_len_cde (int, optional): Longueur de séquence pour le CDE (défaut: 5)
        
    Returns:
        dict: Dictionnaire contenant :
            - psi (np.ndarray): Historique des tensions (cbar), shape (season_length+1,)
            - S (np.ndarray): Historique des réserves (mm), shape (season_length+1,)
            - I (np.ndarray): Historique des irrigations (mm), shape (season_length,)
            - R (np.ndarray): Historique des pluies (mm), shape (season_length,)
            - ETc (np.ndarray): Historique des ETc (mm), shape (season_length,)
            - D (np.ndarray): Historique des drainages (mm), shape (season_length,)
            - env_summary (dict): Paramètres de l'environnement (S_fc, S_wp, season_length)
    """
    # Import des environnements : version simple (scénario 2) et version avancée (scénarios 3-5)
    try:
        from src.utils_env_modeles import IrrigationEnvPhysical as IrrigationEnvPhysicalExternal
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils_env_modeles import IrrigationEnvPhysical as IrrigationEnvPhysicalExternal
    try:
        from src.utils_env_gymnasium import IrrigationEnvPhysical as IrrigationEnvPhysicalSimple
    except ImportError:
        IrrigationEnvPhysicalSimple = None
    
    # Création d'un nouvel environnement pour l'évaluation
    # Utilise une graine différente de l'entraînement pour tester la généralisation
    # Convertir soil_params en format 'soil' si nécessaire
    soil_dict = None
    if soil_params:
        # IrrigationEnvPhysical attend 'soil' (dict) pas 'soil_params'
        soil_dict = soil_params
    
    # Choix de l'environnement :
    # - Scénario 2 (pas de modèle résiduel) : utiliser utils_env_gymnasium
    #   pour rester cohérent avec l'entraînement et la génération météo du scénario 1.
    # - Scénarios 3-4 : utiliser utils_env_modeles (support des modèles résiduels).
    use_simple_env = (
        residual_ode is None
        and residual_cde is None
    )
    if use_simple_env and IrrigationEnvPhysicalSimple is not None:
        env = IrrigationEnvPhysicalSimple(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil_params=soil_params,
            weather_params=weather_params
        )
    else:
        env = IrrigationEnvPhysicalExternal(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_dict,  # Utiliser 'soil' au lieu de 'soil_params'
            weather_params=weather_params,
            residual_ode=residual_ode,  # Modèle résiduel pour le scénario 3
            residual_cde=residual_cde,  # Modèle résiduel pour le scénario 4
            seq_len_cde=seq_len_cde,  # Longueur de séquence pour le CDE
            device="cpu"
        )
        
    obs, info = env.reset()

    # Initialisation des historiques avec l'état initial
    psi_hist = [float(env.psi)]
    S_hist = [float(env.S)]
    I_hist, R_hist, ETc_hist, D_hist = [], [], [], []

    # Boucle principale : simulation de la saison
    terminated = truncated = False
    while not (terminated or truncated):
        # Prédiction de l'action avec politique déterministe (pas d'exploration)
        # deterministic=True : choisit l'action la plus probable (mode exploitation)
        action, _ = model.predict(obs, deterministic=True)
        
        # Exécution de l'action et mise à jour de l'environnement
        obs, reward, terminated, truncated, info = env.step(action)

        # Enregistrement des variables d'état et des actions
        I_hist.append(float(action[0] if hasattr(action, "__len__") else action))
        R_hist.append(float(env.rain[env.day - 1]))
        psi_hist.append(float(env.psi))
        S_hist.append(float(env.S))

        # Calcul rétrospectif de l'ETc et du drainage pour le jour précédent
        # (nécessaire car ces valeurs sont calculées dans step() mais pas stockées directement)
        t_prev = env.day - 1
        # Récupération du facteur de stress ET (utiliser méthode privée de l'environnement externe)
        # L'environnement externe utilise _f_ET() et _drainage() comme méthodes privées
        if hasattr(env, '_f_ET') and hasattr(env, '_drainage'):
            # Environnement externe (src.utils_env_modeles)
            f_ET = env._f_ET(psi_hist[-2])
            ETc = env.Kc[t_prev] * env.et0[t_prev] * f_ET
            D = env._drainage(S_hist[-2])
        else:
            # Environnement local (classe dans ce fichier) - fallback
            f_ET = env.soil.f_ET(psi_hist[-2])
            ETc = env.Kc[t_prev] * env.et0[t_prev] * f_ET
            D = env.soil.drainage(S_hist[-2])
        ETc_hist.append(float(ETc))
        D_hist.append(float(D))

    return {
        "psi": np.asarray(psi_hist),
        "S": np.asarray(S_hist),
        "I": np.asarray(I_hist),
        "R": np.asarray(R_hist),
        "ETc": np.asarray(ETc_hist),
        "D": np.asarray(D_hist),
        "env_summary": {
            "S_fc": env.S_fc if hasattr(env, 'S_fc') else (env.soil.S_fc if hasattr(env, 'soil') else 90.0),
            "S_wp": env.S_wp if hasattr(env, 'S_wp') else (env.soil.S_wp if hasattr(env, 'soil') else 30.0),
            "season_length": season_length
        },
    }
