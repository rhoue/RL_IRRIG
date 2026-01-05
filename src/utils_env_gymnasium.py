"""
Utilitaires pour l'environnement Gymnasium.

Ce module contient :
- IrrigationEnvPhysical : Environnement Gymnasium pour l'apprentissage par renforcement de l'irrigation
"""

import numpy as np
from typing import Dict, Any, Optional

# Imports Gymnasium
try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    # Créer des objets factices pour éviter les erreurs de type
    class DummyGym:
        class Env:
            pass
    gym = DummyGym()
    spaces = None

# Imports des utilitaires
try:
    from src.utils_physical_model import PhysicalBucket
    from src.utils_weather import generate_weather
except ImportError:
    # Fallback si l'import ne fonctionne pas
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils_physical_model import PhysicalBucket
    from src.utils_weather import generate_weather


# ============================================================================
# ENVIRONNEMENT GYMNASIUM
# ============================================================================

class IrrigationEnvPhysical(gym.Env if GYM_AVAILABLE else object):
    """
    Environnement Gymnasium pour l'apprentissage par renforcement de l'irrigation.
    
    SCÉNARIO 2 : RL sur modèle physique (avec ψ_t observée)
    
    PRINCIPE :
    Cet environnement implémente le Scénario 2 où un agent RL observe la tension
    matricielle ψ_t (et le contexte météorologique) et choisit une dose d'irrigation I_t.
    L'environnement simule la saison via le modèle physique (bucket + courbe de rétention).
    
    ARCHITECTURE MDP (Markov Decision Process) :
    - État (observation) : o_t = [ψ_t, rain_t, et0_t, Kc_t]
      * ψ_t : tension matricielle actuelle (cbar) - variable observée par tensiomètre
      * rain_t : pluie du jour (mm)
      * et0_t : évapotranspiration de référence (mm/j)
      * Kc_t : coefficient cultural du jour (-)
    
    - Action : a_t = I_t ∈ [0, max_irrigation] (mm)
      * Dose d'irrigation continue choisie par l'agent
    
    - Transition : S_{t+1} = f_physique(S_t, I_t, rain_t, ETc_t, D_t)
      * Modèle physique bucket pour l'évolution de la réserve S
      * Conversion S ↔ ψ via courbe de rétention
    
    - Récompense : r_t = -α × stress(ψ_t) - β × I_t - γ × D_t
      * Pénalise le stress hydrique (écart de ψ par rapport à zone de confort)
      * Pénalise l'eau utilisée (coût économique/environnemental)
      * Pénalise le drainage (pertes d'eau)
    
    OBJECTIF :
    Apprendre une politique π_θ(I_t | o_t) qui minimise le stress hydrique tout en
    économisant l'eau, en respectant la physique du sol.
    """
    
    metadata = {"render_modes": []}

    def __init__(
        self,
        season_length=120,      # Longueur de la saison en jours
        max_irrigation=20.0,     # Dose maximale d'irrigation par jour (mm)
        seed=123,                  # Graine aléatoire pour reproductibilité
        soil_params: Optional[Dict[str, float]] = None,    # Paramètres du sol (optionnel)
        weather_params: Optional[Dict[str, Any]] = None,   # Paramètres météo (optionnel)
        goal_spec: Optional[Dict[str, Any]] = None,        # Spécification lexicographique (optionnel)
        weather_shift_cfg: Optional[Dict[str, Any]] = None,  # Décalage météo (robustesse)
    ):
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium n'est pas installé. Installez-le avec: pip install gymnasium")
        
        super().__init__()
        self.season_length = season_length
        self.max_irrigation = max_irrigation
        self.goal_spec = goal_spec or {}
        self.weather_shift_cfg = weather_shift_cfg or {}
        
        # Initialisation du modèle physique du sol avec paramètres personnalisés
        # Si soil_params est fourni, utilise ces valeurs, sinon valeurs par défaut
        if soil_params:
            self.soil = PhysicalBucket(**soil_params)
        else:
            self.soil = PhysicalBucket()
        
        # Génération des données météorologiques avec paramètres personnalisés
        # Génère rain, et0, Kc pour toute la saison (déterministe pour une graine donnée)
        weather_kwargs = weather_params if weather_params else {}
        self.rng = np.random.default_rng(seed)
        self.rain, self.et0, self.Kc = generate_weather(T=season_length, seed=seed, **weather_kwargs)
        # Appliquer un éventuel décalage météo (robustesse)
        try:
            from src.utils_weather_shift import apply_weather_shift
            self.rain, self.et0 = apply_weather_shift(self.rain, self.et0, self.weather_shift_cfg, rng=self.rng)
        except Exception:
            pass

        # Espace d'observation : [ψ, rain, et0, Kc]
        # Limites supérieures réalistes :
        # - ψ max ~1000 cbar (sol très sec)
        # - pluie max ~200 mm (événement extrême)
        # - ET0 max ~20 mm/j (jour très chaud)
        # - Kc max ~2 (culture très exigeante)
        high_obs = np.array([1e3, 200.0, 20.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high_obs, shape=(4,), dtype=np.float32)
        
        # Espace d'action : dose d'irrigation continue [0, max_irrigation]
        # Action continue (Box) plutôt que discrète pour plus de réalisme
        self.action_space = spaces.Box(
            low=0.0,
            high=np.array([self.max_irrigation], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # État initial : jour 0, réserve à la capacité au champ
        # Condition initiale typique : sol à capacité au champ (optimal pour démarrer)
        self.day = 0
        self.S = float(self.soil.S_fc)  # Réserve à capacité au champ
        self.psi = float(self.soil.S_to_psi(self.S))  # Tension correspondante
        self.cum_irrig = 0.0
        self.cum_drain = 0.0
        self.events_count = 0

    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset environment to initial state.
        
        IMPORTANT BEHAVIOR:
        - Weather is NOT regenerated on reset (generated once at initialization)
        - Only state variables (day, S, psi) are reset
        - If seed is provided, it updates the RNG but weather array remains unchanged
        - To get new weather, create a new environment instance
        
        This differs from utils_env_modeles.py which regenerates weather on each reset().
        For consistency when comparing scenarios, ensure environments use same seed at initialization.
        """
        if seed is not None:
            self.seed(seed)  # Updates RNG but weather arrays (rain, et0, Kc) are not regenerated
        self.day = 0
        self.S = float(self.soil.S_fc)
        self.psi = float(self.soil.S_to_psi(self.S))
        self.cum_irrig = 0.0
        self.cum_drain = 0.0
        self.events_count = 0
        return self._obs(), {}

    def step(self, action):
        """
        Exécute une étape de simulation : applique l'action (irrigation) et fait évoluer l'état.
        
        ÉTAPES DE LA SIMULATION :
        1. Clippe l'action (dose d'irrigation) dans [0, max_irrigation]
        2. Récupère les données météo du jour t
        3. Calcule l'évapotranspiration culturelle ETc = Kc × ET0 × f_ET(ψ)
        4. Calcule le drainage D si S > S_fc
        5. Met à jour la réserve : S_{t+1} = S_t + η_I × I_t + R_t - ETc - D
        6. Convertit S_{t+1} en ψ_{t+1} via courbe de rétention
        7. Calcule la récompense (pénalité stress + pénalité eau)
        8. Vérifie si l'épisode est terminé
        
        FONCTION DE RÉCOMPENSE :
        r_t = -|ψ_t - clip(ψ_t, 20, 60)| / 10.0 - 0.05 × I_t
        - Premier terme : pénalise l'écart de ψ par rapport à la zone de confort [20, 60] cbar
        - Second terme : pénalise la quantité d'eau utilisée (coût)
        
        Args:
            action (np.ndarray ou float): Dose d'irrigation choisie par l'agent (mm)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info) où :
                - observation (np.ndarray): Nouvel état [psi, rain, et0, Kc]
                - reward (float): Récompense instantanée
                - terminated (bool): True si la saison est terminée
                - truncated (bool): False (non utilisé ici)
                - info (dict): Informations supplémentaires (I, rain, ETc, D)
        """
        # Clippage de l'action dans les limites autorisées
        # Gère les cas où l'action est un array ou un scalaire
        action = float(np.clip(action[0] if hasattr(action, "__len__") else action, 0.0, self.max_irrigation))
        t = self.day
        
        # Récupération des données météorologiques du jour t
        rain_t, et0_t, Kc_t = float(self.rain[t]), float(self.et0[t]), float(self.Kc[t])

        # Calcul de l'évapotranspiration culturelle ETc
        # ETc = Kc × ET0 × f_ET(ψ) où :
        # - Kc : coefficient cultural (demande de la culture selon son stade)
        # - ET0 : évapotranspiration de référence (demande climatique)
        # - f_ET(ψ) : facteur de stress hydrique (réduction si sol sec)
        fET = self.soil.f_ET(self.psi)
        ETc = Kc_t * et0_t * fET
        
        # Calcul du drainage (se produit si S > S_fc)
        # Le drainage représente l'eau qui percole vers les couches profondes
        D = float(self.soil.drainage(self.S))

        # Mise à jour du bilan hydrique : S_{t+1} = S_t + η_I × I_t + R_t - ETc - D
        # η_I : efficacité d'irrigation (fraction de l'eau qui atteint la zone racinaire)
        # Les pertes sont dues à l'évaporation, au ruissellement, à l'inefficacité du système
        S_next = np.clip(self.S + self.soil.eta_I * action + rain_t - ETc - D, 0.0, self.soil.S_max)
        
        # Conversion de la réserve en tension matricielle
        # C'est cette valeur qui sera observée par l'agent au pas suivant
        psi_next = float(self.soil.S_to_psi(S_next))

        # Calcul de la récompense instantanée
        # Pénalité 1 : écart de ψ par rapport à la zone de confort [20, 60] cbar
        #   Zone optimale pour la croissance des plantes
        # Pénalité 2 : quantité d'eau utilisée (coût économique/environnemental)
        #   Encourage l'économie d'eau
        reward = -abs(psi_next - np.clip(psi_next, 20.0, 60.0)) / 10.0 - 0.05 * action
        # Compteurs cumulatifs pour les déviations lexicographiques
        self.cum_irrig += action
        self.cum_drain += D
        if action > 0.0:
            self.events_count += 1

        # Mise à jour de l'état pour le prochain pas de temps
        self.S, self.psi = S_next, psi_next
        self.day += 1
        
        # Vérification de la fin de l'épisode
        terminated = self.day >= self.season_length
        truncated = False
        
        # Informations supplémentaires pour le débogage et l'analyse
        deviations = self._lexico_deviations(psi_next)
        info = {"I": action, "rain": rain_t, "ETc": ETc, "D": D}
        if deviations is not None:
            info["lexico_deviations"] = deviations

        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        t = min(self.day, self.season_length - 1)
        return np.array([self.psi, self.rain[t], self.et0[t], self.Kc[t]], dtype=np.float32)

    def _lexico_deviations(self, psi_value: float):
        """
        Calcule les déviations lexicographiques à partir de goal_spec (si fourni).
        Renvoie une liste [d_P1, d_P2, d_P3] ou None si non configuré.
        """
        if not self.goal_spec:
            return None
        targets = self.goal_spec.get(
            "targets",
            {"stress_max": 55.0, "irrig_max": 250.0, "drain_max": 60.0, "events_max": 20},
        )
        priorities = self.goal_spec.get(
            "priorities",
            {"P1": ["stress"], "P2": ["drainage"], "P3": ["irrigation"]},
        )
        deviations_map = {
            "stress": max(0.0, psi_value - targets.get("stress_max", 55.0)),
            "irrigation": max(0.0, self.cum_irrig - targets.get("irrig_max", 250.0)),
            "drainage": max(0.0, self.cum_drain - targets.get("drain_max", 60.0)),
            "events": max(0.0, self.events_count - targets.get("events_max", 20)),
        }
        result = []
        for tier in ("P1", "P2", "P3"):
            objs = priorities.get(tier, [])
            tier_dev = sum(deviations_map.get(obj, 0.0) for obj in objs)
            result.append(tier_dev)
        return result
