"""
Utilitaires pour l'environnement Gymnasium avec modèles résiduels.

Ce module contient :
- IrrigationEnvPhysical : Environnement Gymnasium pour l'apprentissage par renforcement
  avec support des modèles résiduels (Neural ODE, Neural CDE) pour les scénarios 3, 4, 5
"""

import numpy as np
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore

import torch  # type: ignore
import torch.nn as nn  # type: ignore

from typing import Optional, Dict, Any


# %% — Hybrid Gymnasium Environment
class IrrigationEnvPhysical(gym.Env):
    """
    Irrigation environment based on a physical water balance model,
    with *option* to add a learned residual on tension (ψ) via:

      - Neural ODE (NNODE) one-step model: Δψ_t = f_θ(ψ_t, I_t, R_t, ET0_t)
      - Discrete Neural CDE (NNCDE) model: Δψ_t = f_θ({X_{t-L+1},...,X_t}),
        where X_k = (ψ_k, I_k, R_k, ET0_k)

    If no model is provided, we get Scenario 1 (Pure Physics).
    If an NNODE model is provided (residual_ode), we get Scenario 3.
    If an NNCDE model is provided (residual_cde), we get Scenario 4.

    API: Gymnasium
      - reset(seed, options) -> obs, info
      - step(action) -> obs, reward, terminated, truncated, info
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        season_length: int = 120,
        max_irrigation: float = 20.0,
        seed: int = 123,
        weather_params: Optional[Dict[str, Any]] = None,
        residual_ode: Optional[nn.Module] = None,
        residual_cde: Optional[nn.Module] = None,
        seq_len_cde: int = 5,
        device: str = "cpu",
        reward_cfg: Optional[Dict[str, float]] = None,
        soil: Optional[Dict[str, float]] = None,
        hazard_cfg: Optional[Dict[str, Any]] = None,
        weather_shift_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # RNG
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = device

        # Season length (duration in days)
        self.T = season_length

        # Default soil parameters (can be modified via `soil` argument)
        # Soil water balance parameters (all in mm except psi which is in centibars)
        soil_defaults = dict(
            S_max=120.0,   # S_max: Maximum soil water storage capacity (mm)
            #              # This is the total water-holding capacity of the soil profile
            S_fc=90.0,     # S_fc: Field capacity - maximum water content after drainage (mm)
            #              # At field capacity, soil is at optimal moisture for plant growth
            S_wp=30.0,     # S_wp: Wilting point - minimum water content before plant wilting (mm)
            #              # Below this point, plants cannot extract water effectively
            psi_sat=10.0,  # psi_sat: Soil water tension at saturation (centibars, cbar)
            #              # Low tension (near 0) = saturated soil, water easily available
            psi_fc=30.0,   # psi_fc: Soil water tension at field capacity (cbar)
            #              # Optimal tension range for plant growth is typically 30-60 cbar
            psi_wp=150.0,  # psi_wp: Soil water tension at wilting point (cbar)
            #              # High tension = dry soil, water is difficult for plants to extract
        )
        if soil is not None:
            soil_defaults.update(soil)
        self.S_max = soil_defaults["S_max"]
        self.S_fc = soil_defaults["S_fc"]
        self.S_wp = soil_defaults["S_wp"]
        self.psi_sat = soil_defaults["psi_sat"]
        self.psi_fc = soil_defaults["psi_fc"]
        self.psi_wp = soil_defaults["psi_wp"]

        # Paramètres de reward (Reward function parameters)
        # The reward function has two components:
        # 1. Daily reward: reward_t = -alpha * stress(ψ_t) - beta * I_t - gamma * D_t
        #    - Penalizes plant stress (alpha), irrigation cost (beta), and drainage loss (gamma)
        # 2. Terminal reward: reward_terminal = Y(cum_stress) - lambda_water * cum_irrig
        #    - Rewards final crop yield Y (decreases with cumulative stress)
        #    - Penalizes total water usage (lambda_water)
        reward_defaults = dict(
            alpha=1.0,        # alpha: Weight for plant stress penalty (higher = more important to avoid stress)
            beta=0.05,       # beta: Weight for irrigation cost penalty (water conservation)
            gamma=0.01,      # gamma: Weight for drainage loss penalty (water efficiency)
            lambda_water=0.001,  # lambda_water: Weight for total water usage penalty in terminal reward
            Y_max=1.0,       # Y_max: Maximum possible crop yield (normalized, typically 0-1)
            k_yield=0.01,    # k_yield: Yield decay rate with cumulative stress
            #                # Yield formula: Y = Y_max * exp(-k_yield * cum_stress)
        )
        if reward_cfg is not None:
            reward_defaults.update(reward_cfg)
        self.alpha = reward_defaults["alpha"]
        self.beta = reward_defaults["beta"]
        self.gamma = reward_defaults["gamma"]
        self.lambda_water = reward_defaults["lambda_water"]
        self.Y_max = reward_defaults["Y_max"]
        self.k_yield = reward_defaults["k_yield"]

        # Efficacité d'irrigation (Irrigation efficiency)
        # eta_I: Fraction of applied irrigation that actually reaches the soil
        # Accounts for losses due to evaporation, runoff, or system inefficiency
        # eta_I = 0.8 means 80% of applied water is effective, 20% is lost
        self.eta_I = 0.8

        # Irrigation max par pas de temps
        self.I_max = max_irrigation

        # State variables
        self.day = 0  # Current day in the season
        self.S = self.S_fc  # Soil water storage (mm), initialized at field capacity
        self.psi = self._S_to_psi(self.S)  # Soil water tension (cbar), derived from S

        # Cumulative values for reward calculation
        self.cum_stress = 0.0  # Cumulative plant water stress over the season
        self.cum_irrig = 0.0  # Cumulative irrigation applied over the season
        self.cum_drain = 0.0  # Cumulative drainage (mm)
        self.events_count = 0  # Number of irrigation events
        self.weather_shift_cfg = weather_shift_cfg or {}

        # Hazard events configuration (MUST BE BEFORE weather generation)
        hazard_defaults = dict(
            enable_hazards=False,
            drought_prob=0.0,          # Probability of drought period per season
            drought_duration=7,        # Days of drought
            drought_rain_mult=0.1,     # Rain multiplier during drought (0.1 = 10% normal)
            drought_et0_mult=1.5,      # ET0 multiplier during drought (1.5 = 150% normal)
            
            flood_prob=0.0,            # Probability of flood event per season
            flood_duration=3,          # Days of flood
            flood_rain_mult=5.0,       # Rain multiplier during flood
            
            heatwave_prob=0.0,         # Probability of heatwave per season
            heatwave_duration=5,       # Days of heatwave
            heatwave_et0_mult=1.8,     # ET0 multiplier during heatwave
            
            equipment_failure_prob=0.0,  # Probability of equipment failure
            equipment_failure_duration=5, # Days without irrigation capability
            equipment_failure_delay=True, # Can predict failure 2 days before?
            
            water_restriction_prob=0.0,   # Probability of water restrictions
            water_restriction_duration=10, # Days with restrictions
            water_restriction_limit=0.5,   # Max irrigation multiplier (0.5 = 50% of normal)
        )
        if hazard_cfg is not None:
            hazard_defaults.update(hazard_cfg)
        
        self.hazard_cfg = hazard_defaults
        self.enable_hazards = hazard_defaults["enable_hazards"]
        self.active_hazards = {}  # {day: [hazard_type, remaining_days]}
        self.hazard_history = []  # Track hazard occurrences

        # Weather configuration (used to align scenarios on the same generator)
        self.weather_params = weather_params

        # Weather series and crop coefficients
        self.rain = None  # Daily rainfall series (mm)
        self.et0 = None  # Daily reference evapotranspiration series (mm/day)
        self.Kc = None  # Daily crop coefficient series (dimensionless)
        self._generate_weather()
        
        # Generate hazard events if enabled
        if self.enable_hazards:
            self._generate_hazard_events()

        # Residual models (NNODE / NNCDE) - optional learned corrections to physical model
        self.residual_ode = residual_ode.to(device) if residual_ode is not None else None
        if self.residual_ode is not None:
            self.residual_ode.eval()

        self.residual_cde = residual_cde.to(device) if residual_cde is not None else None
        if self.residual_cde is not None:
            self.residual_cde.eval()

        self.seq_len_cde = seq_len_cde  # Sequence length for CDE model
        self.history_cde = []  # History list of X_k = [ψ_k, I_k, R_k, ET0_k] for CDE model

        # Action space: scalar irrigation between 0 and I_max
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.I_max], dtype=np.float32),
            dtype=np.float32,
        )

        # Espace d'observation (Observation space definition)
        # obs_t = [ψ_t, S_t, R_t (jour courant), ET0_t (jour courant)]
        # Observation vector contains 4 values:
        # - ψ_t: Current soil water tension (cbar) - indicates water availability
        # - S_t: Current soil water storage (mm) - total water in soil
        # - R_t: Current day's rainfall (mm) - water input from precipitation
        # - ET0_t: Current day's reference evapotranspiration (mm/day) - water demand
        low = np.array(
            [self.psi_sat, 0.0, 0.0, 0.0],  # Minimum values for each observation
            dtype=np.float32,
        )
        high = np.array(
            [self.psi_wp, self.S_max, 100.0, 15.0],  # Maximum values for each observation
            # psi_wp: Maximum tension (wilting point)
            # S_max: Maximum soil storage capacity
            # 100.0: Maximum expected daily rainfall (mm)
            # 15.0: Maximum expected daily ET0 (mm/day)
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    #  Physical Helper Functions
    # ------------------------------------------------------------------

    def _generate_weather(self):
        """
        Generates simplified weather series: rainfall (R), reference evapotranspiration (ET0),
        and crop coefficient (Kc) for the entire season.
        
        Two methods are available:
        1. Shared generator (utils_weather.generate_weather): Used when weather_params is provided
           - ET0: Sinusoidal seasonal variation + noise (matches notebooks/Scenario 1)
           - Rain: Phase-based probabilities (early/mid/late season)
           - Kc: Fixed day-based phases (consistent across scenarios)
           This ensures consistency when comparing Scenario 1, 3, 4
        
        2. Internal fallback: Used when weather_params is None
           - ET0: Normal distribution (no seasonal variation)
           - Rain: Gamma distribution
           - Kc: Percentage-based phases (15%/25%/40%/20%)
           This is a simpler fallback method
        """
        # Use shared weather generator when weather_params is provided (even if empty dict)
        # This ensures weather consistency across all scenarios for fair comparison
        # Pass weather_params={} to use default parameters from shared generator
        # Pass weather_params=None to use internal fallback (gamma/normal distributions)
        # 
        # CRITICAL FOR CONSISTENCY: 
        # - Scenarios 1, 2, 3, 4, 5, 6 should all use weather_params={} to use shared generator
        # - This ensures identical weather patterns when comparing scenarios with same seed
        if self.weather_params is not None:
            try:
                from src.utils_weather import generate_weather
            except ImportError:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from src.utils_weather import generate_weather
            rain, et0, Kc = generate_weather(
                T=self.T,
                seed=self.seed,
                **self.weather_params
            )
            self.rain = np.asarray(rain, dtype=np.float32)
            self.et0 = np.asarray(et0, dtype=np.float32)
            self.Kc = np.asarray(Kc, dtype=np.float32)
            try:
                from src.utils_weather_shift import apply_weather_shift
                self.rain, self.et0 = apply_weather_shift(self.rain, self.et0, self.weather_shift_cfg, rng=self.rng)
            except Exception:
                pass
            return

        # ==================================================================
        # FALLBACK WEATHER GENERATION (when weather_params is None)
        # ==================================================================
        # NOTE: This method differs from the shared generator in utils_weather.py
        # For consistency across scenarios, prefer using weather_params={} to use shared generator
        T = self.T  # Season length in days
        
        # Rainfall (mm) - intermittent events using GAMMA distribution
        # Uses gamma distribution to model rainfall events (skewed, many zeros, occasional heavy rain)
        # shape=0.8, scale=3.0 creates realistic rainfall pattern with most days dry
        # DIFFERENCE from shared generator: Shared uses phase-based probabilities (early/mid/late)
        self.rain = self.rng.gamma(shape=0.8, scale=3.0, size=T)
        # Set very small values (< 1mm) to zero (negligible rainfall)
        self.rain[self.rain < 1.0] = 0.0

        # ET0 (mm/j) - Reference evapotranspiration (mm/day) using NORMAL distribution
        # ET0 represents the maximum water loss from a reference crop (grass)
        # under standard conditions. It depends on temperature, humidity, wind, solar radiation
        # Normal distribution with mean=4.0 mm/day, std=0.8 mm/day
        # DIFFERENCE from shared generator: Shared uses SINUSOIDAL seasonal variation + noise
        self.et0 = self.rng.normal(loc=4.0, scale=0.8, size=T)
        # Clip to realistic range: 2-6 mm/day (typical for temperate climates)
        self.et0 = np.clip(self.et0, 2.0, 6.0)

        # Coefficients culturaux (Crop coefficients Kc - 4 growth phases)
        # Kc relates actual crop evapotranspiration (ETc) to reference ET0: ETc = Kc * ET0
        # Kc varies with crop growth stage:
        Kc_ini = 0.6   # Initial stage: small plants, low water demand
        Kc_mid = 1.15  # Mid-season: full canopy, maximum water demand
        Kc_end = 0.7   # Late season: maturing crop, reduced water demand
        
        # Phase lengths as PERCENTAGE of total season length T
        # DIFFERENCE from shared generator: Shared uses FIXED day thresholds (20, 50, 90)
        # This method uses percentages, so phases scale with season length
        L_ini = int(0.15 * T)   # Initial phase: 15% of season
        L_dev = int(0.25 * T)   # Development phase: 25% of season (transition)
        L_mid = int(0.4 * T)    # Mid-season phase: 40% of season (peak demand)
        L_end = T               # End phase: remaining 20% of season
        
        # Build Kc array with linear transitions between phases
        Kc = np.zeros(T)
        Kc[:L_ini] = Kc_ini  # Initial phase: constant low Kc
        # Development: linear increase from Kc_ini to Kc_mid
        Kc[L_ini:L_ini+L_dev] = np.linspace(Kc_ini, Kc_mid, L_dev)
        # Mid-season: constant high Kc (peak water demand)
        Kc[L_ini+L_dev:L_ini+L_dev+L_mid] = Kc_mid
        # Late season: linear decrease from Kc_mid to Kc_end
        Kc[L_ini+L_dev+L_mid:] = np.linspace(Kc_mid, Kc_end, T - (L_ini+L_dev+L_mid))
        self.Kc = Kc

        # Appliquer un éventuel décalage météo (robustesse)
        try:
            from src.utils_weather_shift import apply_weather_shift
            self.rain, self.et0 = apply_weather_shift(self.rain, self.et0, self.weather_shift_cfg, rng=self.rng)
        except Exception:
            pass

    def _generate_hazard_events(self):
        """
        Generate hazard events for the season.
        
        Creates a schedule of when hazards occur and their duration.
        Each hazard type has an independent probability of occurring.
        Hazards are randomly scheduled but avoid overlapping (where possible).
        
        This function is called at the start of each episode (in reset())
        to create a new random hazard scenario.
        """
        # Initialize hazard tracking structures
        self.active_hazards = {}  # Dictionary mapping start_day -> hazard info
        # Format: {start_day: {"type": str, "remaining": int, ...}}
        self.hazard_history = []  # List of all hazard events (for logging/analysis)
        cfg = self.hazard_cfg  # Short reference to configuration
        
        # Skip hazard generation if disabled
        if not self.enable_hazards:
            return
        
        # Generate drought periods
        # Drought: Extended period with reduced rainfall and increased ET0
        if cfg["drought_prob"] > 0:
            # Randomly decide if drought occurs this season
            if self.rng.random() < cfg["drought_prob"]:
                # Randomly schedule drought start day
                # Avoid very early (day < 20) and very late (wouldn't fit duration)
                start_day = self.rng.integers(20, self.T - cfg["drought_duration"] - 10)
                
                # Store hazard in active_hazards dictionary
                self.active_hazards[start_day] = {
                    "type": "drought",
                    "remaining": cfg["drought_duration"],  # Days left in hazard
                    "original_start": start_day  # Track original start for logging
                }
                
                # Add to history for episode summary
                self.hazard_history.append({
                    "type": "drought",
                    "start": start_day,
                    "duration": cfg["drought_duration"]
                })
        
        # Generate flood events
        # Flood: Short period with excessive rainfall (risk of waterlogging)
        if cfg["flood_prob"] > 0:
            if self.rng.random() < cfg["flood_prob"]:
                # Schedule flood later in season (after day 30)
                start_day = self.rng.integers(30, self.T - cfg["flood_duration"] - 5)
                
                # Check for overlap with existing hazards (e.g., drought)
                # Prevents conflicting weather conditions
                overlap = any(start_day <= h <= start_day + cfg["flood_duration"] 
                             for h in self.active_hazards.keys())
                
                # Only add flood if it doesn't overlap with other hazards
                if not overlap:
                    self.active_hazards[start_day] = {
                        "type": "flood",
                        "remaining": cfg["flood_duration"],
                        "original_start": start_day
                    }
                    self.hazard_history.append({
                        "type": "flood",
                        "start": start_day,
                        "duration": cfg["flood_duration"]
                    })
        
        # Generate heatwaves
        # Heatwave: Period with significantly increased ET0 (higher water demand)
        if cfg["heatwave_prob"] > 0:
            if self.rng.random() < cfg["heatwave_prob"]:
                # Schedule heatwave mid-to-late season (after day 40)
                start_day = self.rng.integers(40, self.T - cfg["heatwave_duration"] - 5)
                # Check for overlap with other hazards
                overlap = any(start_day <= h <= start_day + cfg["heatwave_duration"] 
                             for h in self.active_hazards.keys())
                if not overlap:
                    self.active_hazards[start_day] = {
                        "type": "heatwave",
                        "remaining": cfg["heatwave_duration"],
                        "original_start": start_day
                    }
                    self.hazard_history.append({
                        "type": "heatwave",
                        "start": start_day,
                        "duration": cfg["heatwave_duration"]
                    })
        
        # Generate equipment failure
        # Equipment failure: Irrigation system breaks down (can't irrigate)
        if cfg["equipment_failure_prob"] > 0:
            if self.rng.random() < cfg["equipment_failure_prob"]:
                # Schedule failure mid-to-late season (after day 50)
                start_day = self.rng.integers(50, self.T - cfg["equipment_failure_duration"] - 5)
                self.active_hazards[start_day] = {
                    "type": "equipment_failure",
                    "remaining": cfg["equipment_failure_duration"],
                    "original_start": start_day,
                    # Optional warning period: agent knows failure is coming
                    "warning_start": start_day - 2 if cfg["equipment_failure_delay"] else start_day
                }
                self.hazard_history.append({
                    "type": "equipment_failure",
                    "start": start_day,
                    "duration": cfg["equipment_failure_duration"],
                    "warning_day": start_day - 2 if cfg["equipment_failure_delay"] else None
                })
        
        # Generate water restrictions
        # Water restrictions: Limited irrigation capacity (conservation measures)
        if cfg["water_restriction_prob"] > 0:
            if self.rng.random() < cfg["water_restriction_prob"]:
                # Schedule restrictions late season (after day 60)
                start_day = self.rng.integers(60, self.T - cfg["water_restriction_duration"] - 5)
                self.active_hazards[start_day] = {
                    "type": "water_restriction",
                    "remaining": cfg["water_restriction_duration"],
                    "original_start": start_day
                }
                self.hazard_history.append({
                    "type": "water_restriction",
                    "start": start_day,
                    "duration": cfg["water_restriction_duration"]
                })

    def _apply_hazards(self, day: int, R_t: float, ET0_t: float, I_t: float) -> tuple:
        """
        Apply active hazard effects to weather and irrigation.
        
        Returns: (modified_R, modified_ET0, modified_I, active_hazard_types)
        """
        if not self.enable_hazards:
            return R_t, ET0_t, I_t, []
        
        modified_R = R_t
        modified_ET0 = ET0_t
        modified_I = I_t
        active_types = []
        
        # Check for hazards that should be active today
        hazards_to_process = []
        
        # Check all scheduled hazards
        for hazard_start in list(self.active_hazards.keys()):
            hazard = self.active_hazards[hazard_start]
            hazard_day = hazard_start
            
            # Check if this hazard is active today
            # Active if: (day >= hazard_start) and (day < hazard_start + original_duration)
            # But we track by remaining days, so check if remaining > 0 and we're in the period
            if day >= hazard_day and hazard["remaining"] > 0:
                hazards_to_process.append(hazard_start)
        
        # Apply hazard effects (can have multiple active at once)
        for hazard_start in hazards_to_process:
            hazard = self.active_hazards[hazard_start]
            
            # Check for equipment failure warning (2 days before actual failure)
            if hazard["type"] == "equipment_failure" and self.hazard_cfg["equipment_failure_delay"]:
                warning_day = hazard.get("warning_start", hazard_start)
                if day == warning_day and day < hazard_start:
                    # Warning period - no effect yet, but agent should know
                    active_types.append("equipment_failure_warning")
                    continue
            
            # Only apply effects if hazard is actually active (not just warning)
            if day >= hazard_start:
                # Apply hazard effects
                if hazard["type"] == "drought":
                    modified_R *= self.hazard_cfg["drought_rain_mult"]
                    modified_ET0 *= self.hazard_cfg["drought_et0_mult"]
                    active_types.append("drought")
                
                elif hazard["type"] == "flood":
                    modified_R *= self.hazard_cfg["flood_rain_mult"]
                    active_types.append("flood")
                
                elif hazard["type"] == "heatwave":
                    modified_ET0 *= self.hazard_cfg["heatwave_et0_mult"]
                    active_types.append("heatwave")
                
                elif hazard["type"] == "equipment_failure":
                    modified_I = 0.0  # No irrigation possible
                    active_types.append("equipment_failure")
                
                elif hazard["type"] == "water_restriction":
                    modified_I *= self.hazard_cfg["water_restriction_limit"]
                    active_types.append("water_restriction")
                
                # Decrement remaining days (only once per day)
                hazard["remaining"] -= 1
        
        # Clean up expired hazards
        self.active_hazards = {
            k: v for k, v in self.active_hazards.items() 
            if v["remaining"] > 0
        }
        
        return modified_R, modified_ET0, modified_I, active_types

    def _S_to_psi(self, S: float) -> float:
        """
        Relationship S -> ψ (simplified and monotonic).
        Converts soil water storage S (mm) to soil water tension ψ (centibars).
        
        Physical relationship: As soil water content (S) decreases, 
        water tension (ψ) increases (water becomes harder to extract).
        
        Uses piecewise linear interpolation between key points:
        - (S_wp, psi_wp): Wilting point - driest condition
        - (S_fc, psi_fc): Field capacity - optimal condition
        - (S_max, psi_sat): Saturation - wettest condition
        
        Args:
            S: Soil water storage (mm), range [0, S_max]
        Returns:
            psi: Soil water tension (centibars), range [psi_sat, psi_wp]
        """
        S = float(np.clip(S, 0.0, self.S_max))
        
        if S >= self.S_fc:
            # Wet zone: S between field capacity and saturation
            # ψ decreases from psi_fc to psi_sat as S increases
            # Linear interpolation: ψ = psi_fc + (psi_sat - psi_fc) * (S - S_fc) / (S_max - S_fc)
            if self.S_max > self.S_fc:
                psi = self.psi_fc + (self.psi_sat - self.psi_fc) * (S - self.S_fc) / (self.S_max - self.S_fc)
            else:
                psi = self.psi_fc
        elif S <= self.S_wp:
            # Very dry zone: S at or below wilting point
            # Maximum tension - water is very difficult to extract
            psi = self.psi_wp
        else:
            # Intermediate zone: S between wilting point and field capacity
            # Linear interpolation: ψ = psi_wp + (psi_fc - psi_wp) * (S - S_wp) / (S_fc - S_wp)
            # As S increases, ψ decreases (water becomes easier to extract)
            psi = self.psi_wp + (self.psi_fc - self.psi_wp) * (S - self.S_wp) / (self.S_fc - self.S_wp)
        
        return float(np.clip(psi, self.psi_sat, self.psi_wp))

    def _psi_to_S(self, psi: float) -> float:
        """
        Inverse approximatif de _S_to_psi.
        Inverse function: converts soil water tension ψ (centibars) to soil water storage S (mm).
        
        This is the inverse of _S_to_psi, used to convert tension back to storage.
        Uses the same piecewise linear relationship but inverted.
        
        Args:
            psi: Soil water tension (centibars), range [psi_sat, psi_wp]
        Returns:
            S: Soil water storage (mm), range [0, S_max]
        """
        psi = float(np.clip(psi, self.psi_sat, self.psi_wp))
        
        # Extreme cases: maximum tension = minimum water, minimum tension = maximum water
        if psi >= self.psi_wp:
            return self.S_wp  # Maximum tension -> wilting point (minimum usable water)
        if psi <= self.psi_sat:
            return self.S_max  # Minimum tension -> saturation (maximum water)

        if psi <= self.psi_fc:
            # Wet zone: Invert the relationship (S_fc, psi_fc) -> (S_max, psi_sat)
            # As psi decreases (toward saturation), S increases
            if self.psi_fc > self.psi_sat:
                S = self.S_fc + (self.S_max - self.S_fc) * (psi - self.psi_fc) / (self.psi_sat - self.psi_fc)
            else:
                S = self.S_fc
        else:
            # Intermediate zone: Invert between (S_wp, psi_wp) and (S_fc, psi_fc)
            # As psi decreases (toward field capacity), S increases
            S = self.S_wp + (self.S_fc - self.S_wp) * (psi - self.psi_wp) / (self.psi_fc - self.psi_wp)
        
        return float(np.clip(S, 0.0, self.S_max))

    def _f_ET(self, psi: float) -> float:
        """
        Reduction of ETc as a function of tension ψ.
        Reduction factor for crop evapotranspiration (ETc) based on soil water tension ψ.
        
        Physical principle: When soil is dry (high ψ), plants cannot extract water easily,
        so actual ETc is reduced compared to potential ETc.
        
        Formula: ETc_actual = Kc * ET0 * f_ET(psi)
        - f_ET = 1.0 in optimal zone (30-60 cbar): full water availability
        - f_ET < 1.0 outside optimal zone: reduced water extraction
        
        Args:
            psi: Soil water tension (centibars)
        Returns:
            f_ET: Reduction factor [0, 1], where 1.0 = full ETc, 0.0 = no ETc
        """
        psi = float(psi)
        
        if 30.0 <= psi <= 60.0:
            # Optimal zone: Full evapotranspiration (no reduction)
            # Plants can extract water at maximum rate
            return 1.0
        elif psi < 30.0:
            # Too wet (below optimal): Slight reduction
            # Linear reduction from 1.0 at psi=30 to 0.5 at psi=0
            # f_ET = 0.5 + 0.5 * (psi / 30.0)
            return 0.5 + 0.5 * (psi / 30.0)
        else:
            # Too dry (above optimal, psi > 60): Significant reduction
            # Linear reduction from 1.0 at psi=60 to 0.1 at psi=psi_wp
            # x = normalized distance from 60 to psi_wp
            x = (psi - 60.0) / (self.psi_wp - 60.0)
            # f_ET decreases from 1.0 to 0.1 (90% reduction at wilting point)
            return float(max(0.0, 1.0 - 0.9 * x))

    def _drainage(self, S: float) -> float:
        """
        Simplified drainage: proportional to excess above S_fc.
        Simplified drainage model: water loss due to gravity when soil exceeds field capacity.
        
        Physical principle: When soil water content exceeds field capacity (S > S_fc),
        excess water drains due to gravity. Drainage is proportional to the excess.
        
        Formula: D = k_drain * max(0, S - S_fc)
        - D = 0 if S <= S_fc (no excess water)
        - D > 0 if S > S_fc (excess water drains)
        
        Args:
            S: Current soil water storage (mm)
        Returns:
            D: Drainage amount (mm/day)
        """
        S = float(S)
        if S <= self.S_fc:
            # No drainage if water content is at or below field capacity
            return 0.0
        
        # Drainage coefficient: fraction of excess water that drains per day
        # k_drain = 0.1 means 10% of excess water drains each day
        k_drain = 0.1
        # Drainage is proportional to excess water above field capacity
        return k_drain * (S - self.S_fc)

    def _stress(self, psi: float) -> float:
        """
        Water stress indicator based on deviation from comfort zone [30, 60] cbar.
        Plant water stress indicator based on deviation from optimal tension zone [30, 60] cbar.
        
        Stress represents how far the soil conditions are from optimal for plant growth.
        - Stress = 0.0 in optimal zone (30-60 cbar): no stress
        - Stress > 0.0 outside optimal zone: increasing stress
        - Higher stress reduces crop yield
        
        Formula:
        - If 30 <= psi <= 60: stress = 0 (optimal)
        - If psi < 30: stress = (30 - psi) / 30 (too wet, normalized distance from lower bound)
        - If psi > 60: stress = (psi - 60) / 60 (too dry, normalized distance from upper bound)
        
        Args:
            psi: Soil water tension (centibars)
        Returns:
            stress: Water stress indicator [0, inf), where 0 = no stress
        """
        psi = float(psi)
        
        if 30.0 <= psi <= 60.0:
            # Optimal zone: no stress
            return 0.0
        
        if psi < 30.0:
            # Too wet: stress increases as psi decreases (approaches saturation)
            # Normalized: stress = 1.0 at psi = 0 (saturation), stress = 0 at psi = 30
            return (30.0 - psi) / 30.0
        else:
            # Too dry: stress increases as psi increases (approaches wilting point)
            # Normalized: stress = 0 at psi = 60, stress increases toward infinity as psi -> psi_wp
            return (psi - 60.0) / 60.0

    def _yield(self, cum_stress: float) -> float:
        """
        Yield function at end of season (decreases with cumulative stress).
        Crop yield function at end of season (decreases with cumulative stress).
        
        Physical principle: Cumulative water stress throughout the season reduces final crop yield.
        Uses exponential decay model: yield decreases exponentially with accumulated stress.
        
        Formula: Y = Y_max * exp(-k_yield * cum_stress)
        - Y_max: Maximum possible yield (normalized, typically 0-1)
        - k_yield: Yield decay rate (higher = more sensitive to stress)
        - cum_stress: Sum of daily stress values over the season
        
        Example: If cum_stress = 10 and k_yield = 0.01:
        Y = 1.0 * exp(-0.01 * 10) = exp(-0.1) ≈ 0.905 (90.5% of max yield)
        
        Args:
            cum_stress: Cumulative water stress over the season (sum of daily stress values)
        Returns:
            Y: Final crop yield [0, Y_max], normalized value
        """
        return float(self.Y_max * np.exp(-self.k_yield * cum_stress))

    def _build_obs(self) -> np.ndarray:
        """
        Observation : [ψ_t, S_t, R_t, ET0_t]
        Builds the observation vector for the RL agent.
        
        The observation contains all information the agent needs to make decisions:
        - Current soil state (tension and storage)
        - Current weather conditions (rainfall and evapotranspiration demand)
        
        Returns:
            obs: Observation array [psi, S, R, ET0] of shape (4,)
        """
        # Get current day index (ensure within bounds)
        t = min(self.day, self.T - 1)
        
        # Get current day's weather data
        R_t = self.rain[t] if t < len(self.rain) else 0.0  # Rainfall (mm)
        ET0_t = self.et0[t] if t < len(self.et0) else 0.0  # Reference ET (mm/day)
        
        # Build observation vector: [soil tension, soil storage, rainfall, ET0]
        obs = np.array([self.psi, self.S, R_t, ET0_t], dtype=np.float32)
        return obs

    def _build_features_cde(self, psi: float, I: float, R: float, ET0: float) -> np.ndarray:
        """
        Build feature vector for Neural CDE (Controlled Differential Equation) model.
        
        The CDE model requires a time series of features. This function creates
        a single feature vector X_t = [ψ_t, I_t, R_t, ET0_t] at time t.
        
        These features represent:
        - psi: Soil water tension (state variable)
        - I: Irrigation action (control input)
        - R: Rainfall (external input)
        - ET0: Reference evapotranspiration (external input)
        
        Args:
            psi: Soil water tension (cbar)
            I: Irrigation amount (mm)
            R: Rainfall (mm)
            ET0: Reference evapotranspiration (mm/day)
        
        Returns:
            Feature array of shape (4,) with dtype float32
        """
        return np.array([psi, I, R, ET0], dtype=np.float32)

    def _build_cde_sequence_tensor(self) -> torch.Tensor:
        """
        Build a sequence tensor [1, seq_len_cde, 4] from self.history_cde.
        
        Neural CDE models require a sequence of historical features to predict
        the residual correction. This function constructs a tensor containing
        the last seq_len_cde time steps of features [ψ, I, R, ET0].
        
        If history is shorter than seq_len_cde, the beginning is padded with
        the first available feature vector (repeating initial state).
        
        Tensor shape: [batch_size=1, sequence_length, features=4]
        - batch_size=1: Single sequence (one episode)
        - sequence_length: seq_len_cde (typically 5)
        - features=4: [psi, I, R, ET0]
        
        Returns:
            torch.Tensor: Sequence tensor of shape [1, seq_len_cde, 4]
                Device location matches self.device (cpu or cuda)
        """
        seq = self.history_cde  # List of feature vectors [[psi, I, R, ET0], ...]
        
        # Handle case where history is shorter than required sequence length
        if len(seq) < self.seq_len_cde:
            # Pad beginning with first element (repeat initial state)
            # This assumes the initial state is representative of early season
            pad_elem = seq[0] if len(seq) > 0 else np.zeros(4, dtype=np.float32)
            pad_count = self.seq_len_cde - len(seq)
            seq = [pad_elem] * pad_count + seq  # Prepend padding
        else:
            # Take only the last seq_len_cde elements (most recent history)
            seq = seq[-self.seq_len_cde:]

        # Convert list of arrays to numpy array: [seq_len_cde, 4]
        seq_np = np.stack(seq, axis=0)
        
        # Add batch dimension: [1, seq_len_cde, 4]
        # Neural networks expect batch dimension even for single sequences
        seq_np = seq_np[np.newaxis, :, :]
        
        # Convert to PyTorch tensor and move to appropriate device (CPU/GPU)
        X_seq = torch.from_numpy(seq_np).float().to(self.device)
        return X_seq

    # ------------------------------------------------------------------
    #  Gymnasium API: reset / step
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset environment to initial state and generate new weather/hazards.
        
        This method is called at the start of each episode to:
        1. Reset state variables to initial conditions
        2. Generate new weather series (rain, ET0, Kc) for the season
        3. Generate new hazard events (if enabled)
        4. Initialize CDE history for neural residual models
        5. Return initial observation
        
        Args:
            seed: Optional random seed for reproducibility
                If provided, resets the random number generator
            options: Optional dictionary with reset options (not currently used)
        
        Returns:
            obs: Initial observation [psi_0, S_0, R_0, ET0_0]
            info: Empty info dictionary (Gymnasium API requirement)
        """
        # Reset random number generator if new seed is provided
        # This ensures reproducibility across episodes
        # IMPORTANT: If seed is provided, it changes the RNG, which affects weather generation
        # If seed is None, uses the existing RNG state (different weather each reset)
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        # Reset state variables to initial conditions
        self.day = 0  # Start at day 0
        self.S = self.S_fc  # Initialize soil at field capacity (optimal starting condition)
        self.psi = self._S_to_psi(self.S)  # Calculate corresponding tension
        self.cum_stress = 0.0  # Reset cumulative stress counter
        self.cum_irrig = 0.0  # Reset cumulative irrigation counter
        self.cum_drain = 0.0  # Reset cumulative drainage
        self.events_count = 0  # Reset irrigation events counter

        # Generate new weather series for this episode
        # IMPORTANT: Weather is regenerated on each reset() call
        # - If seed is provided: same seed = same weather (reproducible)
        # - If seed is None: uses current RNG state = different weather each time
        # This differs from utils_env_gymnasium.py which generates weather only at initialization
        self._generate_weather()
        
        # Regenerate hazard events if enabled
        # Hazards are randomly scheduled at the start of each episode
        if self.enable_hazards:
            self._generate_hazard_events()

        # Initialize CDE history for neural residual models
        # CDE models need a sequence of historical features
        self.history_cde = []  # Clear previous episode's history
        
        # Add initial state to history
        # This represents the state before any actions are taken
        R0 = self.rain[0]  # First day's rainfall
        ET0_0 = self.et0[0]  # First day's reference ET
        feats0 = self._build_features_cde(self.psi, 0.0, R0, ET0_0)
        # Initial irrigation is 0 (no action taken yet)
        self.history_cde.append(feats0)

        # Build and return initial observation
        obs = self._build_obs()  # [psi_0, S_0, R_0, ET0_0]
        info = {}  # Empty info dict (Gymnasium API)
        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.
        
        This is the core simulation function that:
        1. Processes the irrigation action
        2. Applies hazard effects (if any)
        3. Simulates soil-plant-water dynamics (physical model)
        4. Applies neural residual corrections (if models are provided)
        5. Calculates reward
        6. Updates state
        7. Returns observation, reward, and info
        
        Args:
            action: Irrigation action (scalar or array)
                - Scalar: Direct irrigation amount (mm)
                - Array: [irrigation_amount] (mm)
                Clipped to [0, I_max]
        
        Returns:
            obs: Next observation [psi_{t+1}, S_{t+1}, R_{t+1}, ET0_{t+1}]
            reward: Reward for this step (float)
            terminated: Whether episode ended naturally (season complete)
            truncated: Whether episode was cut short (always False in this env)
            info: Dictionary with detailed step information
        """
        # Process action: handle different input formats (scalar, list, array)
        # Extract irrigation amount and ensure it's a float
        if isinstance(action, (list, tuple, np.ndarray)):
            I_t = float(action[0])  # Extract first element if array-like
        else:
            I_t = float(action)  # Use directly if scalar
        
        # Clip action to valid range [0, I_max]
        # Prevents invalid negative or excessive irrigation amounts
        I_t = float(np.clip(I_t, 0.0, self.I_max))

        # Get current day and weather data
        t = self.day  # Current day index (0 to T-1)
        R_t_base = self.rain[t]  # Base rainfall (before hazard effects)
        ET0_t_base = self.et0[t]  # Base reference ET (before hazard effects)
        Kc_t = self.Kc[t]  # Crop coefficient for current growth stage

        # Apply hazard effects (modifies R, ET0, and I based on active hazards)
        # Hazards can modify weather conditions and irrigation capacity:
        # - Drought: reduces rain, increases ET0 (more water demand, less supply)
        # - Flood: increases rain (excessive water input)
        # - Heatwave: increases ET0 (higher water demand)
        # - Equipment failure: sets irrigation to 0 (can't irrigate)
        # - Water restrictions: limits irrigation capacity (conservation measures)
        R_t, ET0_t, I_t, active_hazards = self._apply_hazards(t, R_t_base, ET0_t_base, I_t)

        # Get current state before applying action
        psi_t = self.psi  # Current soil water tension (cbar)
        S_t = self.S  # Current soil water storage (mm)

        # --- 1) Physical model: ETc and drainage ---
        # Calculate actual crop evapotranspiration (ETc)
        # ETc = Kc * ET0 * f_ET(psi)
        # - Kc_t: Crop coefficient (varies with growth stage)
        # - ET0_t: Reference evapotranspiration (weather-dependent)
        # - f_ET(psi_t): Reduction factor based on soil water tension (water availability)
        f_ET = self._f_ET(psi_t)  # ET reduction factor [0, 1]
        ETc_t = Kc_t * ET0_t * f_ET  # Actual crop evapotranspiration (mm/day)
        
        # Calculate drainage (water loss due to gravity when soil exceeds field capacity)
        D_t = self._drainage(S_t)  # Drainage amount (mm/day)

        # Water balance equation: S_{t+1} = S_t + inputs - outputs
        # S_next_phys = S_t + eta_I * I_t + R_t - ETc_t - D_t
        # - S_t: Current soil water storage (mm)
        # - eta_I * I_t: Effective irrigation (irrigation efficiency applied)
        # - R_t: Rainfall (mm)
        # - ETc_t: Crop evapotranspiration (water loss to atmosphere) (mm)
        # - D_t: Drainage (water loss to deep percolation) (mm)
        S_next_phys = S_t + self.eta_I * I_t + R_t - ETc_t - D_t
        S_next_phys = float(np.clip(S_next_phys, 0.0, self.S_max))  # Ensure within bounds
        
        # Convert updated soil storage to soil water tension
        psi_next_phys = self._S_to_psi(S_next_phys)  # Physical prediction of tension (cbar)

        # --- 2) Neural residual (ODE or CDE) ---
        # Optional learned correction to physical model prediction
        # These models learn patterns that the physical model doesn't capture perfectly
        delta_psi = 0.0  # Default: no correction (pure physics)

        # CDE (Controlled Differential Equation) model takes priority if specified
        # CDE uses sequence of historical states to predict residual
        # More powerful than ODE but requires maintaining history
        if self.residual_cde is not None:
            # Build current feature vector and add to history
            feats_t = self._build_features_cde(psi_t, I_t, R_t, ET0_t)
            self.history_cde.append(feats_t)
            
            # Build sequence tensor from history: [1, seq_len_cde, 4]
            # Contains last seq_len_cde time steps of [psi, I, R, ET0]
            X_seq = self._build_cde_sequence_tensor()
            
            # Predict residual correction (no gradient computation needed during inference)
            with torch.no_grad():
                delta_psi_tensor = self.residual_cde(X_seq)
            # Extract scalar value from tensor output
            delta_psi = float(delta_psi_tensor.cpu().numpy()[0, 0])

        elif self.residual_ode is not None:
            # One-step residual ODE: Δψ = f(ψ_t, I_t, R_t, ET0_t)
            # Simpler than CDE: only uses current state, no history needed
            # Build input feature vector: [psi, I, R, ET0] as [1, 4] array
            x = np.array([[psi_t, I_t, R_t, ET0_t]], dtype=np.float32)
            x_torch = torch.from_numpy(x).to(self.device)  # Move to CPU/GPU
            
            # Predict residual correction (no gradient computation during inference)
            with torch.no_grad():
                dpsi_tensor = self.residual_ode(x_torch)
            # Extract scalar value from tensor output
            delta_psi = float(dpsi_tensor.cpu().numpy()[0, 0])

        # Correction of ψ_{t+1} with neural residual
        # Add learned correction to physical model prediction
        # delta_psi can be positive or negative, adjusting the physical prediction
        # This allows the model to learn systematic biases or missing physics
        psi_next = psi_next_phys + delta_psi
        # Clip to valid range to ensure physically realistic values
        psi_next = float(np.clip(psi_next, self.psi_sat, self.psi_wp))

        # Projection to S_{t+1}
        # Convert corrected tension back to soil storage
        # This ensures consistency between S and psi representations
        S_next = self._psi_to_S(psi_next)

        # --- 3) Reward calculation ---
        # Calculate daily reward based on current state and actions
        # Reward = -alpha * stress_t - beta * I_t - gamma * D_t
        # - Negative reward (penalty) for stress, irrigation cost, and drainage loss
        # - Agent should minimize these to maximize reward
        
        stress_t = self._stress(psi_t)  # Current plant water stress [0, inf)
        self.cum_stress += stress_t  # Accumulate stress for yield calculation
        self.cum_irrig += I_t  # Accumulate irrigation for terminal penalty
        self.cum_drain += D_t  # Accumulate drainage
        if I_t > 0.0:
            self.events_count += 1

        # Daily reward: penalize stress, irrigation, and drainage
        # - alpha * stress_t: Penalty for plant stress (higher stress = lower reward)
        # - beta * I_t: Penalty for water usage (conservation incentive)
        # - gamma * D_t: Penalty for drainage loss (efficiency incentive)
        reward = - self.alpha * stress_t - self.beta * I_t - self.gamma * D_t

        # Update state
        self.S = S_next  # Update soil water storage
        self.psi = psi_next  # Update soil water tension
        self.day += 1  # Advance to next day

        # Check if season is complete
        done = self.day >= self.T
        
        if done:
            # Terminal reward: final yield minus water usage penalty
            # Y = Y_max * exp(-k_yield * cum_stress)
            # Terminal reward = Y - lambda_water * cum_irrig
            # - Y: Final crop yield (reward for good management)
            # - lambda_water * cum_irrig: Penalty for total water usage (conservation)
            Y = self._yield(self.cum_stress)  # Final yield based on cumulative stress
            terminal_penalty = self.lambda_water * self.cum_irrig  # Total water usage penalty
            terminal_reward = Y - terminal_penalty  # Net terminal reward
            reward += terminal_reward  # Add terminal reward to daily reward
        else:
            Y = None  # No yield yet (season not finished)
            terminal_reward = 0.0  # No terminal reward during season

        # Build observation for next step
        # Contains updated state and next day's weather forecast
        obs = self._build_obs()  # [psi_{t+1}, S_{t+1}, R_{t+1}, ET0_{t+1}]
        
        # Build info dictionary with detailed step information
        # This is useful for analysis, debugging, and visualization
        info = {
            "day": self.day,  # Current day (after increment, so this is the day just completed)
            "S": self.S,  # Updated soil water storage (mm)
            "psi": self.psi,  # Updated soil water tension (cbar)
            "cum_stress": self.cum_stress,  # Cumulative plant stress so far
            "cum_irrig": self.cum_irrig,  # Total irrigation applied so far
            "ETc": ETc_t,  # Actual crop evapotranspiration this step (mm)
            "D": D_t,  # Drainage this step (mm)
            "delta_psi": delta_psi,  # Neural residual correction applied (if any)
            "psi_next_phys": psi_next_phys,  # Physical model prediction before correction
            "active_hazards": active_hazards,  # List of hazards active this day
            "hazard_count": len(active_hazards),  # Number of active hazards
        }
        
        # Add terminal information if episode is complete
        if done:
            info["yield"] = Y  # Final crop yield (normalized 0-1)
            info["terminal_reward"] = terminal_reward  # Terminal reward component
            info["hazard_history"] = self.hazard_history.copy()  # Full hazard event log

        # Episode termination flags (Gymnasium API)
        terminated = done  # True when season completes naturally
        truncated = False  # Always False (episodes always run to completion)
        
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    #  Render (optional)
    # ------------------------------------------------------------------

    def render(self):
        """Print current state information for visualization/debugging."""
        print(
            f"Day {self.day}/{self.T} | "
            f"S = {self.S:.1f} mm, ψ = {self.psi:.1f} cbar | "
            f"cum_stress = {self.cum_stress:.2f}, cum_I = {self.cum_irrig:.1f} mm"
        )
