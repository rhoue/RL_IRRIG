"""
Utilitaires pour le Scénario 3 : Neural ODE.

Ce module contient les classes et fonctions spécifiques au scénario 3 :
- ResidualODEModel : Modèle Neural ODE résiduel
- ResidualODEDataset : Dataset pour pré-entraînement
- pretrain_residual_ode : Fonction de pré-entraînement
- train_ppo_hybrid_ode : Fonction d'entraînement PPO sur modèle hybride
"""

import numpy as np
from typing import Dict, Optional, Any
import sys
import os

# Imports PyTorch
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

# Imports stable-baselines3
try:
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.monitor import Monitor  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    PPO = None
    Monitor = None
    DummyVecEnv = None
    BaseCallback = None

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
# CLASSES
# ============================================================================

class ResidualODEModel(nn.Module):
    """
    Modèle Neural ODE résiduel pour corriger la prédiction physique.
    
    Principe : Apprend une correction Δψ à partir de l'état actuel
    Input : [psi_t, I_t, R_t, ET0_t]
    Output : Δψ (correction à ajouter à la prédiction physique)
    """
    def __init__(self, in_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forme [batch_size, 4] contenant [psi, I, R, ET0]
        Returns:
            Tensor de forme [batch_size, 1] contenant Δψ
        """
        return self.net(x)


class ResidualODEDataset(Dataset):
    """
    Dataset pour pré-entraîner le modèle Neural ODE résiduel.
    
    Génère des trajectoires simulées et calcule les résidus entre
    la prédiction physique et la "vérité terrain" (avec bruit).
    """
    def __init__(self, T: int, N_traj: int, soil, max_irrigation: float, seed: int = 123):
        self.X, self.Y = [], []
        rng = np.random.default_rng(seed)
        
        for n in range(N_traj):
            # Générer météo pour cette trajectoire
            rain, et0, Kc = generate_weather(T=T, seed=seed + 100 + n)
            S = float(soil.S_fc)  # État initial
            
            for t in range(T - 1):
                psi = soil.S_to_psi(S)
                # Irrigation aléatoire pour diversité
                I = float(np.clip(
                    rng.normal(loc=max_irrigation / 2, scale=max_irrigation / 4),
                    0.0, max_irrigation
                ))
                
                # Calcul physique
                fET = soil.f_ET(psi)
                ETc = Kc[t] * et0[t] * fET
                D = soil.drainage(S)
                
                # "Vérité terrain" avec bruit (simule des observations réelles)
                delta_true = soil.eta_I * I + rain[t] - ETc - D + rng.normal(0.0, 0.2)
                S_true = np.clip(S + delta_true, 0.0, soil.S_max)
                
                # Prédiction physique pure
                delta_phys = soil.eta_I * I + rain[t] - ETc - D
                S_phys = np.clip(S + delta_phys, 0.0, soil.S_max)
                
                # Résidu à apprendre : différence entre vérité et physique
                psi_true = soil.S_to_psi(S_true)
                psi_phys = soil.S_to_psi(S_phys)
                delta_psi = psi_true - psi_phys
                
                # Features : [psi_t, I_t, R_t, ET0_t]
                self.X.append([psi, I, rain[t], et0[t]])
                self.Y.append([delta_psi])
                
                S = float(S_true)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================================
# FONCTIONS
# ============================================================================

def pretrain_residual_ode(
    soil,
    max_irrigation: Optional[float] = None,
    T: Optional[int] = None,
    N_traj: int = 32,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress_callback=None
) -> ResidualODEModel:
    """
    Pré-entraîne le modèle Neural ODE résiduel.
    
    Args:
        soil: Instance de PhysicalBucket
        max_irrigation: Dose maximale d'irrigation (défaut depuis utils_physics_config)
        T: Longueur de saison (défaut depuis utils_physics_config)
        N_traj: Nombre de trajectoires à générer
        n_epochs: Nombre d'epochs d'entraînement
        batch_size: Taille des batches
        lr: Taux d'apprentissage
        seed: Graine aléatoire (défaut depuis utils_physics_config)
        device: Device PyTorch ('cpu' ou 'cuda')
        progress_callback: Fonction appelée à chaque epoch pour le suivi
    
    Returns:
        ResidualODEModel entraîné
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas installé. Installez-le avec: pip install torch")
    
    # Importer la configuration par défaut
    from src.utils_physics_config import get_default_physics_config
    
    # Obtenir les paramètres par défaut et surcharger avec les valeurs fournies
    default_config = get_default_physics_config(
        max_irrigation=max_irrigation,
        season_length=T,
        seed=seed
    )
    max_irrigation = default_config["max_irrigation"]
    T = default_config["season_length"]
    seed = default_config["seed"]
    
    model = ResidualODEModel(in_dim=4, hidden=64).to(device)
    dataset = ResidualODEDataset(T, N_traj, soil, max_irrigation, seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    
    model.train()
    for epoch in range(1, n_epochs + 1):
        losses = []
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        avg_loss = float(np.mean(losses))
        if progress_callback:
            progress_callback(epoch, n_epochs, avg_loss)
    
    model.eval()
    final_loss = avg_loss  # Garder la dernière loss moyenne
    return model, final_loss


def train_ppo_hybrid_ode(
    residual_ode_model: ResidualODEModel,
    season_length: Optional[int] = None,
    max_irrigation: Optional[float] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    soil_params: Optional[Dict] = None,
    weather_params: Optional[Dict] = None,
    weather_shift_cfg: Optional[Dict[str, Any]] = None,
    ppo_kwargs: Optional[Dict] = None,
    progress_callback=None,
    lexico_config: Optional["LexicoConfig"] = None,
    goal_spec: Optional[Dict[str, Any]] = None,
    teacher_model=None,
    distill_coef: float = 0.0,
    external_weather: Optional[Dict[str, Any]] = None,
):
    """
    Entraîne un agent PPO sur l'environnement hybride (physique + Neural ODE).
    
    Args:
        residual_ode_model: Modèle Neural ODE pré-entraîné
        season_length: Longueur de la saison (défaut depuis utils_physics_config)
        max_irrigation: Dose maximale d'irrigation (défaut depuis utils_physics_config)
        total_timesteps: Nombre total de pas d'entraînement (défaut depuis utils_physics_config)
        seed: Graine aléatoire (défaut depuis utils_physics_config)
        soil_params: Paramètres du sol
        weather_params: Paramètres météo
        ppo_kwargs: Hyperparamètres PPO
        progress_callback: Callback pour le suivi de l'entraînement
    
    Returns:
        Tuple (Modèle PPO entraîné, dictionnaire des métriques d'entraînement)
    """
    if not PPO_AVAILABLE:
        raise ImportError("stable-baselines3 n'est pas installé")

    from src.utils_physics_config import get_default_config

    # Obtenir les paramètres par défaut et surcharger avec les valeurs fournies
    default_config = get_default_config(
        season_length=season_length,
        max_irrigation=max_irrigation,
        total_timesteps=total_timesteps,
        seed=seed
    )
    season_length = default_config["season_length"]
    max_irrigation = default_config["max_irrigation"]
    total_timesteps = default_config["total_timesteps"]
    seed = default_config["seed"]
    
    # Import de l'environnement (gestion des chemins relatifs/absolus)
    try:
        from src.utils_env_modeles import IrrigationEnvPhysical
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils_env_modeles import IrrigationEnvPhysical
    
    # Créer l'environnement avec le modèle résiduel
    def make_env():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,  # Le paramètre s'appelle 'soil', pas 'soil_params'
            weather_params=weather_params,
            residual_ode=residual_ode_model,
            device="cpu",
            goal_spec=goal_spec,
            weather_shift_cfg=weather_shift_cfg,
            external_weather=external_weather,
        )
        try:
            from src.utils_lexico_goal import wrap_with_lexico
            env = wrap_with_lexico(env, lexico_config)
        except Exception:
            pass
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    
    # Importer les utilitaires PPO
    from src.utils_ppo_training import create_ppo_callbacks, get_default_ppo_config
    
    # Configuration PPO : utiliser les valeurs par défaut et surcharger avec les valeurs spécifiques
    # (ent_coef=0.0 et verbose=1 pour les modèles hybrides) puis avec les paramètres personnalisés
    ppo_overrides = dict(ppo_kwargs or {})
    ppo_overrides.setdefault("ent_coef", 0.0)  # Spécifique aux modèles hybrides (Neural ODE/CDE)
    ppo_overrides.setdefault("verbose", 1)     # Feedback détaillé pour les scénarios complexes
    ppo_config = get_default_ppo_config(**ppo_overrides)
    
    # Extraire 'policy' du config si présent (car il est passé comme argument positionnel)
    policy = ppo_config.pop("policy", "MlpPolicy")
    
    # Créer les callbacks (progression + collecte de métriques)
    callbacks, metrics_callback = create_ppo_callbacks(
        progress_callback=progress_callback,
        total_timesteps=total_timesteps
    )
    if teacher_model is not None and distill_coef > 0:
        try:
            from src.utils_distillation import DistillationRewardCallback
            callbacks = callbacks or []
            callbacks.append(DistillationRewardCallback(teacher_model, coef=distill_coef))
        except Exception:
            pass
    
    # Entraîner le modèle
    model = PPO(policy, env, **ppo_config)
    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)
    
    # Récupérer les métriques finales
    training_metrics = metrics_callback.get_final_metrics()
    
    return model, training_metrics
