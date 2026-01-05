"""
Utilitaires pour le Scénario 4 : Neural CDE.

Ce module contient les classes et fonctions spécifiques au scénario 4 :
- NeuralCDEPsiDiscrete : Modèle Neural CDE discretisé
- CDEPsiDiscreteDataset : Dataset pour pré-entraînement
- pretrain_residual_cde : Fonction de pré-entraînement
- train_ppo_hybrid_cde : Fonction d'entraînement PPO sur modèle hybride
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

class NeuralCDEPsiDiscrete(nn.Module):
    """
    Modèle Neural CDE discretisé pour corriger la tension ψ.
    
    Principe : Utilise une séquence d'états passés pour prédire la correction
    Schéma d'Euler : Z_{k+1} = Z_k + f_θ(Z_k, X_k) · ΔX_k
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, hidden_mlp: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Z_0 = φ_θ(X_0)
        self.init_net = nn.Sequential(
            nn.Linear(input_dim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, hidden_dim),
        )
        
        # f_θ(Z_k, X_k) -> [batch, hidden_dim, input_dim]
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_mlp),
            nn.Tanh(),
            nn.Linear(hidden_mlp, hidden_dim * input_dim),
        )
        
        # Lecture finale : Z_T -> Δψ
        self.readout = nn.Linear(hidden_dim, 1)
    
    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_seq: Tensor de forme [batch, seq_len, input_dim] avec X_k = [ψ_k, I_k, R_k, ET0_k]
        Returns:
            Tensor de forme [batch, 1] contenant Δψ
        """
        batch_size, seq_len, d = X_seq.size()
        assert d == self.input_dim
        
        # Initialisation Z_0 à partir de X_0
        X0 = X_seq[:, 0, :]  # [batch, input_dim]
        Z = self.init_net(X0)  # [batch, hidden_dim]
        
        # Intégration discrète sur la fenêtre
        for k in range(seq_len - 1):
            Xk = X_seq[:, k, :]
            Xk1 = X_seq[:, k + 1, :]
            dX = Xk1 - Xk  # [batch, input_dim]
            
            ZX = torch.cat([Z, Xk], dim=-1)  # [batch, hidden_dim + input_dim]
            f_val = self.f_net(ZX)  # [batch, hidden_dim * input_dim]
            f_val = f_val.view(batch_size, self.hidden_dim, self.input_dim)
            
            # Contraction : f_θ(Z_k, X_k) · ΔX_k -> [batch, hidden_dim]
            dZ = torch.einsum("bhi,bi->bh", f_val, dX)
            Z = Z + dZ
        
        # Correction sur la dernière étape
        delta_psi = self.readout(Z)  # [batch, 1]
        return delta_psi


class CDEPsiDiscreteDataset(Dataset):
    """
    Dataset pour pré-entraîner le modèle Neural CDE discretisé.
    
    Génère des séquences de trajectoires simulées et calcule les résidus.
    """
    def __init__(self, T: int, N_traj: int, seq_len: int, soil, max_irrigation: float, seed: int = 123):
        self.X_seq_list, self.Y_list = [], []
        rng = np.random.default_rng(seed)
        
        for n in range(N_traj):
            # Générer météo pour cette trajectoire
            rain, et0, Kc = generate_weather(T=T, seed=seed + 200 + n)
            S = float(soil.S_fc)  # État initial
            
            # Stocker l'historique pour créer les séquences
            history = []
            
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
                
                # "Vérité terrain" avec bruit
                delta_true = soil.eta_I * I + rain[t] - ETc - D + rng.normal(0.0, 0.2)
                S_true = np.clip(S + delta_true, 0.0, soil.S_max)
                
                # Prédiction physique pure
                delta_phys = soil.eta_I * I + rain[t] - ETc - D
                S_phys = np.clip(S + delta_phys, 0.0, soil.S_max)
                
                # Résidu à apprendre
                psi_true = soil.S_to_psi(S_true)
                psi_phys = soil.S_to_psi(S_phys)
                delta_psi = psi_true - psi_phys
                
                # Features : [psi_t, I_t, R_t, ET0_t]
                feats = [psi, I, rain[t], et0[t]]
                history.append(feats)
                
                # Créer une séquence si on a assez d'historique
                if len(history) >= seq_len:
                    # Prendre les seq_len derniers éléments
                    seq = history[-seq_len:]
                    self.X_seq_list.append(seq)
                    self.Y_list.append([delta_psi])
                
                S = float(S_true)
        
        # Convertir en tensors
        self.X_seq = torch.tensor(self.X_seq_list, dtype=torch.float32)  # [N, seq_len, 4]
        self.Y = torch.tensor(self.Y_list, dtype=torch.float32)  # [N, 1]
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y[idx]


# ============================================================================
# FONCTIONS
# ============================================================================

def pretrain_residual_cde(
    soil,
    max_irrigation: Optional[float] = None,
    T: Optional[int] = None,
    N_traj: int = 32,
    seq_len: int = 5,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress_callback=None
) -> NeuralCDEPsiDiscrete:
    """
    Pré-entraîne le modèle Neural CDE résiduel.
    
    Args:
        soil: Instance de PhysicalBucket
        max_irrigation: Dose maximale d'irrigation (défaut depuis utils_physics_config)
        T: Longueur de saison (défaut depuis utils_physics_config)
        N_traj: Nombre de trajectoires à générer
        seq_len: Longueur de la séquence pour le CDE
        n_epochs: Nombre d'epochs d'entraînement
        batch_size: Taille des batches
        lr: Taux d'apprentissage
        seed: Graine aléatoire (défaut depuis utils_physics_config)
        device: Device PyTorch ('cpu' ou 'cuda')
        progress_callback: Fonction appelée à chaque epoch pour le suivi
    
    Returns:
        NeuralCDEPsiDiscrete entraîné
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
    
    model = NeuralCDEPsiDiscrete(input_dim=4, hidden_dim=64, hidden_mlp=64).to(device)
    dataset = CDEPsiDiscreteDataset(T, N_traj, seq_len, soil, max_irrigation, seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
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


def train_ppo_hybrid_cde(
    residual_cde_model: NeuralCDEPsiDiscrete,
    season_length: Optional[int] = None,
    max_irrigation: Optional[float] = None,
    seq_len_cde: int = 5,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    soil_params: Optional[Dict] = None,
    weather_params: Optional[Dict] = None,
    ppo_kwargs: Optional[Dict] = None,
    progress_callback=None,
    lexico_config=None,
    goal_spec: Optional[Dict[str, Any]] = None,
    teacher_model=None,
    distill_coef: float = 0.0,
    weather_shift_cfg: Optional[Dict[str, Any]] = None,
):
    """
    Entraîne un agent PPO sur l'environnement hybride (physique + Neural CDE).
    
    Args:
        residual_cde_model: Modèle Neural CDE pré-entraîné
        season_length: Longueur de la saison (défaut depuis utils_physics_config)
        max_irrigation: Dose maximale d'irrigation (défaut depuis utils_physics_config)
        seq_len_cde: Longueur de séquence pour le CDE
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
    
    # Importer la configuration par défaut
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
    
    # Import de l'environnement
    try:
        from src.utils_env_modeles import IrrigationEnvPhysical
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils_env_modeles import IrrigationEnvPhysical
    
    # Créer l'environnement avec le modèle résiduel CDE
    def make_env():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,
            weather_params=weather_params,
            residual_cde=residual_cde_model,
            seq_len_cde=seq_len_cde,
            device="cpu",
            goal_spec=goal_spec,
            weather_shift_cfg=weather_shift_cfg,
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
    # Note: ent_coef=0.0 car l'exploration est déjà gérée par le modèle résiduel Neural CDE
    # qui apprend à corriger les prédictions physiques, donc pas besoin d'encourager l'exploration
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
