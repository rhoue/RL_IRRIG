"""
Utilitaires pour le Scénario 6 (World Model).

Ce module contient les classes et fonctions pour les trois phases du World Model :
- Phase 1 : World Model simple (encodeur PatchTST + transition Neural ODE)
- Phase 2 : World Model complet (encodeur PatchTST + transition Neural CDE + décodeur)
- Phase 3 : Intégration avec modèle physique (hybridation physics-informed)

Classes et fonctions :
- Phase 1 :
  * LatentTransitionODE : Modèle de transition Neural ODE dans l'espace latent
  * WorldModelDataset : Dataset pour entraîner le World Model Phase 1
  * train_world_model_phase1 : Fonction d'entraînement du World Model Phase 1
  * imagination_rollout : Fonction pour effectuer des rollouts d'imagination
  * train_ppo_with_world_model_phase1 : Fonction d'entraînement PPO avec World Model Phase 1

- Phase 2 :
  * LatentDecoder : Décodeur pour reconstruire les observables depuis l'espace latent
  * LatentTransitionCDE : Modèle de transition Neural CDE dans l'espace latent
  * WorldModelPhase2Dataset : Dataset pour entraîner le World Model Phase 2
  * train_world_model_phase2 : Fonction d'entraînement du World Model Phase 2
  * train_ppo_with_world_model_phase2 : Fonction d'entraînement PPO avec World Model Phase 2

- Phase 3 :
  * PhysicsInformedWorldModelWrapper : Wrapper hybridant modèle physique + world model
  * train_ppo_with_world_model_phase3 : Fonction d'entraînement PPO phase 3
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
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

# Imports Gymnasium
try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    from gymnasium import Wrapper  # type: ignore
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    gym = None
    spaces = None
    Wrapper = None

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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils_weather import generate_weather

# Import de PatchTSTFeatureExtractor depuis utils_patch_tst (pour les types)
try:
    from src.utils_patch_tst import PatchTSTFeatureExtractor, PatchTSTEnvWrapper
except ImportError:
    PatchTSTFeatureExtractor = None
    PatchTSTEnvWrapper = None


# ============================================================================
# SCÉNARIO 6 : WORLD MODEL - PHASE 1 (CLASSES ET FONCTIONS)
# ============================================================================

class LatentTransitionODE(nn.Module):
    """
    Modèle de transition Neural ODE dans l'espace latent pour le World Model.
    
    Principe : Prédit la transition dans l'espace latent z_{t+1} = F_θ(z_t, a_t, inputs_exogènes)
    où z_t est la représentation latente encodée par PatchTST.
    
    Input : [z_t, a_t, R_t, ET0_t, Kc_t] (état latent + action + inputs exogènes)
    Output : Δz_t (variation de l'état latent)
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        """
        Args:
            latent_dim: Dimension de l'espace latent (feature_dim de PatchTST)
            hidden_dim: Dimension des couches cachées
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible")
        super().__init__()
        # Input: [z_t (latent_dim), a_t (1), R_t (1), ET0_t (1), Kc_t (1)]
        input_dim = latent_dim + 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # Output: Δz_t
        )
    
    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor, inputs_exog: torch.Tensor) -> torch.Tensor:
        """
        Prédit la variation de l'état latent.
        
        Args:
            z_t: État latent actuel [batch_size, latent_dim]
            a_t: Action (irrigation) [batch_size, 1]
            inputs_exog: Inputs exogènes [batch_size, 3] = [R_t, ET0_t, Kc_t]
        
        Returns:
            Δz_t: Variation de l'état latent [batch_size, latent_dim]
        """
        # Concaténer les inputs
        x = torch.cat([z_t, a_t, inputs_exog], dim=1)  # [batch_size, latent_dim + 4]
        return self.net(x)


class WorldModelDataset(Dataset):
    """
    Dataset pour entraîner le World Model (Phase 1).
    
    Génère des paires (z_t, a_t, inputs_exog, z_{t+1}) où :
    - z_t est encodé par PatchTST à partir de l'historique
    - z_{t+1} est encodé par PatchTST à partir de l'historique suivant
    """
    def __init__(
        self,
        encoder_model: PatchTSTFeatureExtractor,
        T: int,
        N_traj: int,
        seq_len: int,
        soil,
        max_irrigation: float,
        seed: int = 123,
        device: str = "cpu"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible")
        self.encoder_model = encoder_model
        self.seq_len = seq_len
        self.device = device
        self.z_t_list = []
        self.a_t_list = []
        self.inputs_exog_list = []
        self.z_tp1_list = []
        
        rng = np.random.default_rng(seed)
        encoder_model.eval()
        
        with torch.no_grad():
            for n in range(N_traj):
                rain, et0, Kc = generate_weather(T=T, seed=seed + 500 + n)
                S = float(soil.S_fc)
                
                history = []  # Historique pour PatchTST
                
                for t in range(T - 1):
                    psi = soil.S_to_psi(S)
                    I = float(np.clip(
                        rng.normal(loc=max_irrigation / 2, scale=max_irrigation / 4),
                        0.0, max_irrigation
                    ))
                    
                    fET = soil.f_ET(psi)
                    ETc = Kc[t] * et0[t] * fET
                    D = soil.drainage(S)
                    
                    delta = soil.eta_I * I + rain[t] - ETc - D
                    S_next = np.clip(S + delta, 0.0, soil.S_max)
                    psi_next = soil.S_to_psi(S_next)
                    
                    # Observation actuelle
                    X_t = np.array([psi, I, rain[t], et0[t]], dtype=np.float32)
                    history.append(X_t)
                    
                    # Observation suivante
                    X_tp1 = np.array([psi_next, I, rain[t+1] if t+1 < T-1 else rain[t], et0[t+1] if t+1 < T-1 else et0[t]], dtype=np.float32)
                    
                    # Encoder z_t si on a assez d'historique
                    if len(history) >= seq_len:
                        seq_t = np.array(history[-seq_len:], dtype=np.float32)
                        seq_t_tensor = torch.from_numpy(seq_t).unsqueeze(0).to(device)
                        z_t = encoder_model(seq_t_tensor).cpu().numpy().squeeze(0)
                        
                        # Pour z_{t+1}, on ajoute X_tp1 à l'historique
                        history_tp1 = history + [X_tp1]
                        if len(history_tp1) >= seq_len:
                            seq_tp1 = np.array(history_tp1[-seq_len:], dtype=np.float32)
                            seq_tp1_tensor = torch.from_numpy(seq_tp1).unsqueeze(0).to(device)
                            z_tp1 = encoder_model(seq_tp1_tensor).cpu().numpy().squeeze(0)
                            
                            # Stocker les données
                            self.z_t_list.append(z_t)
                            self.a_t_list.append(I)
                            self.inputs_exog_list.append(np.array([rain[t], et0[t], Kc[t]], dtype=np.float32))
                            self.z_tp1_list.append(z_tp1)
                    
                    S = S_next
        
        # Convertir en tensors
        self.z_t_list = [torch.tensor(z, dtype=torch.float32) for z in self.z_t_list]
        self.a_t_list = [torch.tensor(a, dtype=torch.float32) for a in self.a_t_list]
        self.inputs_exog_list = [torch.tensor(inp, dtype=torch.float32) for inp in self.inputs_exog_list]
        self.z_tp1_list = [torch.tensor(z, dtype=torch.float32) for z in self.z_tp1_list]
    
    def __len__(self):
        return len(self.z_t_list)
    
    def __getitem__(self, idx):
        return (
            self.z_t_list[idx],
            self.a_t_list[idx],
            self.inputs_exog_list[idx],
            self.z_tp1_list[idx]
        )


def train_world_model_phase1(
    encoder_model: PatchTSTFeatureExtractor,
    soil,
    max_irrigation: float = 20.0,
    T: int = 120,
    N_traj: int = 32,
    seq_len: int = 30,
    latent_dim: int = 16,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 123,
    device: str = "cpu",
    progress_callback=None
) -> Tuple[LatentTransitionODE, float]:
    """
    Entraîne le World Model Phase 1 (encodeur PatchTST + transition Neural ODE).
    
    Args:
        encoder_model: Modèle PatchTST pré-entraîné (encodeur)
        soil: Objet PhysicalBucket
        max_irrigation: Dose maximale d'irrigation
        T: Longueur de la saison
        N_traj: Nombre de trajectoires
        seq_len: Longueur de séquence pour PatchTST
        latent_dim: Dimension de l'espace latent
        n_epochs: Nombre d'epochs
        batch_size: Taille des batches
        lr: Taux d'apprentissage
        seed: Graine aléatoire
        device: Device PyTorch
        progress_callback: Callback pour la progression
    
    Returns:
        Tuple (transition_model, final_loss)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas disponible")
    
    device_torch = torch.device(device)
    
    # Créer le modèle de transition
    transition_model = LatentTransitionODE(
        latent_dim=latent_dim,
        hidden_dim=128
    ).to(device_torch)
    
    # Créer le dataset
    dataset = WorldModelDataset(
        encoder_model=encoder_model,
        T=T,
        N_traj=N_traj,
        seq_len=seq_len,
        soil=soil,
        max_irrigation=max_irrigation,
        seed=seed,
        device=device
    )
    
    if len(dataset) == 0:
        raise ValueError("Le dataset est vide. Vérifiez les paramètres (T, seq_len, N_traj).")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(transition_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    transition_model.train()
    encoder_model.eval()  # L'encodeur est figé
    
    final_loss = 0.0
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (z_t, a_t, inputs_exog, z_tp1) in enumerate(dataloader):
            z_t = z_t.to(device_torch)
            a_t = a_t.unsqueeze(1).to(device_torch)  # [batch_size, 1]
            inputs_exog = inputs_exog.to(device_torch)
            z_tp1 = z_tp1.to(device_torch)
            
            # Prédire la transition
            delta_z = transition_model(z_t, a_t, inputs_exog)  # [batch_size, latent_dim]
            z_tp1_pred = z_t + delta_z  # [batch_size, latent_dim]
            
            # Calculer la loss
            loss = criterion(z_tp1_pred, z_tp1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        final_loss = avg_loss
        
        if progress_callback:
            progress = min(1.0, max(0.0, (epoch + 1) / n_epochs))
            progress_callback(progress, f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    transition_model.eval()
    return transition_model, final_loss


def imagination_rollout(
    encoder_model: PatchTSTFeatureExtractor,
    transition_model: LatentTransitionODE,
    z_t: torch.Tensor,
    policy,
    horizon: int = 5,
    env_real=None,
    device: str = "cpu"
) -> list:
    """
    Effectue un rollout d'imagination dans l'espace latent.
    
    Args:
        encoder_model: Encodeur PatchTST (figé)
        transition_model: Modèle de transition dans l'espace latent
        z_t: État latent initial [latent_dim]
        policy: Fonction qui prend une observation et retourne une action
        horizon: Nombre de pas de temps à simuler
        env_real: Environnement réel (optionnel, pour obtenir les inputs exogènes)
        device: Device PyTorch
    
    Returns:
        Liste de tuples (z_k, a_k, reward_k) pour k dans [t, t+horizon]
    """
    encoder_model.eval()
    transition_model.eval()
    
    rollout = []
    z_k = z_t.clone().to(device)
    
    with torch.no_grad():
        for k in range(horizon):
            # Décoder z_k en observation (approximation : utiliser z_k directement comme observation enrichie)
            # Pour Phase 1, on travaille directement dans l'espace latent
            obs_latent = z_k.cpu().numpy()
            
            # Obtenir l'action de la politique
            # Note: La politique doit être adaptée pour accepter des observations dans l'espace latent
            # Pour l'instant, on utilise une action aléatoire ou une heuristique
            if env_real is not None:
                # Utiliser l'environnement réel pour obtenir les inputs exogènes
                # (approximation : utiliser les valeurs actuelles)
                R_k = env_real.rain[env_real.day] if hasattr(env_real, 'rain') else 0.0
                ET0_k = env_real.et0[env_real.day] if hasattr(env_real, 'et0') else 5.0
                Kc_k = env_real.Kc[env_real.day] if hasattr(env_real, 'Kc') else 1.0
            else:
                # Valeurs par défaut
                R_k = 0.0
                ET0_k = 5.0
                Kc_k = 1.0
            
            # Action (sera remplacée par la politique réelle)
            a_k = policy(obs_latent) if callable(policy) else np.array([10.0])  # Action par défaut
            a_k_tensor = torch.tensor([a_k[0] if isinstance(a_k, np.ndarray) else a_k], dtype=torch.float32).to(device).unsqueeze(0)
            
            # Inputs exogènes
            inputs_exog = torch.tensor([[R_k, ET0_k, Kc_k]], dtype=torch.float32).to(device)
            
            # Transition dans l'espace latent
            z_k = z_k.unsqueeze(0)  # [1, latent_dim]
            delta_z = transition_model(z_k, a_k_tensor, inputs_exog)
            z_kp1 = z_k + delta_z
            z_k = z_kp1.squeeze(0)  # [latent_dim]
            
            # Reward (approximation : sera calculé par l'environnement réel)
            reward_k = 0.0  # Placeholder
            
            rollout.append((z_k.clone(), a_k, reward_k))
    
    return rollout


def train_ppo_with_world_model_phase1(
    encoder_model: PatchTSTFeatureExtractor,
    transition_model: LatentTransitionODE,
    soil_params: Optional[Dict[str, float]] = None,
    weather_params: Optional[Dict[str, float]] = None,
    season_length: Optional[int] = None,
    max_irrigation: Optional[float] = None,
    total_timesteps: Optional[int] = None,
    imagination_horizon: int = 5,
    imagination_ratio: float = 0.5,  # Ratio de rollouts d'imagination vs réels
    seed: Optional[int] = None,
    device: str = "cpu",
    progress_callback=None,
    **ppo_kwargs
):
    """
    Entraîne un agent PPO avec World Model Phase 1 (rollouts d'imagination).
    
    Args:
        encoder_model: Encodeur PatchTST pré-entraîné
        transition_model: Modèle de transition pré-entraîné
        soil_params: Paramètres du sol
        season_length: Longueur de saison (défaut depuis utils_physics_config)
        max_irrigation: Dose maximale d'irrigation (défaut depuis utils_physics_config)
        total_timesteps: Nombre total de pas d'entraînement (défaut depuis utils_physics_config)
        imagination_horizon: Horizon des rollouts d'imagination
        imagination_ratio: Ratio de rollouts d'imagination (0.0 = seulement réel, 1.0 = seulement imagination)
        seed: Graine aléatoire (défaut depuis utils_physics_config)
        device: Device PyTorch
        progress_callback: Callback pour la progression
        **ppo_kwargs: Arguments additionnels pour PPO
    
    Returns:
        Tuple (Modèle PPO entraîné, dictionnaire des métriques d'entraînement)
    """
    if not PPO_AVAILABLE:
        raise ImportError("stable-baselines3 n'est pas disponible")
    
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
    
    from src.utils_env_modeles import IrrigationEnvPhysical
    
    # Créer l'environnement réel
    def make_env():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,
            weather_params=weather_params,
            device=device
        )
        # Envelopper avec PatchTST pour encoder les observations
        if PatchTSTEnvWrapper is not None:
            env = PatchTSTEnvWrapper(
                env,
                patchtst_model=encoder_model,
                seq_len=30,  # Utiliser la même longueur que pour l'entraînement
                device=device
            )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Importer les utilitaires PPO
    from src.utils_ppo_training import create_ppo_callbacks, get_default_ppo_config
    
    # Créer les callbacks (progression + collecte de métriques)
    callbacks, metrics_callback = create_ppo_callbacks(
        progress_callback=progress_callback,
        total_timesteps=total_timesteps
    )
    
    # Configuration PPO par défaut
    ppo_config = get_default_ppo_config(**ppo_kwargs)
    
    # Extraire 'policy' du config si présent (car il est passé comme argument positionnel)
    policy = ppo_config.pop("policy", "MlpPolicy")
    
    # Créer et entraîner le modèle
    model = PPO(policy, env, **ppo_config)
    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)
    
    # Récupérer les métriques finales
    training_metrics = metrics_callback.get_final_metrics()
    
    return model, training_metrics


# ============================================================================
# SCÉNARIO 6 : WORLD MODEL - PHASE 2 (CLASSES ET FONCTIONS)
# ============================================================================

class LatentDecoder(nn.Module):
    """
    Décodeur pour reconstruire les observables depuis l'espace latent.
    
    Principe : Décode z_t (état latent) vers les observables [psi, S, R, ET0]
    Utilisé dans la Phase 2 pour permettre les rollouts d'imagination complets.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128, output_dim: int = 4):
        """
        Args:
            latent_dim: Dimension de l'espace latent
            hidden_dim: Dimension des couches cachées
            output_dim: Dimension de sortie (4 pour [psi, S, R, ET0])
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # Output: [psi, S, R, ET0]
        )
    
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Décode l'état latent vers les observables.
        
        Args:
            z_t: État latent [batch_size, latent_dim]
        
        Returns:
            observables: Observables reconstruits [batch_size, output_dim]
        """
        return self.net(z_t)


class LatentTransitionCDE(nn.Module):
    """
    Modèle de transition Neural CDE dans l'espace latent pour le World Model Phase 2.
    
    Principe : Utilise Neural CDE pour capturer la mémoire temporelle dans les transitions.
    Schéma discret : Z_{k+1} = Z_k + f_θ(Z_k, [a_k, R_k, ET0_k, Kc_k]) · ΔX_k
    où X_k = [a_k, R_k, ET0_k, Kc_k] et ΔX_k est la variation des inputs exogènes.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128, input_dim: int = 4):
        """
        Args:
            latent_dim: Dimension de l'espace latent
            hidden_dim: Dimension de l'état latent du CDE
            input_dim: Dimension des inputs exogènes (4 pour [a, R, ET0, Kc])
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible")
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Initialisation Z_0 à partir de z_t
        self.init_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # f_θ(Z_k, X_k) -> [batch, hidden_dim, input_dim]
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * input_dim),
        )
        
        # Lecture finale : Z_T -> Δz_t
        self.readout = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, z_t: torch.Tensor, X_seq: torch.Tensor) -> torch.Tensor:
        """
        Prédit la variation de l'état latent en utilisant Neural CDE.
        
        Args:
            z_t: État latent initial [batch_size, latent_dim]
            X_seq: Séquence d'inputs exogènes [batch_size, seq_len, input_dim] = [a, R, ET0, Kc]
        
        Returns:
            Δz_t: Variation de l'état latent [batch_size, latent_dim]
        """
        batch_size, seq_len, _ = X_seq.size()
        
        # Initialisation Z_0 à partir de z_t
        Z = self.init_net(z_t)  # [batch_size, hidden_dim]
        
        # Intégration discrète sur la séquence
        for k in range(seq_len - 1):
            Xk = X_seq[:, k, :]  # [batch_size, input_dim]
            Xk1 = X_seq[:, k + 1, :]  # [batch_size, input_dim]
            dX = Xk1 - Xk  # [batch_size, input_dim]
            
            ZX = torch.cat([Z, Xk], dim=-1)  # [batch_size, hidden_dim + input_dim]
            f_val = self.f_net(ZX)  # [batch_size, hidden_dim * input_dim]
            f_val = f_val.view(batch_size, self.hidden_dim, self.input_dim)
            
            # Contraction : f_θ(Z_k, X_k) · ΔX_k -> [batch_size, hidden_dim]
            dZ = torch.einsum("bhi,bi->bh", f_val, dX)
            Z = Z + dZ
        
        # Lire la variation de l'état latent
        delta_z = self.readout(Z)  # [batch_size, latent_dim]
        return delta_z


class WorldModelPhase2Dataset(Dataset):
    """
    Dataset pour entraîner le World Model Phase 2 (avec décodeur).
    
    Génère des triplets (z_t, a_seq, inputs_exog_seq, z_{t+1}, obs_{t+1}) où :
    - z_t est encodé par PatchTST
    - a_seq et inputs_exog_seq sont des séquences pour le Neural CDE
    - z_{t+1} est encodé par PatchTST
    - obs_{t+1} sont les observables réels pour entraîner le décodeur
    """
    def __init__(
        self,
        encoder_model: PatchTSTFeatureExtractor,
        T: int,
        N_traj: int,
        seq_len: int,
        cde_seq_len: int,
        soil,
        max_irrigation: float,
        seed: int = 123,
        device: str = "cpu"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible")
        self.encoder_model = encoder_model
        self.seq_len = seq_len
        self.cde_seq_len = cde_seq_len
        self.device = device
        self.z_t_list = []
        self.a_seq_list = []
        self.inputs_exog_seq_list = []
        self.z_tp1_list = []
        self.obs_tp1_list = []
        
        rng = np.random.default_rng(seed)
        encoder_model.eval()
        
        with torch.no_grad():
            for n in range(N_traj):
                rain, et0, Kc = generate_weather(T=T, seed=seed + 600 + n)
                S = float(soil.S_fc)
                
                history = []  # Historique pour PatchTST
                a_history = []  # Historique des actions
                inputs_exog_history = []  # Historique des inputs exogènes
                
                for t in range(T - 1):
                    psi = soil.S_to_psi(S)
                    I = float(np.clip(
                        rng.normal(loc=max_irrigation / 2, scale=max_irrigation / 4),
                        0.0, max_irrigation
                    ))
                    
                    fET = soil.f_ET(psi)
                    ETc = Kc[t] * et0[t] * fET
                    D = soil.drainage(S)
                    
                    delta = soil.eta_I * I + rain[t] - ETc - D
                    S_next = np.clip(S + delta, 0.0, soil.S_max)
                    psi_next = soil.S_to_psi(S_next)
                    
                    # Observation actuelle
                    X_t = np.array([psi, I, rain[t], et0[t]], dtype=np.float32)
                    history.append(X_t)
                    
                    # Action et inputs exogènes pour CDE
                    a_history.append(I)
                    inputs_exog_history.append(np.array([rain[t], et0[t], Kc[t]], dtype=np.float32))
                    
                    # Observation suivante
                    X_tp1 = np.array([psi_next, I, rain[t+1] if t+1 < T-1 else rain[t], et0[t+1] if t+1 < T-1 else et0[t]], dtype=np.float32)
                    
                    # Encoder z_t si on a assez d'historique
                    if len(history) >= seq_len and len(a_history) >= cde_seq_len:
                        seq_t = np.array(history[-seq_len:], dtype=np.float32)
                        seq_t_tensor = torch.from_numpy(seq_t).unsqueeze(0).to(device)
                        z_t = encoder_model(seq_t_tensor).cpu().numpy().squeeze(0)
                        
                        # Séquence pour CDE (actions et inputs exogènes)
                        a_seq = np.array(a_history[-cde_seq_len:], dtype=np.float32)
                        inputs_exog_seq = np.array(inputs_exog_history[-cde_seq_len:], dtype=np.float32)
                        # Construire X_seq pour CDE : [a, R, ET0, Kc]
                        X_seq_cde = np.zeros((cde_seq_len, 4), dtype=np.float32)
                        X_seq_cde[:, 0] = a_seq
                        X_seq_cde[:, 1:4] = inputs_exog_seq
                        
                        # Pour z_{t+1}, on ajoute X_tp1 à l'historique
                        history_tp1 = history + [X_tp1]
                        if len(history_tp1) >= seq_len:
                            seq_tp1 = np.array(history_tp1[-seq_len:], dtype=np.float32)
                            seq_tp1_tensor = torch.from_numpy(seq_tp1).unsqueeze(0).to(device)
                            z_tp1 = encoder_model(seq_tp1_tensor).cpu().numpy().squeeze(0)
                            
                            # Observables réels pour entraîner le décodeur
                            obs_tp1 = np.array([psi_next, S_next, rain[t+1] if t+1 < T-1 else rain[t], et0[t+1] if t+1 < T-1 else et0[t]], dtype=np.float32)
                            
                            # Stocker les données
                            self.z_t_list.append(z_t)
                            self.a_seq_list.append(X_seq_cde)
                            self.inputs_exog_seq_list.append(inputs_exog_seq)
                            self.z_tp1_list.append(z_tp1)
                            self.obs_tp1_list.append(obs_tp1)
                    
                    S = S_next
        
        # Convertir en tensors
        self.z_t_list = [torch.tensor(z, dtype=torch.float32) for z in self.z_t_list]
        self.a_seq_list = [torch.tensor(a, dtype=torch.float32) for a in self.a_seq_list]
        self.z_tp1_list = [torch.tensor(z, dtype=torch.float32) for z in self.z_tp1_list]
        self.obs_tp1_list = [torch.tensor(obs, dtype=torch.float32) for obs in self.obs_tp1_list]
    
    def __len__(self):
        return len(self.z_t_list)
    
    def __getitem__(self, idx):
        return (
            self.z_t_list[idx],
            self.a_seq_list[idx],
            self.z_tp1_list[idx],
            self.obs_tp1_list[idx]
        )


def train_world_model_phase2(
    encoder_model: PatchTSTFeatureExtractor,
    soil,
    max_irrigation: float = 20.0,
    T: int = 120,
    N_traj: int = 32,
    seq_len: int = 30,
    cde_seq_len: int = 10,
    latent_dim: int = 16,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 123,
    device: str = "cpu",
    progress_callback=None
) -> Tuple[LatentTransitionCDE, LatentDecoder, float]:
    """
    Entraîne le World Model Phase 2 (encodeur PatchTST + transition Neural CDE + décodeur).
    
    Args:
        encoder_model: Modèle PatchTST pré-entraîné (encodeur)
        soil: Objet PhysicalBucket
        max_irrigation: Dose maximale d'irrigation
        T: Longueur de la saison
        N_traj: Nombre de trajectoires
        seq_len: Longueur de séquence pour PatchTST
        cde_seq_len: Longueur de séquence pour Neural CDE
        latent_dim: Dimension de l'espace latent
        n_epochs: Nombre d'epochs
        batch_size: Taille des batches
        lr: Taux d'apprentissage
        seed: Graine aléatoire
        device: Device PyTorch
        progress_callback: Callback pour la progression
    
    Returns:
        Tuple (transition_model, decoder_model, final_loss)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas disponible")
    
    device_torch = torch.device(device)
    
    # Créer les modèles
    transition_model = LatentTransitionCDE(
        latent_dim=latent_dim,
        hidden_dim=128,
        input_dim=4
    ).to(device_torch)
    
    decoder_model = LatentDecoder(
        latent_dim=latent_dim,
        hidden_dim=128,
        output_dim=4
    ).to(device_torch)
    
    # Créer le dataset
    dataset = WorldModelPhase2Dataset(
        encoder_model=encoder_model,
        T=T,
        N_traj=N_traj,
        seq_len=seq_len,
        cde_seq_len=cde_seq_len,
        soil=soil,
        max_irrigation=max_irrigation,
        seed=seed,
        device=device
    )
    
    if len(dataset) == 0:
        raise ValueError("Le dataset est vide. Vérifiez les paramètres (T, seq_len, cde_seq_len, N_traj).")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(
        list(transition_model.parameters()) + list(decoder_model.parameters()),
        lr=lr
    )
    criterion = nn.MSELoss()
    
    transition_model.train()
    decoder_model.train()
    encoder_model.eval()  # L'encodeur est figé
    
    final_loss = 0.0
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (z_t, X_seq_cde, z_tp1, obs_tp1) in enumerate(dataloader):
            z_t = z_t.to(device_torch)
            X_seq_cde = X_seq_cde.to(device_torch)
            z_tp1 = z_tp1.to(device_torch)
            obs_tp1 = obs_tp1.to(device_torch)
            
            # Prédire la transition dans l'espace latent
            delta_z = transition_model(z_t, X_seq_cde)  # [batch_size, latent_dim]
            z_tp1_pred = z_t + delta_z  # [batch_size, latent_dim]
            
            # Loss de transition (prédire z_{t+1})
            loss_transition = criterion(z_tp1_pred, z_tp1)
            
            # Décodeur : reconstruire les observables depuis z_{t+1}
            obs_tp1_pred = decoder_model(z_tp1_pred)  # [batch_size, 4]
            
            # Loss de reconstruction (reconstruire obs_{t+1})
            loss_decoder = criterion(obs_tp1_pred, obs_tp1)
            
            # Loss totale (pondérée)
            loss = loss_transition + loss_decoder
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        final_loss = avg_loss
        
        if progress_callback:
            progress = min(1.0, max(0.0, (epoch + 1) / n_epochs))
            progress_callback(progress, f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    transition_model.eval()
    decoder_model.eval()
    return transition_model, decoder_model, final_loss


def train_ppo_with_world_model_phase2(
    encoder_model: PatchTSTFeatureExtractor,
    transition_model: LatentTransitionCDE,
    decoder_model: LatentDecoder,
    soil_params: Optional[Dict[str, float]] = None,
    weather_params: Optional[Dict[str, float]] = None,
    season_length: Optional[int] = None,
    max_irrigation: Optional[float] = None,
    total_timesteps: Optional[int] = None,
    imagination_horizon: int = 20,
    imagination_ratio: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress_callback=None,
    **ppo_kwargs
):
    """
    Entraîne un agent PPO avec World Model Phase 2 (rollouts d'imagination longs avec décodeur).
    
    Args:
        encoder_model: Encodeur PatchTST pré-entraîné
        transition_model: Modèle de transition Neural CDE pré-entraîné
        decoder_model: Décodeur pré-entraîné
        soil_params: Paramètres du sol
        season_length: Longueur de saison (défaut depuis utils_physics_config)
        max_irrigation: Dose maximale d'irrigation (défaut depuis utils_physics_config)
        total_timesteps: Nombre total de pas d'entraînement (défaut depuis utils_physics_config)
        imagination_horizon: Horizon des rollouts d'imagination (20-30 pas pour Phase 2)
        imagination_ratio: Ratio de rollouts d'imagination
        seed: Graine aléatoire (défaut depuis utils_physics_config)
        device: Device PyTorch
        progress_callback: Callback pour la progression
        **ppo_kwargs: Arguments additionnels pour PPO
    
    Returns:
        Tuple (Modèle PPO entraîné, dictionnaire des métriques d'entraînement)
    """
    if not PPO_AVAILABLE:
        raise ImportError("stable-baselines3 n'est pas disponible")
    
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
    
    from src.utils_env_modeles import IrrigationEnvPhysical
    
    # Créer l'environnement réel
    def make_env():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,
            weather_params=weather_params,
            device=device
        )
        # Envelopper avec PatchTST pour encoder les observations
        if PatchTSTEnvWrapper is not None:
            env = PatchTSTEnvWrapper(
                env,
                patchtst_model=encoder_model,
                seq_len=30,
                device=device
            )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Importer les utilitaires PPO
    from src.utils_ppo_training import create_ppo_callbacks, get_default_ppo_config
    
    # Créer les callbacks (progression + collecte de métriques)
    callbacks, metrics_callback = create_ppo_callbacks(
        progress_callback=progress_callback,
        total_timesteps=total_timesteps
    )
    
    # Configuration PPO par défaut
    ppo_config = get_default_ppo_config(**ppo_kwargs)
    
    # Extraire 'policy' du config si présent (car il est passé comme argument positionnel)
    policy = ppo_config.pop("policy", "MlpPolicy")
    
    # Créer et entraîner le modèle
    model = PPO(policy, env, **ppo_config)
    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)
    
    # Récupérer les métriques finales
    training_metrics = metrics_callback.get_final_metrics()
    
    return model, training_metrics


# ============================================================================
# SCÉNARIO 6 : WORLD MODEL - PHASE 3 (PHYSICS-INFORMED)
# ============================================================================

if PPO_AVAILABLE and GYM_AVAILABLE and Wrapper is not None:
    class PhysicsInformedWorldModelWrapper(Wrapper):
        """
        Wrapper qui hybridise le modèle physique (env de base) avec un World Model.

        Principe :
        - On laisse l'environnement physique évoluer normalement.
        - On encode l'historique avec PatchTST, fait une transition latente (ODE/CDE)
          puis on décode les observables.
        - On mélange (blend) la prédiction physique et la prédiction du world model
          pour ψ et S afin d'ajouter une contrainte physique (physics-informed).
        """
        def __init__(
            self,
            env,
            encoder_model: PatchTSTFeatureExtractor,
            transition_model: nn.Module,
            decoder_model: LatentDecoder,
            seq_len: int = 30,
            hybrid_alpha: float = 0.5,
            device: str = "cpu"
        ):
            super().__init__(env)
            self.encoder_model = encoder_model.to(device)
            self.transition_model = transition_model.to(device)
            self.decoder_model = decoder_model.to(device)
            self.seq_len = seq_len
            self.hybrid_alpha = hybrid_alpha
            self.device = device
            self.history = []

            # Mode évaluation pour stabiliser les prédictions
            self.encoder_model.eval()
            self.transition_model.eval()
            self.decoder_model.eval()

        def reset(self, *, seed=None, options=None):
            obs, info = self.env.reset(seed=seed, options=options)
            self.history = [obs]
            # Synchroniser les attributs exposés (pour compatibilité evaluate_episode)
            self.psi = float(obs[0]) if len(obs) > 0 else getattr(self.env, "psi", 0.0)
            self.S = float(obs[1]) if len(obs) > 1 else getattr(self.env, "S", 0.0)
            return obs, info

        def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).float().to(self.device)

        def _get_inputs_exog(self, obs: np.ndarray) -> torch.Tensor:
            t = max(0, getattr(self.env, "day", 1) - 1)
            Kc_t = 1.0
            if hasattr(self.env, "Kc"):
                try:
                    if t < len(self.env.Kc):
                        Kc_t = float(self.env.Kc[t])
                except Exception:
                    Kc_t = 1.0
            inputs = np.array([[obs[2], obs[3], Kc_t]], dtype=np.float32)
            return self._to_tensor(inputs)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if info is None:
                info = {}
            # Historique des observations physiques
            self.history.append(obs)
            if len(self.history) > self.seq_len:
                self.history = self.history[-self.seq_len:]

            # Lorsque suffisamment d'historique est disponible, appliquer le blend
            if len(self.history) >= 2:
                seq = np.array(self.history[-self.seq_len:], dtype=np.float32)
                seq_tensor = self._to_tensor(seq).unsqueeze(0)  # [1, L, 4]

                with torch.no_grad():
                    z_t = self.encoder_model(seq_tensor)  # [1, latent_dim]
                    a_val = float(action[0] if hasattr(action, "__len__") else action)
                    a_tensor = torch.tensor([[a_val]], dtype=torch.float32, device=self.device)
                    delta_z = None
                    # Essayer le mode CDE si le modèle le supporte (input_dim attribut)
                    try:
                        if hasattr(self.transition_model, "input_dim"):
                            Kc_t = 1.0
                            if hasattr(self, "_get_inputs_exog"):
                                try:
                                    Kc_t = float(self._get_inputs_exog(obs).cpu().numpy().squeeze()[-1])
                                except Exception:
                                    Kc_t = 1.0
                            X_row = np.array([a_val, obs[2], obs[3], Kc_t], dtype=np.float32)
                            n = max(2, self.seq_len)
                            X_seq = np.tile(X_row, (n, 1)).astype(np.float32)
                            X_seq_tensor = torch.from_numpy(X_seq).unsqueeze(0).to(self.device)
                            delta_z = self.transition_model(z_t, X_seq_tensor)
                    except TypeError:
                        delta_z = None

                    if delta_z is None:
                        exog = self._get_inputs_exog(obs)
                        delta_z = self.transition_model(z_t, a_tensor, exog)

                    z_next = z_t + delta_z
                    decoded = self.decoder_model(z_next).cpu().numpy().squeeze(0)

                psi_phys = float(obs[0])
                S_phys = float(obs[1])
                psi_hat = float(decoded[0]) if decoded.shape[0] > 0 else psi_phys
                S_hat = float(decoded[1]) if decoded.shape[0] > 1 else S_phys

                psi_blend = (1.0 - self.hybrid_alpha) * psi_phys + self.hybrid_alpha * psi_hat
                S_blend = (1.0 - self.hybrid_alpha) * S_phys + self.hybrid_alpha * S_hat

                # Injection des valeurs hybridées dans l'env physique
                if hasattr(self.env, "psi"):
                    self.env.psi = psi_blend
                if hasattr(self.env, "S"):
                    self.env.S = S_blend

                obs = np.array([psi_blend, S_blend, obs[2], obs[3]], dtype=np.float32)
                info.update({
                    "psi_phys": psi_phys,
                    "psi_wm": psi_hat,
                    "psi_blend": psi_blend
                })

                # Exposer aussi sur le wrapper
                self.psi = psi_blend
                self.S = S_blend

            return obs, reward, terminated, truncated, info

        def __getattr__(self, name):
            # Délègue aux attributs de l'env sous-jacent pour compatibilité
            return getattr(self.env, name)


def train_ppo_with_world_model_phase3(
    encoder_model: PatchTSTFeatureExtractor,
    transition_model: nn.Module,
    decoder_model: LatentDecoder,
    soil_params: Optional[Dict[str, float]] = None,
    weather_params: Optional[Dict[str, float]] = None,
    season_length: Optional[int] = None,
    max_irrigation: Optional[float] = None,
    total_timesteps: Optional[int] = None,
    seq_len_wm: int = 30,
    hybrid_alpha: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress_callback=None,
    **ppo_kwargs
):
    """
    Entraîne un agent PPO (Phase 3) en hybridant modèle physique + World Model.

    Args:
        encoder_model: Encodeur PatchTST pré-entraîné
        transition_model: Modèle de transition latent (ODE ou CDE)
        decoder_model: Décodeur latent -> observables
        soil_params: Paramètres du sol
        weather_params: Paramètres météo (alignement pluie)
        season_length: Longueur de la saison
        max_irrigation: Dose max d'irrigation
        total_timesteps: Pas d'entraînement PPO
        seq_len_wm: Longueur d'historique pour l'encodeur
        hybrid_alpha: Poids du blend (0 = physique pur, 1 = world model pur)
        seed: Graine aléatoire
        device: Device PyTorch
        progress_callback: Callback progression
        **ppo_kwargs: Hyperparamètres PPO supplémentaires
    """
    if not PPO_AVAILABLE:
        raise ImportError("stable-baselines3 n'est pas disponible")
    if 'PhysicsInformedWorldModelWrapper' not in globals():
        raise ImportError("PhysicsInformedWorldModelWrapper indisponible (Gym/PyTorch manquants)")

    from src.utils_physics_config import get_default_config
    from src.utils_ppo_training import create_ppo_callbacks, get_default_ppo_config

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

    from src.utils_env_modeles import IrrigationEnvPhysical

    def make_env():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,
            weather_params=weather_params,
            device=device
        )
        env = PhysicsInformedWorldModelWrapper(
            env,
            encoder_model=encoder_model,
            transition_model=transition_model,
            decoder_model=decoder_model,
            seq_len=seq_len_wm,
            hybrid_alpha=hybrid_alpha,
            device=device
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])

    callbacks, metrics_callback = create_ppo_callbacks(
        progress_callback=progress_callback,
        total_timesteps=total_timesteps
    )

    ppo_config = get_default_ppo_config(**ppo_kwargs)
    policy = ppo_config.pop("policy", "MlpPolicy")

    model = PPO(policy, env, **ppo_config)
    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)

    training_metrics = metrics_callback.get_final_metrics()
    return model, training_metrics
