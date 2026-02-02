"""
Utilitaires pour le Scénario 5 (PatchTST).

Ce module contient :
- Scénario 5 (PatchTST) :
  * PatchTSTFeatureExtractor : Extracteur de features temporelles basé sur Transformer
  * PatchTSTFeatureDataset : Dataset pour pré-entraîner PatchTST
  * pretrain_patchtst_features : Fonction de pré-entraînement
  * PatchTSTEnvWrapper : Wrapper d'environnement pour enrichir les observations
  * train_ppo_with_patchtst : Fonction d'entraînement PPO avec PatchTST

Note : Les classes et fonctions du Scénario 6 (World Model) sont dans utils_world_model.py
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


# ============================================================================
# SCÉNARIO 5 : PATCHTST - CLASSES ET FONCTIONS
# ============================================================================

class PatchTSTFeatureExtractor(nn.Module):
    """
    Extracteur de features temporelles basé sur PatchTST (Time Series Transformer).
    
    Principe : Utilise un Transformer avec patchification pour extraire des features
    temporelles avancées (tendance, saisonnalité, motifs) à partir d'une séquence
    d'états passés.
    
    Architecture simplifiée de PatchTST :
    - Patchification : Découpe la séquence en patches
    - Embedding : Transforme chaque patch en vecteur
    - Transformer : Attention multi-têtes sur les patches
    - Pooling : Agrégation pour extraire des features globales
    """
    def __init__(
        self,
        input_dim: int = 4,  # [psi, I, R, ET0]
        patch_len: int = 5,  # Longueur d'un patch (jours)
        stride: int = 1,  # Stride pour la patchification
        d_model: int = 64,  # Dimension du modèle Transformer
        n_heads: int = 4,  # Nombre de têtes d'attention
        n_layers: int = 2,  # Nombre de couches Transformer
        feature_dim: int = 16,  # Dimension des features extraites
        max_seq_len: int = 60,  # Longueur maximale de la séquence
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible")
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        
        # Calcul du nombre de patches
        self.n_patches = (max_seq_len - patch_len) // stride + 1
        
        # Embedding des patches : [patch_len * input_dim] -> [d_model]
        self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)
        
        # Positional encoding pour les patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Pooling et extraction de features
        # On extrait plusieurs types de features : mean, std, trend, etc.
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feature_dim),
        )
        
        # Pooling global (mean pooling sur les patches)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        """
        Extrait des features temporelles à partir d'une séquence d'états.
        
        Args:
            X_seq: Tensor de forme [batch, seq_len, input_dim]
                   où seq_len <= max_seq_len
                   X_k = [psi_k, I_k, R_k, ET0_k]
        
        Returns:
            features: Tensor de forme [batch, feature_dim]
                     Features temporelles extraites (tendance, saisonnalité, motifs, etc.)
        """
        batch_size, seq_len, d = X_seq.size()
        assert d == self.input_dim
        assert seq_len <= self.max_seq_len
        
        # Padding si nécessaire pour avoir exactement max_seq_len
        if seq_len < self.max_seq_len:
            padding = torch.zeros(
                batch_size, 
                self.max_seq_len - seq_len, 
                self.input_dim,
                device=X_seq.device
            )
            X_seq = torch.cat([X_seq, padding], dim=1)
        
        # Patchification : découper la séquence en patches
        patches = []
        for i in range(0, self.max_seq_len - self.patch_len + 1, self.stride):
            patch = X_seq[:, i:i+self.patch_len, :]  # [batch, patch_len, input_dim]
            patch_flat = patch.reshape(batch_size, -1)  # [batch, patch_len * input_dim]
            patches.append(patch_flat)
        
        # Stack les patches : [batch, n_patches, patch_len * input_dim]
        patches_tensor = torch.stack(patches, dim=1)
        
        # Embedding des patches : [batch, n_patches, d_model]
        patch_embeds = self.patch_embedding(patches_tensor)
        
        # Ajouter positional encoding
        patch_embeds = patch_embeds + self.pos_encoding[:, :patch_embeds.size(1), :]
        
        # Transformer encoder : [batch, n_patches, d_model]
        encoded = self.transformer(patch_embeds)
        
        # Pooling global : moyenne sur les patches
        # encoded: [batch, n_patches, d_model]
        # On transpose pour le pooling : [batch, d_model, n_patches]
        encoded_pooled = self.global_pool(encoded.transpose(1, 2))  # [batch, d_model, 1]
        encoded_pooled = encoded_pooled.squeeze(-1)  # [batch, d_model]
        
        # Extraction de features finales
        features = self.feature_extractor(encoded_pooled)  # [batch, feature_dim]
        
        return features


class PatchTSTFeatureDataset(Dataset):
    """
    Dataset pour pré-entraîner PatchTST comme extracteur de features.
    
    Génère des séquences de trajectoires simulées pour apprendre des
    représentations temporelles utiles (auto-supervisé ou supervisé).
    """
    def __init__(
        self, 
        T: int, 
        N_traj: int, 
        seq_len: int, 
        soil, 
        max_irrigation: float, 
        seed: int = 123,
        task: str = "auto"  # "auto" (auto-supervisé) ou "supervised" (supervisé)
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible")
        self.X_seq_list = []
        self.task = task
        rng = np.random.default_rng(seed)
        
        if task == "supervised":
            # Pour tâche supervisée : prédire des features cibles (tendance, etc.)
            self.Y_list = []
        
        for n in range(N_traj):
            # Générer météo pour cette trajectoire
            rain, et0, Kc = generate_weather(T=T, seed=seed + 300 + n)
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
                
                # Mise à jour
                delta = soil.eta_I * I + rain[t] - ETc - D
                S = np.clip(S + delta, 0.0, soil.S_max)
                
                # Features : [psi, I, R, ET0]
                X_t = np.array([psi, I, rain[t], et0[t]], dtype=np.float32)
                history.append(X_t)
                
                # Créer des séquences de longueur seq_len
                if len(history) >= seq_len:
                    # Séquence : [X_{t-seq_len+1}, ..., X_t]
                    seq = np.array(history[-seq_len:], dtype=np.float32)
                    self.X_seq_list.append(seq)
                    
                    if task == "supervised":
                        # Features cibles : tendance, variance, etc.
                        psi_seq = seq[:, 0]  # Séquence de psi
                        trend = (psi_seq[-1] - psi_seq[0]) / seq_len  # Tendance
                        variance = np.var(psi_seq)  # Variance
                        mean_psi = np.mean(psi_seq)  # Moyenne
                        # Features cibles combinées
                        y = np.array([trend, variance, mean_psi], dtype=np.float32)
                        self.Y_list.append(y)
        
        self.X_seq_list = [torch.tensor(x, dtype=torch.float32) for x in self.X_seq_list]
        if task == "supervised":
            self.Y_list = [torch.tensor(y, dtype=torch.float32) for y in self.Y_list]
    
    def __len__(self):
        return len(self.X_seq_list)
    
    def __getitem__(self, idx):
        if self.task == "auto":
            return self.X_seq_list[idx]  # Auto-supervisé : reconstruction
        else:
            return self.X_seq_list[idx], self.Y_list[idx]  # Supervisé : prédiction de features


def pretrain_patchtst_features(
    soil,
    max_irrigation: float = 20.0,
    T: int = 120,
    N_traj: int = 32,
    seq_len: int = 30,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 123,
    device: str = "cpu",
    progress_callback=None,
    feature_dim: int = 16,
    task: str = "auto"
) -> Tuple[PatchTSTFeatureExtractor, float]:
    """
    Pré-entraîne PatchTST comme extracteur de features temporelles.
    
    Args:
        soil: Instance de PhysicalBucket
        max_irrigation: Dose maximale d'irrigation
        T: Longueur de saison
        N_traj: Nombre de trajectoires à générer
        seq_len: Longueur de la séquence pour PatchTST (historique)
        n_epochs: Nombre d'epochs d'entraînement
        batch_size: Taille des batches
        lr: Taux d'apprentissage
        seed: Graine aléatoire
        device: Device PyTorch ('cpu' ou 'cuda')
        progress_callback: Fonction appelée à chaque epoch pour le suivi
        feature_dim: Dimension des features extraites
        task: "auto" (auto-supervisé) ou "supervised" (supervisé)
    
    Returns:
        Tuple (Modèle PatchTSTFeatureExtractor pré-entraîné, loss finale)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas disponible")
    
    device_torch = torch.device(device)
    
    # Créer le modèle
    model = PatchTSTFeatureExtractor(
        input_dim=4,
        patch_len=5,
        stride=1,
        d_model=64,
        n_heads=4,
        n_layers=2,
        feature_dim=feature_dim,
        max_seq_len=seq_len
    ).to(device_torch)
    
    # Créer le dataset
    dataset = PatchTSTFeatureDataset(
        T=T,
        N_traj=N_traj,
        seq_len=seq_len,
        soil=soil,
        max_irrigation=max_irrigation,
        seed=seed,
        task=task
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimiseur
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Fonction de perte selon la tâche
    if task == "auto":
        # Auto-supervisé : reconstruction (on essaie de reconstruire la séquence)
        # On utilise une approche de contrastive learning ou de reconstruction
        criterion = nn.MSELoss()
        # Pour l'auto-supervisé, on prédit des statistiques de la séquence
        predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len * 4)  # Reconstruire la séquence
        ).to(device_torch)
        optimizer.add_param_group({'params': predictor.parameters()})
    else:
        # Supervisé : prédiction de features cibles
        criterion = nn.MSELoss()
        predictor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [trend, variance, mean]
        ).to(device_torch)
        optimizer.add_param_group({'params': predictor.parameters()})
    
    # Entraînement
    model.train()
    if task == "supervised":
        predictor.train()
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            if task == "auto":
                X_seq = batch_data.to(device_torch)  # [batch, seq_len, 4]
                # Features extraites
                features = model(X_seq)  # [batch, feature_dim]
                # Reconstruction
                X_recon = predictor(features)  # [batch, seq_len * 4]
                X_recon = X_recon.view(X_seq.size(0), X_seq.size(1), -1)
                # Perte de reconstruction
                loss = criterion(X_recon, X_seq)
            else:
                X_seq, Y_target = batch_data
                X_seq = X_seq.to(device_torch)
                Y_target = Y_target.to(device_torch)
                # Features extraites
                features = model(X_seq)  # [batch, feature_dim]
                # Prédiction des features cibles
                Y_pred = predictor(features)  # [batch, 3]
                # Perte
                loss = criterion(Y_pred, Y_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Callback de progression
        if progress_callback:
            progress = min(1.0, max(0.0, (epoch + 1) / n_epochs))
            progress_callback(progress, f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    model.eval()
    final_loss = avg_loss  # Garder la dernière loss moyenne
    return model, final_loss


# ============================================================================
# SCÉNARIO 5 : PATCHTST - WRAPPER D'ENVIRONNEMENT
# ============================================================================

if PPO_AVAILABLE and GYM_AVAILABLE and Wrapper is not None:
    class PatchTSTEnvWrapper(Wrapper):
        """
        Wrapper d'environnement qui enrichit l'observation avec des features PatchTST.
        
        Principe : À chaque pas de temps, on extrait des features temporelles
        à partir de l'historique des états et on les ajoute à l'observation.
        """
        def __init__(
            self,
            env,
            patchtst_model: PatchTSTFeatureExtractor,
            seq_len: int = 30,
            device: str = "cpu"
        ):
            super().__init__(env)
            self.patchtst_model = patchtst_model
            self.seq_len = seq_len
            self.device = device
            
            # Historique des états pour PatchTST
            self.history = []
            
            # Mettre à jour l'espace d'observation
            # Observation originale : [psi, S, R, ET0] (4 dimensions)
            # Features PatchTST : feature_dim dimensions
            # Observation enrichie : [psi, S, R, ET0, features...] (4 + feature_dim dimensions)
            original_low = self.observation_space.low
            original_high = self.observation_space.high
            feature_dim = patchtst_model.feature_dim
            
            # Bounds pour les features (normalisées, typiquement entre -1 et 1)
            feature_low = np.full(feature_dim, -10.0, dtype=np.float32)
            feature_high = np.full(feature_dim, 10.0, dtype=np.float32)
            
            new_low = np.concatenate([original_low, feature_low])
            new_high = np.concatenate([original_high, feature_high])
            
            self.observation_space = spaces.Box(
                low=new_low,
                high=new_high,
                dtype=np.float32
            )
            
            # Mettre le modèle en mode évaluation
            self.patchtst_model.eval()
        
        def reset(self, *, seed=None, options=None):
            """Reset l'environnement et l'historique."""
            obs, info = self.env.reset(seed=seed, options=options)
            self.history = []
            # Ajouter l'état initial à l'historique
            self.history.append(obs)
            # Enrichir l'observation avec PatchTST
            obs_enriched = self._enrich_observation(obs)
            return obs_enriched, info
        
        def step(self, action):
            """Step avec observation enrichie."""
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Ajouter l'état à l'historique
            self.history.append(obs)
            # Garder seulement les seq_len derniers états
            if len(self.history) > self.seq_len:
                self.history = self.history[-self.seq_len:]
            # Enrichir l'observation avec PatchTST
            obs_enriched = self._enrich_observation(obs)
            return obs_enriched, reward, terminated, truncated, info
        
        def _enrich_observation(self, obs: np.ndarray) -> np.ndarray:
            """
            Enrichit l'observation avec des features PatchTST.
            
            Args:
                obs: Observation originale [psi, S, R, ET0]
            
            Returns:
                obs_enriched: Observation enrichie [psi, S, R, ET0, features...]
            """
            # Si on n'a pas assez d'historique, on retourne l'observation originale
            # avec des features à zéro
            if len(self.history) < 2:
                feature_dim = self.patchtst_model.feature_dim
                features = np.zeros(feature_dim, dtype=np.float32)
                return np.concatenate([obs, features])
            
            # Construire la séquence pour PatchTST
            # On prend les seq_len derniers états (ou moins si on n'en a pas assez)
            seq = np.array(self.history[-self.seq_len:], dtype=np.float32)
            
            # Ajouter une dimension batch : [1, seq_len, 4]
            seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            
            # Extraire les features avec PatchTST
            with torch.no_grad():
                features = self.patchtst_model(seq_tensor)  # [1, feature_dim]
                features = features.cpu().numpy().squeeze(0)  # [feature_dim]
            
            # Concaténer l'observation originale avec les features
            obs_enriched = np.concatenate([obs, features])
            return obs_enriched.astype(np.float32)
        
        def __getattr__(self, name):
            """
            Délègue tous les attributs non définis à l'environnement sous-jacent.
            Permet d'accéder à env.psi, env.S, env.rain, etc. depuis le wrapper.
            """
            return getattr(self.env, name)
else:
    # Si Wrapper n'est pas disponible, définir une classe factice
    PatchTSTEnvWrapper = None


def train_ppo_with_patchtst(
    soil_params: Dict[str, float],
    season_length: Optional[int] = None,
    max_irrigation: Optional[float] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    weather_params: Optional[Dict[str, float]] = None,
    patchtst_model: Optional[PatchTSTFeatureExtractor] = None,
    seq_len_patchtst: int = 30,
    device: str = "cpu",
    progress_callback=None,
    lexico_config=None,
    goal_spec: Optional[Dict[str, Any]] = None,
    teacher_model=None,
    distill_coef: float = 0.0,
    weather_shift_cfg: Optional[Dict[str, Any]] = None,
    external_weather: Optional[Dict[str, Any]] = None,
    **ppo_kwargs
):
    """
    Entraîne un agent PPO avec observation enrichie par PatchTST.
    
    Args:
        soil_params: Paramètres du sol
        season_length: Longueur de saison (défaut depuis utils_physics_config)
        max_irrigation: Dose maximale d'irrigation (défaut depuis utils_physics_config)
        total_timesteps: Nombre total de pas d'entraînement (défaut depuis utils_physics_config)
        seed: Graine aléatoire (défaut depuis utils_physics_config)
        weather_params: Paramètres météo (aligner la pluie avec le scénario 1)
        patchtst_model: Modèle PatchTST pré-entraîné (optionnel)
        seq_len_patchtst: Longueur de séquence pour PatchTST
        device: Device PyTorch
        progress_callback: Callback pour le suivi de progression
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
    try:
        from src.utils_lexico_goal import wrap_with_lexico
    except Exception:
        wrap_with_lexico = None  # type: ignore
    
    # Créer l'environnement de base
    def make_env():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,
            weather_params=weather_params,
            device=device,
            goal_spec=goal_spec,
            weather_shift_cfg=weather_shift_cfg,
            external_weather=external_weather,
        )
        # Envelopper avec PatchTST si un modèle est fourni
        if patchtst_model is not None and PatchTSTEnvWrapper is not None:
            env = PatchTSTEnvWrapper(
                env,
                patchtst_model=patchtst_model,
                seq_len=seq_len_patchtst,
                device=device
            )
        if wrap_with_lexico is not None:
            env = wrap_with_lexico(env, lexico_config)
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
    if teacher_model is not None and distill_coef > 0:
        try:
            from src.utils_distillation import DistillationRewardCallback
            callbacks = callbacks or []
            callbacks.append(DistillationRewardCallback(teacher_model, coef=distill_coef))
        except Exception:
            pass
    
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
    
