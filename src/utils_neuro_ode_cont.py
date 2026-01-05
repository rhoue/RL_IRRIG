"""
Neural ODE continu (correction résiduelle) pour l'irrigation intelligente.

Ce module propose une variante continue du modèle Neural ODE utilisé dans
le scénario 3. La dérivée dψ/dt est apprise par un MLP et intégrée sur un
pas de temps (par défaut 1 jour) via torchdiffeq si disponible, sinon via
un schéma d'Euler explicite.
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

# Imports torchdiffeq (optionnel)
try:
    from torchdiffeq import odeint  # type: ignore
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None

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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils_weather import generate_weather


# ============================================================================
# CLASSES
# ============================================================================

class ContinuousResidualODEFunc(nn.Module):
    """
    Modélise la dérivée dψ/dt = f_θ(ψ, I, R, ET0).
    """

    def __init__(self, hidden: int = 64):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t: torch.Tensor, psi: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Temps (inutile ici mais requis par l'API odeint)
            psi: Tension actuelle [batch, 1]
            controls: Features [batch, 3] = [I, R, ET0]
        Returns:
            dpsi/dt [batch, 1]
        """
        x = torch.cat([psi, controls], dim=1)  # [batch, 4]
        return self.net(x)


class ContinuousResidualODE(nn.Module):
    """
    Intègre dψ/dt pour produire Δψ sur un pas (dt) via torchdiffeq ou Euler.
    """

    def __init__(self, hidden: int = 64, rtol: float = 1e-4, atol: float = 1e-4, solver: str = "rk4"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible.")
        super().__init__()
        self.func = ContinuousResidualODEFunc(hidden=hidden)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver

    def forward(self, x: torch.Tensor, dt: float = 1.0, substeps: int = 4) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch, 4] = [psi, I, R, ET0]
            dt: Durée d'intégration (jours)
            substeps: Nombre de sous-pas si torchdiffeq indisponible
        Returns:
            Δψ intégré sur dt [batch, 1]
        """
        psi0 = x[:, :1]
        controls = x[:, 1:]
        device = psi0.device

        if TORCHDIFFEQ_AVAILABLE:
            t_span = torch.tensor([0.0, float(dt)], device=device, dtype=psi0.dtype)
            # Lie controls dans une closure pour éviter de passer args (certains backends ne le supportent pas)
            def ode_func(t, psi):
                return self.func(t, psi, controls)
            psi_traj = odeint(
                ode_func,
                psi0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
            )
            psi_end = psi_traj[-1]
        else:
            # Euler explicite avec sous-pas pour stabiliser l'intégration
            h = dt / max(1, substeps)
            psi_curr = psi0
            for _ in range(max(1, substeps)):
                dpsi = self.func(None, psi_curr, controls)
                psi_curr = psi_curr + h * dpsi
            psi_end = psi_curr

        return psi_end - psi0


class ContinuousResidualODEDataset(Dataset):
    """
    Dataset pour pré-entraîner la version continue (cible = Δψ sur 1 pas).
    """

    def __init__(self, T: int, N_traj: int, soil, max_irrigation: float, seed: int = 123):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible.")
        self.X, self.Y = [], []
        rng = np.random.default_rng(seed)

        for n in range(N_traj):
            rain, et0, Kc = generate_weather(T=T, seed=seed + 150 + n)
            S = float(soil.S_fc)

            for t in range(T - 1):
                psi = soil.S_to_psi(S)
                I = float(np.clip(
                    rng.normal(loc=max_irrigation / 2, scale=max_irrigation / 4),
                    0.0, max_irrigation
                ))

                fET = soil.f_ET(psi)
                ETc = Kc[t] * et0[t] * fET
                D = soil.drainage(S)

                delta_true = soil.eta_I * I + rain[t] - ETc - D + rng.normal(0.0, 0.2)
                S_true = np.clip(S + delta_true, 0.0, soil.S_max)

                delta_phys = soil.eta_I * I + rain[t] - ETc - D
                S_phys = np.clip(S + delta_phys, 0.0, soil.S_max)

                psi_true = soil.S_to_psi(S_true)
                psi_phys = soil.S_to_psi(S_phys)
                delta_psi = psi_true - psi_phys

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

def pretrain_continuous_residual_ode(
    soil,
    max_irrigation: Optional[float] = None,
    T: Optional[int] = None,
    N_traj: int = 32,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress_callback=None,
) -> Any:
    """
    Pré-entraîne le modèle Neural ODE continu (Δψ intégré sur dt=1).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas installé. Installez-le avec: pip install torch")

    from src.utils_physics_config import get_default_physics_config

    default_config = get_default_physics_config(
        max_irrigation=max_irrigation,
        season_length=T,
        seed=seed,
    )
    max_irrigation = default_config["max_irrigation"]
    T = default_config["season_length"]
    seed = default_config["seed"]

    model = ContinuousResidualODE(hidden=64).to(device)
    dataset = ContinuousResidualODEDataset(T, N_traj, soil, max_irrigation, seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for epoch in range(1, n_epochs + 1):
        losses = []
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)  # Δψ prédit
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        if progress_callback:
            progress_callback(epoch, n_epochs, avg_loss)

    model.eval()
    final_loss = avg_loss
    return model, final_loss


def train_ppo_hybrid_ode_cont(
    residual_ode_model: nn.Module,
    season_length: Optional[int] = None,
    max_irrigation: Optional[float] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    soil_params: Optional[Dict] = None,
    weather_params: Optional[Dict] = None,
    ppo_kwargs: Optional[Dict] = None,
    progress_callback=None,
    teacher_model=None,
    distill_coef: float = 0.0,
    weather_shift_cfg: Optional[Dict[str, Any]] = None,
):
    """
    Entraîne PPO avec un résiduel Neural ODE continu (drop-in pour residual_ode).
    """
    if not PPO_AVAILABLE:
        raise ImportError("stable-baselines3 n'est pas installé")

    from src.utils_physics_config import get_default_config

    default_config = get_default_config(
        season_length=season_length,
        max_irrigation=max_irrigation,
        total_timesteps=total_timesteps,
        seed=seed,
    )
    season_length = default_config["season_length"]
    max_irrigation = default_config["max_irrigation"]
    total_timesteps = default_config["total_timesteps"]
    seed = default_config["seed"]

    try:
        from src.utils_env_modeles import IrrigationEnvPhysical
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils_env_modeles import IrrigationEnvPhysical

    def make_env():
        env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,
            weather_params=weather_params,
            residual_ode=residual_ode_model,
            device="cpu",
            weather_shift_cfg=weather_shift_cfg,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])


    from src.utils_ppo_training import create_ppo_callbacks, get_default_ppo_config

    ppo_overrides = dict(ppo_kwargs or {})
    ppo_overrides.setdefault("ent_coef", 0.0)
    ppo_overrides.setdefault("verbose", 1)
    ppo_config = get_default_ppo_config(**ppo_overrides)
    policy = ppo_config.pop("policy", "MlpPolicy")

    callbacks, metrics_callback = create_ppo_callbacks(
        progress_callback=progress_callback,
        total_timesteps=total_timesteps,
    )
    if teacher_model is not None and distill_coef > 0:
        try:
            from src.utils_distillation import DistillationRewardCallback
            callbacks = callbacks or []
            callbacks.append(DistillationRewardCallback(teacher_model, coef=distill_coef))
        except Exception:
            pass

    model = PPO(policy, env, **ppo_config)
    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)

    training_metrics = metrics_callback.get_final_metrics()
    return model, training_metrics
