# RL Intelligent Irrigation

Research prototype for intelligent irrigation using reinforcement learning (RL).
The system combines a physical soil water balance model with RL policies and hybrid
neural corrections (Neural ODE / Neural CDE) to optimize irrigation while keeping
soil tension in a comfort zone and minimizing water losses.

## Scenarios

The project implements four progressive scenarios:

1. **Scenario 1 — Physical model + simple rules**
   - FAO-style bucket model with fixed irrigation rules (threshold, comfort band, proportional).
2. **Scenario 2 — RL on physical model**
   - PPO agent trained directly on the physical environment.
3. **Scenario 3 — Hybrid RL with Neural ODE**
   - Physical prediction corrected by a Neural ODE residual model.
4. **Scenario 4 — Hybrid RL with Neural CDE**
   - Handles irregular observations with a Neural CDE residual model.

> **Scenario 3b (continuous Neural ODE)** is a continuous-time variant of Scenario 3.

## Model overview

The irrigation environment models daily soil water dynamics:

- **State/observations** include soil tension `psi`, storage `S`, rain `R`, ET0, and crop coefficient `Kc`.
- **Action** is the irrigation dose `I_t` in mm/day.
- **Reward** penalizes water stress and excess irrigation/drainage.

Hybrid scenarios learn a residual correction:

- **Neural ODE (Scenario 3)** learns `Δψ` to correct the physical update:
  `ψ_{t+1} = ψ_{t+1}^{phys} + Δψ`
- **Neural CDE (Scenario 4)** learns corrections using a short history to capture temporal dependencies.

## Installation

### Requirements

- Python 3.10+
- pip (or conda)

### Create environment (pip)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-prod.txt
pip install -r requirements-dev.txt
```

### (Optional) Install PyTorch

CPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA (choose the version you need):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Run the app

### Streamlit UI (main)

```bash
streamlit run src/rl_intelli_irrig_streamlit_config.py
```

## Run from Python

### Scenario 1 simulation

```python
from src.utils_physical_model import simulate_scenario1

episode = simulate_scenario1(
    season_length=120,
    rule_fn="rule_seuil_unique",
    rule_kwargs={"threshold_cbar": 80.0, "dose_mm": 15.0},
)
```

### Train a PPO agent (Scenario 2 baseline)

```python
from src.utils_env_gymnasium import IrrigationEnvPhysical
from stable_baselines3 import PPO

env = IrrigationEnvPhysical()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("models/ppo_irrigation")
```

## Project layout (src only)

```
src/
├── rl_intelli_irrig_streamlit_config.py  # Streamlit UI (training + evaluation)
├── utils_env_gymnasium.py                # RL environment (scenario 2)
├── utils_env_modeles.py                  # Hybrid environment (scenarios 3-4)
├── utils_neuro_ode.py                    # Neural ODE (scenario 3)
├── utils_neuro_ode_cont.py               # Neural ODE continuous (scenario 3b)
├── utils_neuro_cde.py                    # Neural CDE (scenario 4)
├── utils_physical_model.py               # Physical model + rules (scenario 1)
├── utils_physics_config.py               # Default configs
├── utils_plot.py                         # Plotting helpers
└── utils_ppo_training.py                 # PPO utilities
```

## Tests

```bash
pytest tests/
```

## License

TBD.
