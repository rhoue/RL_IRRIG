# EIPEx - Efficient Irrigation Policy Explorer

EIPEx is an operational framework for exploring efficient irrigation policies under uncertainty.
It combines reinforcement learning, physical modeling, and policy-oriented evaluation to support
governance of water-efficient use beyond a simple technology demonstrator.

Research prototype for irrigation control with reinforcement learning (RL).
It combines:

- a physical soil-water balance model,
- PPO-based policies,
- hybrid residual world models (Neural ODE variants),
- ERA5-Land based validation workflows.

## Implemented scenarios (main Streamlit app)

1. **Scenario 1 - Physical model + heuristic rules**
2. **Scenario 2 - PPO on physical model**
3. **Scenario 3 - PPO + residual Neural ODE**
4. **Scenario 3b - PPO + continuous Neural ODE**

The app also includes:

- per-scenario evaluation and visualization,
- cross-scenario comparison charts and metrics,
- advanced ERA5-Land validation,
- robustness evaluation over multiple ERA5 files / soil classes.

## Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements-prod.txt
pip install -r requirements-dev.txt
```

If `torch` installation fails or you want a specific build, install it manually:

```bash
# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# or CUDA (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) Run the Streamlit app (main UI)

From repository root:

```bash
streamlit run src/rl_intelli_irrig_streamlit_config.py
```

### 4) (Optional) Run the Gradio UI

```bash
python src/rl_intelli_irrig_gradio.py
```

### 5) Run the pure Python CLI (`main.py`)

Run one scenario:

```bash
# Scenario 1 (rule-based baseline)
python main.py run --scenario scenario1

# Scenario 2 (requires a trained PPO model)
python main.py run --scenario scenario2 --model-path models/scenario2_ppo.zip

# Scenario 3 (requires PPO + residual model)
python main.py run --scenario scenario3 \
  --model-path models/scenario3_ppo.zip \
  --residual-path models/scenario3_residual.pt

# Scenario 3b (continuous Neural ODE residual)
python main.py run --scenario scenario3b \
  --model-path models/scenario3b_ppo.zip \
  --residual-path models/scenario3b_residual.pt
```

Compare saved scenario runs:

```bash
python main.py compare --scenarios scenario1 scenario2 scenario3
```

Useful options:

- `--season-length`
- `--seed`
- `--max-irrigation`
- `--output-dir` / `--input-dir`
- `--era5-path` and `--era5-freq` (for ERA5-Land weather input)

CLI outputs:

- `scenarioX.npz` + `scenarioX_metrics.json` for each run
- `comparison_metrics.csv` and comparison figures for `compare`

## ERA5-Land data usage

Default example files are expected in `data/`, e.g.:

- `data/era5_land_fr_spring2024_all_nc3.nc`
- `data/era5_land_fr_spring2025_all_nc3.nc`

In the app, select ERA5-Land weather source and provide the file path.
The loader resolves relative paths robustly from repository root.

## Troubleshooting

### ERA5-Land cannot be opened

If you see engine errors (`netcdf4`, `h5netcdf`) or file-not-found:

1. Check file exists under `data/`.
2. Try installing xarray netCDF backends:

```bash
pip install xarray netcdf4 h5netcdf scipy
```

### Streamlit command not found

Install streamlit in your active environment:

```bash
pip install streamlit
```

### PPO / gym imports fail

```bash
pip install gymnasium stable-baselines3
```

## Minimal Python usage

```python
from src.utils_physical_model import simulate_scenario1

episode = simulate_scenario1(
    season_length=120,
    rule_fn="rule_seuil_unique",
    rule_kwargs={"threshold_cbar": 80.0, "dose_mm": 15.0},
)
```

## Tests

```bash
pytest tests/
```

## Citation

```bibtex
@software{rhoue_rl_intelligent_irrigation,
  author = {Raymond Houe Ngouna},
  title = {RL Intelligent Irrigation},
  year = {2026},
  url = {https://github.com/rhoue/RL_IRRIG}
}
```

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
