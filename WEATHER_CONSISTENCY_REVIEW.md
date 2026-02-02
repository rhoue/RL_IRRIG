# Weather Generation Consistency Review

## Summary
This document reviews weather generation consistency across scenarios and documents when the random seed from the sidebar is used.

## Key Findings

### 1. Weather Generation Methods

There are **two weather generation methods**:

#### Method 1: Shared Generator (`utils_weather.generate_weather`)
- **Used by**: Scenarios 1, 2, 3, 4, 5, 6 when `weather_params` is provided (even if empty dict `{}`)
- **Characteristics**:
  - ET0: Sinusoidal seasonal variation + Gaussian noise
  - Rain: Phase-based probabilities (early 25%, mid 15%, late 20%)
  - Kc: Fixed day-based phases (t < 20, 20-50, 50-90, > 90)
- **Purpose**: Ensures **consistent weather patterns** across scenarios for fair comparison

#### Method 2: Internal Fallback (in `utils_env_modeles.py`)
- **Used by**: Scenarios 3-6 when `weather_params=None`
- **Characteristics**:
  - ET0: Normal distribution (mean=4.0, std=0.8) - **NO seasonal variation**
  - Rain: Gamma distribution (shape=0.8, scale=3.0)
  - Kc: Percentage-based phases (15%/25%/40%/20%)
- **Purpose**: Fallback method when shared generator is not explicitly requested

### 2. Environment Implementations

#### `utils_env_gymnasium.py` (Scenario 2)
- **Always uses** shared generator (`utils_weather.generate_weather`)
- Weather is generated **once at initialization**
- `reset()` does **NOT** regenerate weather (only resets state variables)
- To get new weather: create a new environment instance

#### `utils_env_modeles.py` (Scenarios 3-6)
- **Uses shared generator** when `weather_params is not None` (including `{}`)
- **Uses fallback** when `weather_params is None`
- Weather is regenerated **on each `reset()` call**
- If `seed` is provided to `reset()`, weather is regenerated with that seed

### 3. Seed Usage

The **random seed from sidebar** (`st.session_state.seed`) is used in:

1. **Scenario 1** (`simulate_scenario1`): 
   - Passed to `generate_weather()` for weather generation
   - ✅ Uses shared generator with `weather_params`

2. **Scenario 2** (`make_env`):
   - Passed to environment initialization
   - Weather generated once at initialization
   - ✅ Uses shared generator with `weather_params`

3. **Scenarios 3-6**:
   - Passed to environment initialization (`seed` parameter)
   - Also used for pretraining neural models
   - **⚠️ Fixed**: Line 1572 now uses `st.session_state.get("seed", 123)` instead of hardcoded `seed=123`

### 4. Consistency Requirements

For **fair comparison across scenarios**, ensure:

1. ✅ **Same seed**: Use the same seed value from sidebar
2. ✅ **Same weather_params**: Pass `weather_params={}` (or same configured params) to all scenarios
3. ✅ **Use shared generator**: Pass `weather_params` (even if empty) to use shared generator

### 5. Code Locations

| File | Function/Class | Weather Generation |
|------|---------------|-------------------|
| `utils_weather.py` | `generate_weather()` | Shared generator (sinusoidal ET0, phase-based rain) |
| `utils_env_gymnasium.py` | `IrrigationEnvPhysical.__init__()` | Always uses shared generator |
| `utils_env_modeles.py` | `IrrigationEnvPhysical._generate_weather()` | Shared if `weather_params is not None`, else fallback |
| `utils_physical_model.py` | `simulate_scenario1()` | Always uses shared generator |

### 6. Recommendations

1. **Always pass `weather_params={}`** when creating environments to ensure consistency
2. **Use same seed** when comparing scenarios
3. **Be aware** that `utils_env_gymnasium.py` doesn't regenerate weather on reset, while `utils_env_modeles.py` does
4. **For reproducibility**: Set seed in environment initialization, not just in reset

### 7. Fixed Issues

- ✅ Fixed hardcoded `seed=123` in Scenario 3 pretraining (line 1572) to use `st.session_state.get("seed", 123)`
- ✅ Added comments explaining weather generation differences
- ✅ Documented seed usage in reset() methods

## Testing Consistency

To verify weather consistency across scenarios:

```python
# Scenario 1
result1 = simulate_scenario1(T=120, seed=42, weather_params={})

# Scenario 2
env2 = IrrigationEnvPhysical(season_length=120, seed=42, weather_params={})

# Scenario 3
env3 = IrrigationEnvPhysical(season_length=120, seed=42, weather_params={})
```

All should generate **identical weather sequences** (rain, et0, Kc) for the same seed when using `weather_params={}`.








