"""
Gradio native interface for the intelligent irrigation app.

This version runs alongside the Streamlit UI without modifying existing files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.gradio_ui_config_sections import (
    build_environment_config,
    build_goal_programming_config,
    build_mlp_policy_config,
    build_ppo_training_section,
    build_soil_and_tension_config,
    build_weather_config,
)
from src.gradio_ui_config_sections.goal_programming_config import build_goal_spec
from src.gradio_ui_doc import render_doc, render_doc_from_module, render_doc_markdown
from src.utils_physical_model import (
    PhysicalBucket,
    rule_seuil_unique,
    rule_bande_confort,
    rule_proportionnelle,
    simulate_scenario1,
    make_env,
    evaluate_episode,
)
from src.utils_physics_config import (
    get_rule_bande_confort_config,
    get_rule_proportionnelle_config,
    get_rule_seuil_unique_config,
    get_rule_seuil_unique_ranges,
)
from src.utils_plot import configure_matplotlib, plot_scenario1, plot_episode_rollout
from src.utils_ppo_training import create_ppo_callbacks
from src.utils_neuro_ode import pretrain_residual_ode, train_ppo_hybrid_ode, TORCH_AVAILABLE as TORCH_AVAILABLE_ODE
from src.utils_neuro_ode_cont import (
    pretrain_continuous_residual_ode,
    train_ppo_hybrid_ode_cont,
    TORCH_AVAILABLE as TORCH_AVAILABLE_ODE_CONT,
)
from src.utils_neuro_cde import pretrain_residual_cde, train_ppo_hybrid_cde, TORCH_AVAILABLE as TORCH_AVAILABLE_CDE
from src.utils_patch_tst import pretrain_patchtst_features, train_ppo_with_patchtst, TORCH_AVAILABLE as TORCH_AVAILABLE_PATCHTST
from src.utils_world_model import (
    train_world_model_phase1,
    train_ppo_with_world_model_phase1,
    train_world_model_phase2,
    train_ppo_with_world_model_phase2,
    train_ppo_with_world_model_phase3,
    TORCH_AVAILABLE as TORCH_AVAILABLE_WM,
)

PPO_AVAILABLE = False
PPO = None
DummyVecEnv = None
try:
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    PPO_AVAILABLE = True
except Exception:
    PPO_AVAILABLE = False


configure_matplotlib()


def _build_rule_inputs(max_irrigation: float, language: str) -> Dict[str, Any]:
    t = {
        "fr": {
            "rule_label": "Type de regle d'irrigation",
            "threshold": "Seuil unique",
            "band": "Bande de confort",
            "prop": "Proportionnelle",
            "threshold_label": "Seuil de tension (cbar)",
            "dose_label": "Dose d'irrigation (mm)",
            "rain_label": "Seuil pluie prevue (mm)",
            "reduce_label": "Facteur de reduction si pluie",
            "psi_low": "psi bas (cbar)",
            "psi_high": "psi haut (cbar)",
            "psi_target": "psi cible (cbar)",
            "k_i": "Coefficient k_I",
        },
        "en": {
            "rule_label": "Irrigation rule type",
            "threshold": "Single threshold",
            "band": "Comfort band",
            "prop": "Proportional",
            "threshold_label": "Tension threshold (cbar)",
            "dose_label": "Irrigation dose (mm)",
            "rain_label": "Forecast rain threshold (mm)",
            "reduce_label": "Reduction factor if rain",
            "psi_low": "psi low (cbar)",
            "psi_high": "psi high (cbar)",
            "psi_target": "psi target (cbar)",
            "k_i": "Coefficient k_I",
        },
    }[language]

    rule_type = gr.Dropdown(
        choices=["threshold", "band", "prop"],
        value="threshold",
        label=t["rule_label"],
    )

    rule_defaults = get_rule_seuil_unique_config()
    rule_ranges = get_rule_seuil_unique_ranges()
    with gr.Row(visible=True) as threshold_row:
        threshold_cbar = gr.Number(
            value=rule_defaults["threshold_cbar"],
            label=t["threshold_label"],
        )
        dose_mm = gr.Number(
            value=rule_defaults["dose_mm"],
            label=t["dose_label"],
        )
        rain_threshold_mm = gr.Number(
            value=rule_defaults["rain_threshold_mm"],
            label=t["rain_label"],
        )
        reduce_factor = gr.Slider(
            rule_ranges["reduce_factor"]["min"],
            rule_ranges["reduce_factor"]["max"],
            value=rule_defaults["reduce_factor"],
            step=rule_ranges["reduce_factor"]["step"],
            label=t["reduce_label"],
        )

    band_defaults = get_rule_bande_confort_config()
    with gr.Row(visible=False) as band_row:
        psi_low = gr.Number(value=band_defaults["psi_low"], label=t["psi_low"])
        psi_high = gr.Number(value=band_defaults["psi_high"], label=t["psi_high"])
        dose_mm_band = gr.Number(
            value=band_defaults["dose_mm"],
            label=t["dose_label"],
        )

    prop_defaults = get_rule_proportionnelle_config()
    with gr.Row(visible=False) as prop_row:
        psi_target = gr.Number(value=prop_defaults["psi_target"], label=t["psi_target"])
        k_I = gr.Number(
            value=prop_defaults["k_I"],
            label=t["k_i"],
            precision=2,
        )

    def _toggle_rule(rule_value: str):
        return (
            gr.update(visible=rule_value == "threshold"),
            gr.update(visible=rule_value == "band"),
            gr.update(visible=rule_value == "prop"),
        )

    rule_type.change(_toggle_rule, inputs=rule_type, outputs=[threshold_row, band_row, prop_row])

    return {
        "rule_type": rule_type,
        "threshold_cbar": threshold_cbar,
        "dose_mm": dose_mm,
        "rain_threshold_mm": rain_threshold_mm,
        "reduce_factor": reduce_factor,
        "psi_low": psi_low,
        "psi_high": psi_high,
        "dose_mm_band": dose_mm_band,
        "psi_target": psi_target,
        "k_I": k_I,
    }


def _run_scenario1(
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    rule_type: str,
    threshold_cbar: float,
    dose_mm: float,
    rain_threshold_mm: float,
    reduce_factor: float,
    psi_low: float,
    psi_high: float,
    dose_mm_band: float,
    psi_target: float,
    k_I: float,
    language: str,
) -> Tuple[Any, str, Dict[str, Any]]:
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }

    if rule_type == "threshold":
        rule_fn = rule_seuil_unique
        rule_kwargs = {
            "threshold_cbar": threshold_cbar,
            "dose_mm": dose_mm,
            "rain_threshold_mm": rain_threshold_mm,
            "reduce_factor": reduce_factor,
        }
    elif rule_type == "band":
        rule_fn = rule_bande_confort
        rule_kwargs = {"psi_low": psi_low, "psi_high": psi_high, "dose_mm": dose_mm_band}
    else:
        rule_fn = rule_proportionnelle
        rule_kwargs = {"psi_target": psi_target, "k_I": k_I}

    soil = PhysicalBucket(**soil_params)
    sim_result = simulate_scenario1(
        T=int(season_length),
        seed=int(seed),
        I_max=float(max_irrigation),
        soil=soil,
        rule_fn=rule_fn,
        rule_kwargs=rule_kwargs,
        weather_params=weather_params,
    )
    fig = plot_scenario1(sim_result, language=language)
    metrics = {
        "irrigation_total_mm": float(sim_result["I"].sum()),
        "rain_total_mm": float(sim_result["rain"].sum()),
        "drainage_total_mm": float(sim_result["D"].sum()),
        "stress_days_pct": float(100.0 * (sim_result["psi"] > psi_ET_crit).mean()),
    }
    metrics_md = "\n".join([f"- **{k}**: {v:.2f}" for k, v in metrics.items()])
    return fig, metrics_md, sim_result


def _build_policy_kwargs(policy_type: str, hidden1: float, hidden2: float) -> Optional[Dict[str, Any]]:
    if policy_type != "MlpPolicy":
        return None
    h1 = int(hidden1) if hidden1 else 0
    h2 = int(hidden2) if hidden2 else 0
    if h1 <= 0:
        return None
    net_arch = [h1]
    if h2 > 0:
        net_arch.append(h2)
    return {"net_arch": net_arch}


def _start_training_status() -> str:
    return "⏳ Entraînement en cours... / Training in progress..."


def _start_eval_status() -> str:
    return "⏳ Évaluation en cours... / Evaluation in progress..."


def _train_scenario2(
    total_timesteps: int,
    policy_type: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    hidden1: float,
    hidden2: float,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    weather_source: str,
    era5_path: str,
    era5_freq: str,
    goal_spec: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Any, str]:
    if not PPO_AVAILABLE or PPO is None or DummyVecEnv is None:
        return None, "⚠️ PPO indisponible (installer gymnasium + stable-baselines3)."
    progress(0.0, desc="Démarrage de l'entraînement PPO…")

    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    data_source = "era5_land" if weather_source == "era5_land" and era5_path else "synthetic"
    era5_land_cfg = (
        {"use_era5_land": True, "data_path": era5_path, "resample_freq": era5_freq or "1D"}
        if data_source == "era5_land"
        else None
    )

    ppo_kwargs = {
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_range": float(clip_range),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "verbose": 0,
    }
    policy_kwargs = _build_policy_kwargs(policy_type, hidden1, hidden2)

    base_env_factory = make_env(
        seed=int(seed),
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        soil_params=soil_params,
        weather_params=weather_params,
        goal_spec=goal_spec or None,
        weather_shift_cfg=None,
        data_source=data_source,
        data_path=era5_path if data_source == "era5_land" else None,
        era5_land_cfg=era5_land_cfg,
    )

    vec_env = DummyVecEnv([base_env_factory])
    model = PPO(
        policy=policy_type,
        env=vec_env,
        seed=int(seed),
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
    )
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    callbacks, _metrics = create_ppo_callbacks(_progress_cb, total_timesteps=int(total_timesteps))
    model.learn(total_timesteps=int(total_timesteps), callback=callbacks)
    progress(1.0, desc="Entraînement terminé.")
    return model, "✅ Entraînement terminé."


def _eval_scenario2(
    model: Any,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    weather_source: str,
    era5_path: str,
    era5_freq: str,
    language: str,
) -> Tuple[Any, str]:
    if model is None:
        return None, "⚠️ Aucun modèle entraîné."

    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    data_source = "era5_land" if weather_source == "era5_land" and era5_path else "synthetic"
    era5_land_cfg = (
        {"use_era5_land": True, "data_path": era5_path, "resample_freq": era5_freq or "1D"}
        if data_source == "era5_land"
        else None
    )

    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        data_source=data_source,
        data_path=era5_path if data_source == "era5_land" else None,
        era5_land_cfg=era5_land_cfg,
    )
    fig = plot_episode_rollout(rollout, language=language)
    metrics_md = "\n".join(
        [
            f"- **irrigation_total_mm**: {float(rollout['I'].sum()):.2f}",
            f"- **rain_total_mm**: {float(rollout['R'].sum()):.2f}",
            f"- **drainage_total_mm**: {float(rollout['D'].sum()):.2f}",
        ]
    )
    return fig, metrics_md


def _rollout_scenario2(
    model: Any,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    weather_source: str = "synthetic",
    era5_path: str = "",
    era5_freq: str = "1D",
):
    if model is None:
        return None, "⚠️ Aucun modèle entraîné."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    data_source = "era5_land" if weather_source == "era5_land" and era5_path else "synthetic"
    era5_land_cfg = (
        {"use_era5_land": True, "data_path": era5_path, "resample_freq": era5_freq or "1D"}
        if data_source == "era5_land"
        else None
    )
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        data_source=data_source,
        data_path=era5_path if data_source == "era5_land" else None,
        era5_land_cfg=era5_land_cfg,
    )
    return rollout, "✅ Scenario 2 évalué."


def _rollout_scenario3(
    model: Any,
    residual: Any,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
):
    if model is None or residual is None:
        return None, "⚠️ Modèle manquant."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        residual_ode=residual,
    )
    return rollout, "✅ Scenario 3 évalué."


def _rollout_scenario4(
    model: Any,
    residual: Any,
    seq_len: int,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
):
    if model is None or residual is None:
        return None, "⚠️ Modèle manquant."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        residual_cde=residual,
        seq_len_cde=int(seq_len),
    )
    return rollout, "✅ Scenario 4 évalué."


def _rollout_scenario5(
    model: Any,
    patchtst_model: Any,
    seq_len: int,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
):
    if model is None or patchtst_model is None:
        return None, "⚠️ Modèle manquant."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        patchtst_model=patchtst_model,
        seq_len_patchtst=int(seq_len),
    )
    return rollout, "✅ Scenario 5 évalué."


def _rollout_scenario6(
    model: Any,
    enc: Any,
    trans: Any,
    dec: Any,
    seq_len: int,
    alpha: float,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
):
    if model is None or enc is None or trans is None or dec is None:
        return None, "⚠️ Modèles requis."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        wm_encoder=enc,
        wm_transition=trans,
        wm_decoder=dec,
        wm_seq_len=int(seq_len),
        wm_hybrid_alpha=float(alpha),
    )
    return rollout, "✅ Scenario 6 évalué."


def _metrics_from_series(psi: np.ndarray, irrigation: np.ndarray, rain: np.ndarray, drainage: np.ndarray) -> Dict[str, float]:
    comfort = (psi >= 20.0) & (psi <= 60.0)
    return {
        "total_irrigation": float(np.nansum(irrigation)),
        "total_rain": float(np.nansum(rain)),
        "total_drainage": float(np.nansum(drainage)),
        "mean_psi": float(np.nanmean(psi)),
        "comfort_pct": float(100.0 * np.nanmean(comfort)),
    }


def _build_comparison_table(
    scenario1_result: Optional[Dict[str, Any]],
    scenario2_rollout: Optional[Dict[str, Any]],
    scenario3_rollout: Optional[Dict[str, Any]],
    scenario3b_rollout: Optional[Dict[str, Any]],
    scenario4_rollout: Optional[Dict[str, Any]],
    scenario5_rollout: Optional[Dict[str, Any]],
    scenario6_rollout: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    if scenario1_result:
        rows.append(
            {"scenario": "Scenario 1", **_metrics_from_series(
                np.array(scenario1_result["psi"]),
                np.array(scenario1_result["I"]),
                np.array(scenario1_result["rain"]),
                np.array(scenario1_result["D"]),
            )}
        )
    mapping = [
        ("Scenario 2", scenario2_rollout),
        ("Scenario 3", scenario3_rollout),
        ("Scenario 3b", scenario3b_rollout),
        ("Scenario 4", scenario4_rollout),
        ("Scenario 5", scenario5_rollout),
        ("Scenario 6", scenario6_rollout),
    ]
    for name, rollout in mapping:
        if rollout:
            rows.append(
                {"scenario": name, **_metrics_from_series(
                    np.array(rollout["psi"]),
                    np.array(rollout["I"]),
                    np.array(rollout.get("R", rollout.get("rain", []))),
                    np.array(rollout["D"]),
                )}
            )
    return pd.DataFrame(rows)


def _plot_rollout_summary(df: pd.DataFrame) -> Any:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["scenario"], df["comfort_pct"], color="tab:green", alpha=0.7)
    ax.set_ylabel("Comfort %")
    ax.set_title("Comfort zone share")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    return fig


def _render_visualization(
    choice: str,
    scenario1_result: Optional[Dict[str, Any]],
    scenario2_rollout: Optional[Dict[str, Any]],
    scenario3_rollout: Optional[Dict[str, Any]],
    scenario3b_rollout: Optional[Dict[str, Any]],
    scenario4_rollout: Optional[Dict[str, Any]],
    scenario5_rollout: Optional[Dict[str, Any]],
    scenario6_rollout: Optional[Dict[str, Any]],
    language: str,
) -> Tuple[Any, str]:
    if choice == "Scenario 1" and scenario1_result:
        fig = plot_scenario1(scenario1_result, language=language)
        return fig, "✅ Plot Scenario 1."
    mapping = {
        "Scenario 2": scenario2_rollout,
        "Scenario 3": scenario3_rollout,
        "Scenario 3b": scenario3b_rollout,
        "Scenario 4": scenario4_rollout,
        "Scenario 5": scenario5_rollout,
        "Scenario 6": scenario6_rollout,
    }
    rollout = mapping.get(choice)
    if rollout:
        fig = plot_episode_rollout(rollout, language=language)
        return fig, "✅ Plot disponible."
    return None, "⚠️ Aucun résultat disponible."


def _doc_loader(lang: str, cache: Dict[str, str], module_path: str, fn_name: str, *args) -> Tuple[str, Dict[str, str]]:
    cache = cache or {}
    key = f"{module_path}:{fn_name}:{lang}:{'|'.join([str(a) for a in args])}"
    if key in cache:
        return cache[key], cache
    md = render_doc_markdown(module_path, fn_name, *args)
    cache[key] = md
    return md, cache


def _pretrain_ode(
    N_traj: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    max_irrigation: float,
    season_length: int,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    progress: gr.Progress = gr.Progress(),
):
    if not TORCH_AVAILABLE_ODE:
        return None, "⚠️ PyTorch indisponible."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    soil = PhysicalBucket(**soil_params)
    def _progress_cb(epoch: int, total: int, avg_loss: float) -> None:
        if total > 0:
            progress(epoch / total, desc=f"epoch {epoch}/{total} loss {avg_loss:.4f}")
    model, loss = pretrain_residual_ode(
        soil,
        max_irrigation=float(max_irrigation),
        T=int(season_length),
        N_traj=int(N_traj),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        progress_callback=_progress_cb,
    )
    return model, f"✅ Pré-entraînement terminé (loss={loss:.4f})."


def _train_scenario3(
    residual_model: Any,
    policy_type: str,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    goal_spec: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
):
    if residual_model is None:
        return None, "⚠️ Aucun modèle résiduel pré-entraîné."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    ppo_kwargs = {
        "policy": policy_type,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_range": float(clip_range),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
    }
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    model, _metrics = train_ppo_hybrid_ode(
        residual_model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        total_timesteps=int(total_timesteps),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        ppo_kwargs=ppo_kwargs,
        goal_spec=goal_spec or None,
        progress_callback=_progress_cb,
    )
    return model, "✅ Entraînement PPO hybride terminé."


def _eval_scenario3(
    model: Any,
    residual_model: Any,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    weather_source: str,
    era5_path: str,
    era5_freq: str,
    language: str,
):
    if model is None or residual_model is None:
        return None, "⚠️ Modèle manquant."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        residual_ode=residual_model,
    )
    fig = plot_episode_rollout(rollout, language=language)
    return fig, "✅ Évaluation terminée."


def _pretrain_ode_cont(
    N_traj: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    max_irrigation: float,
    season_length: int,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    progress: gr.Progress = gr.Progress(),
):
    if not TORCH_AVAILABLE_ODE_CONT:
        return None, "⚠️ PyTorch indisponible."
    soil = PhysicalBucket(
        Z_r=Z_r,
        theta_s=theta_s,
        theta_fc=theta_fc,
        theta_wp=theta_wp,
        psi_sat=psi_sat,
        psi_fc=psi_fc,
        psi_wp=psi_wp,
        k_d=k_d,
        eta_I=eta_I,
        psi_ET_crit=psi_ET_crit,
    )
    def _progress_cb(epoch: int, total: int, avg_loss: float) -> None:
        if total > 0:
            progress(epoch / total, desc=f"epoch {epoch}/{total} loss {avg_loss:.4f}")
    model, loss = pretrain_continuous_residual_ode(
        soil,
        max_irrigation=float(max_irrigation),
        T=int(season_length),
        N_traj=int(N_traj),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        progress_callback=_progress_cb,
    )
    return model, f"✅ Pré-entraînement terminé (loss={loss:.4f})."


def _train_scenario3b(
    residual_model: Any,
    policy_type: str,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    goal_spec: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
):
    if residual_model is None:
        return None, "⚠️ Aucun modèle résiduel pré-entraîné."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    ppo_kwargs = {
        "policy": policy_type,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_range": float(clip_range),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
    }
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    model, _metrics = train_ppo_hybrid_ode_cont(
        residual_model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        total_timesteps=int(total_timesteps),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        ppo_kwargs=ppo_kwargs,
        goal_spec=goal_spec or None,
        progress_callback=_progress_cb,
    )
    return model, "✅ Entraînement PPO hybride continu terminé."


def _eval_scenario3b(
    model: Any,
    residual_model: Any,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    language: str,
):
    if model is None or residual_model is None:
        return None, "⚠️ Modèle manquant."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        residual_ode=residual_model,
    )
    fig = plot_episode_rollout(rollout, language=language)
    return fig, "✅ Évaluation terminée."


def _pretrain_cde(
    N_traj: int,
    seq_len: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    max_irrigation: float,
    season_length: int,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    progress: gr.Progress = gr.Progress(),
):
    if not TORCH_AVAILABLE_CDE:
        return None, "⚠️ PyTorch indisponible."
    soil = PhysicalBucket(
        Z_r=Z_r,
        theta_s=theta_s,
        theta_fc=theta_fc,
        theta_wp=theta_wp,
        psi_sat=psi_sat,
        psi_fc=psi_fc,
        psi_wp=psi_wp,
        k_d=k_d,
        eta_I=eta_I,
        psi_ET_crit=psi_ET_crit,
    )
    def _progress_cb(epoch: int, total: int, avg_loss: float) -> None:
        if total > 0:
            progress(epoch / total, desc=f"epoch {epoch}/{total} loss {avg_loss:.4f}")
    model, loss = pretrain_residual_cde(
        soil,
        max_irrigation=float(max_irrigation),
        T=int(season_length),
        N_traj=int(N_traj),
        seq_len=int(seq_len),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        progress_callback=_progress_cb,
    )
    return model, f"✅ Pré-entraînement terminé (loss={loss:.4f})."


def _train_scenario4(
    residual_model: Any,
    seq_len: int,
    policy_type: str,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    goal_spec: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
):
    if residual_model is None:
        return None, "⚠️ Aucun modèle résiduel pré-entraîné."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    ppo_kwargs = {
        "policy": policy_type,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_range": float(clip_range),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
    }
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    model, _metrics = train_ppo_hybrid_cde(
        residual_model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seq_len_cde=int(seq_len),
        total_timesteps=int(total_timesteps),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        ppo_kwargs=ppo_kwargs,
        goal_spec=goal_spec or None,
        progress_callback=_progress_cb,
    )
    return model, "✅ Entraînement PPO CDE terminé."


def _eval_scenario4(
    model: Any,
    residual_model: Any,
    seq_len: int,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    language: str,
):
    if model is None or residual_model is None:
        return None, "⚠️ Modèle manquant."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        residual_cde=residual_model,
        seq_len_cde=int(seq_len),
    )
    fig = plot_episode_rollout(rollout, language=language)
    return fig, "✅ Évaluation terminée."


def _pretrain_patchtst(
    N_traj: int,
    seq_len: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    feature_dim: int,
    task: str,
    max_irrigation: float,
    season_length: int,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    progress: gr.Progress = gr.Progress(),
):
    if not TORCH_AVAILABLE_PATCHTST:
        return None, "⚠️ PyTorch indisponible."
    soil = PhysicalBucket(
        Z_r=Z_r,
        theta_s=theta_s,
        theta_fc=theta_fc,
        theta_wp=theta_wp,
        psi_sat=psi_sat,
        psi_fc=psi_fc,
        psi_wp=psi_wp,
        k_d=k_d,
        eta_I=eta_I,
        psi_ET_crit=psi_ET_crit,
    )
    def _progress_cb(progress_value: float, message: str) -> None:
        progress(progress_value, desc=message)
    model, loss = pretrain_patchtst_features(
        soil=soil,
        max_irrigation=float(max_irrigation),
        T=int(season_length),
        N_traj=int(N_traj),
        seq_len=int(seq_len),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        feature_dim=int(feature_dim),
        task=task,
        progress_callback=_progress_cb,
    )
    return model, f"✅ Pré-entraînement PatchTST terminé (loss={loss:.4f})."


def _train_scenario5(
    patchtst_model: Any,
    seq_len: int,
    policy_type: str,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    goal_spec: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
):
    if patchtst_model is None:
        return None, "⚠️ Aucun modèle PatchTST pré-entraîné."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    ppo_kwargs = {
        "policy": policy_type,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_range": float(clip_range),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
    }
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    model, _metrics = train_ppo_with_patchtst(
        soil_params=soil_params,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        total_timesteps=int(total_timesteps),
        seed=int(seed),
        weather_params=weather_params,
        patchtst_model=patchtst_model,
        seq_len_patchtst=int(seq_len),
        goal_spec=goal_spec or None,
        progress_callback=_progress_cb,
        **ppo_kwargs,
    )
    return model, "✅ Entraînement PPO PatchTST terminé."


def _eval_scenario5(
    model: Any,
    patchtst_model: Any,
    seq_len: int,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    language: str,
):
    if model is None or patchtst_model is None:
        return None, "⚠️ Modèle manquant."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        patchtst_model=patchtst_model,
        seq_len_patchtst=int(seq_len),
    )
    fig = plot_episode_rollout(rollout, language=language)
    return fig, "✅ Évaluation terminée."


def _train_wm_phase1(
    patchtst_model: Any,
    N_traj: int,
    seq_len: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    progress: gr.Progress = gr.Progress(),
):
    if patchtst_model is None:
        return None, "⚠️ PatchTST requis."
    soil = PhysicalBucket(
        Z_r=Z_r,
        theta_s=theta_s,
        theta_fc=theta_fc,
        theta_wp=theta_wp,
        psi_sat=psi_sat,
        psi_fc=psi_fc,
        psi_wp=psi_wp,
        k_d=k_d,
        eta_I=eta_I,
        psi_ET_crit=psi_ET_crit,
    )
    def _progress_cb(progress_value: float, message: str) -> None:
        progress(progress_value, desc=message)
    transition, loss = train_world_model_phase1(
        encoder_model=patchtst_model,
        soil=soil,
        max_irrigation=float(max_irrigation),
        N_traj=int(N_traj),
        seq_len=int(seq_len),
        latent_dim=int(latent_dim),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        progress_callback=_progress_cb,
    )
    return transition, f"✅ World Model Phase 1 entraîné (loss={loss:.4f})."


def _train_wm_phase2(
    patchtst_model: Any,
    N_traj: int,
    seq_len: int,
    cde_seq_len: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    progress: gr.Progress = gr.Progress(),
):
    if patchtst_model is None:
        return None, None, "⚠️ PatchTST requis."
    soil = PhysicalBucket(
        Z_r=Z_r,
        theta_s=theta_s,
        theta_fc=theta_fc,
        theta_wp=theta_wp,
        psi_sat=psi_sat,
        psi_fc=psi_fc,
        psi_wp=psi_wp,
        k_d=k_d,
        eta_I=eta_I,
        psi_ET_crit=psi_ET_crit,
    )
    def _progress_cb(progress_value: float, message: str) -> None:
        progress(progress_value, desc=message)
    transition, decoder, loss = train_world_model_phase2(
        encoder_model=patchtst_model,
        soil=soil,
        max_irrigation=float(max_irrigation),
        N_traj=int(N_traj),
        seq_len=int(seq_len),
        cde_seq_len=int(cde_seq_len),
        latent_dim=int(latent_dim),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        progress_callback=_progress_cb,
    )
    return transition, decoder, f"✅ World Model Phase 2 entraîné (loss={loss:.4f})."


def _train_ppo_wm_phase1(
    patchtst_model: Any,
    transition_model: Any,
    imagination_horizon: int,
    imagination_ratio: float,
    policy_type: str,
    total_timesteps: int,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    weather_source: str = "synthetic",
    era5_path: str = "",
    era5_freq: str = "1D",
    progress: gr.Progress = gr.Progress(),
):
    if patchtst_model is None or transition_model is None:
        return None, "⚠️ Modèles requis."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    # For now, external weather is only used in simpler scenarios; we keep the
    # signature to avoid breaking UI wiring and future-proof ERA5-Land support.
    _ = (weather_source, era5_path, era5_freq)
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    model, _metrics = train_ppo_with_world_model_phase1(
        encoder_model=patchtst_model,
        transition_model=transition_model,
        soil_params=soil_params,
        weather_params=weather_params,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        total_timesteps=int(total_timesteps),
        imagination_horizon=int(imagination_horizon),
        imagination_ratio=float(imagination_ratio),
        seed=int(seed),
        policy=policy_type,
        progress_callback=_progress_cb,
    )
    return model, "✅ PPO Phase 1 entraîné."


def _train_ppo_wm_phase2(
    patchtst_model: Any,
    transition_model: Any,
    decoder_model: Any,
    imagination_horizon: int,
    imagination_ratio: float,
    policy_type: str,
    total_timesteps: int,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    progress: gr.Progress = gr.Progress(),
):
    if patchtst_model is None or transition_model is None or decoder_model is None:
        return None, "⚠️ Modèles requis."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    model, _metrics = train_ppo_with_world_model_phase2(
        encoder_model=patchtst_model,
        transition_model=transition_model,
        decoder_model=decoder_model,
        soil_params=soil_params,
        weather_params=weather_params,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        total_timesteps=int(total_timesteps),
        imagination_horizon=int(imagination_horizon),
        imagination_ratio=float(imagination_ratio),
        seed=int(seed),
        policy=policy_type,
        progress_callback=_progress_cb,
    )
    return model, "✅ PPO Phase 2 entraîné."


def _train_ppo_wm_phase3(
    patchtst_model: Any,
    transition_model: Any,
    decoder_model: Any,
    seq_len: int,
    hybrid_alpha: float,
    policy_type: str,
    total_timesteps: int,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    progress: gr.Progress = gr.Progress(),
):
    if patchtst_model is None or transition_model is None or decoder_model is None:
        return None, "⚠️ Modèles requis."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    def _progress_cb(current: int, total: int) -> None:
        if total > 0:
            progress(current / total, desc=f"{current}/{total}")
    model, _metrics = train_ppo_with_world_model_phase3(
        encoder_model=patchtst_model,
        transition_model=transition_model,
        decoder_model=decoder_model,
        seq_len_wm=int(seq_len),
        hybrid_alpha=float(hybrid_alpha),
        soil_params=soil_params,
        weather_params=weather_params,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        total_timesteps=int(total_timesteps),
        seed=int(seed),
        policy=policy_type,
        progress_callback=_progress_cb,
    )
    return model, "✅ PPO Phase 3 entraîné."


def _eval_scenario6(
    model: Any,
    patchtst_model: Any,
    transition_model: Any,
    decoder_model: Any,
    seq_len: int,
    hybrid_alpha: float,
    season_length: int,
    max_irrigation: float,
    seed: int,
    Z_r: float,
    theta_s: float,
    theta_fc: float,
    theta_wp: float,
    psi_sat: float,
    psi_fc: float,
    psi_wp: float,
    k_d: float,
    eta_I: float,
    psi_ET_crit: float,
    et0_base: float,
    et0_amp: float,
    et0_noise: float,
    p_rain_early: float,
    p_rain_mid: float,
    p_rain_late: float,
    rain_min: float,
    rain_max: float,
    language: str,
):
    if model is None or patchtst_model is None or transition_model is None or decoder_model is None:
        return None, "⚠️ Modèles requis."
    soil_params = {
        "Z_r": Z_r,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "psi_sat": psi_sat,
        "psi_fc": psi_fc,
        "psi_wp": psi_wp,
        "k_d": k_d,
        "eta_I": eta_I,
        "psi_ET_crit": psi_ET_crit,
    }
    weather_params = {
        "et0_base": et0_base,
        "et0_amp": et0_amp,
        "et0_noise": et0_noise,
        "p_rain_early": p_rain_early,
        "p_rain_mid": p_rain_mid,
        "p_rain_late": p_rain_late,
        "rain_min": rain_min,
        "rain_max": rain_max,
    }
    data_source = "era5_land" if weather_source == "era5_land" and era5_path else "synthetic"
    era5_land_cfg = (
        {"use_era5_land": True, "data_path": era5_path, "resample_freq": era5_freq or "1D"}
        if data_source == "era5_land"
        else None
    )
    rollout = evaluate_episode(
        model,
        season_length=int(season_length),
        max_irrigation=float(max_irrigation),
        seed=int(seed),
        soil_params=soil_params,
        weather_params=weather_params,
        wm_encoder=patchtst_model,
        wm_transition=transition_model,
        wm_decoder=decoder_model,
        wm_seq_len=int(seq_len),
        wm_hybrid_alpha=float(hybrid_alpha),
        data_source=data_source,
        data_path=era5_path if data_source == "era5_land" else None,
        era5_land_cfg=era5_land_cfg,
    )
    fig = plot_episode_rollout(rollout, language=language)
    return fig, "✅ Évaluation terminée."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RL Intelligent Irrigation (Gradio)") as demo:
        gr.Markdown("# 💧 Irrigation intelligente / Smart irrigation (Gradio natif)")
        language = gr.Dropdown(
            choices=["fr", "en"],
            value="fr",
            label="Langue / Language",
        )
        goal_spec_state = gr.State({})
        scenario2_model_state = gr.State(None)
        scenario1_result_state = gr.State(None)
        scenario3_residual_state = gr.State(None)
        scenario3_model_state = gr.State(None)
        scenario3b_residual_state = gr.State(None)
        scenario3b_model_state = gr.State(None)
        scenario4_residual_state = gr.State(None)
        scenario4_model_state = gr.State(None)
        patchtst_model_state = gr.State(None)
        scenario5_model_state = gr.State(None)
        wm_phase1_transition_state = gr.State(None)
        wm_phase2_transition_state = gr.State(None)
        wm_phase2_decoder_state = gr.State(None)
        wm_phase1_ppo_state = gr.State(None)
        wm_phase2_ppo_state = gr.State(None)
        wm_phase3_ppo_state = gr.State(None)
        scenario2_rollout_state = gr.State(None)
        scenario3_rollout_state = gr.State(None)
        scenario3b_rollout_state = gr.State(None)
        scenario4_rollout_state = gr.State(None)
        scenario5_rollout_state = gr.State(None)
        scenario6_rollout_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Configuration")
                env_components = build_environment_config("fr")
                soil_components = build_soil_and_tension_config("fr")
                weather_components = build_weather_config("fr")
                goal_components = build_goal_programming_config("fr")
                ppo_components = build_ppo_training_section("fr")
                mlp_components = build_mlp_policy_config("MlpPolicy", "fr")
                if mlp_components:
                    def _toggle_mlp(policy_choice: str):
                        visible = policy_choice == "MlpPolicy"
                        return gr.update(visible=visible), gr.update(visible=visible)
                    ppo_components["policy_type"].change(
                        _toggle_mlp,
                        inputs=ppo_components["policy_type"],
                        outputs=[mlp_components["hidden_layer_1"], mlp_components["hidden_layer_2"]],
                    )
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📚 Documentation / Docs"):
                        with gr.Tabs():
                            with gr.TabItem("💧 Irrigation intelligente / Smart irrigation"):
                                doc_irrig_cache = gr.State({})
                                doc_irrig_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_irrigation_intelligente"),
                                    inputs=[language, doc_irrig_cache],
                                    outputs=[doc_irrig_md, doc_irrig_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_irrigation_intelligente"),
                                    inputs=[language, doc_irrig_cache],
                                    outputs=[doc_irrig_md, doc_irrig_cache],
                                )
                            with gr.TabItem("📊 Variables d'état / State variables"):
                                doc_vars_cache = gr.State({})
                                doc_vars_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_variables_etat"),
                                    inputs=[language, doc_vars_cache],
                                    outputs=[doc_vars_md, doc_vars_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_variables_etat"),
                                    inputs=[language, doc_vars_cache],
                                    outputs=[doc_vars_md, doc_vars_cache],
                                )
                            with gr.TabItem("🤖 Apprentissage RL / RL"):
                                doc_rl_cache = gr.State({})
                                doc_rl_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_apprentissage_renforcement"),
                                    inputs=[language, doc_rl_cache],
                                    outputs=[doc_rl_md, doc_rl_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_apprentissage_renforcement"),
                                    inputs=[language, doc_rl_cache],
                                    outputs=[doc_rl_md, doc_rl_cache],
                                )
                            with gr.TabItem("🎓 Scénario 2 / Scenario 2"):
                                doc_s2_cache = gr.State({})
                                doc_s2_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_scenario2"),
                                    inputs=[language, doc_s2_cache],
                                    outputs=[doc_s2_md, doc_s2_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_scenario2"),
                                    inputs=[language, doc_s2_cache],
                                    outputs=[doc_s2_md, doc_s2_cache],
                                )
                            with gr.TabItem("🧠 Neuro-ODE / Neural ODE"):
                                doc_ode_cache = gr.State({})
                                doc_ode_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_neural_ode"),
                                    inputs=[language, doc_ode_cache],
                                    outputs=[doc_ode_md, doc_ode_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_neural_ode"),
                                    inputs=[language, doc_ode_cache],
                                    outputs=[doc_ode_md, doc_ode_cache],
                                )
                            with gr.TabItem("🧠 ODE continu / Continuous ODE"):
                                doc_odec_cache = gr.State({})
                                doc_odec_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_neural_ode_cont"),
                                    inputs=[language, doc_odec_cache],
                                    outputs=[doc_odec_md, doc_odec_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_neural_ode_cont"),
                                    inputs=[language, doc_odec_cache],
                                    outputs=[doc_odec_md, doc_odec_cache],
                                )
                            with gr.TabItem("🌀 Neural CDE / CDE"):
                                doc_cde_cache = gr.State({})
                                doc_cde_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_neural_cde"),
                                    inputs=[language, doc_cde_cache],
                                    outputs=[doc_cde_md, doc_cde_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_neural_cde"),
                                    inputs=[language, doc_cde_cache],
                                    outputs=[doc_cde_md, doc_cde_cache],
                                )
                            with gr.TabItem("🔮 PatchTST / PatchTST"):
                                doc_patch_cache = gr.State({})
                                doc_patch_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_patchtst"),
                                    inputs=[language, doc_patch_cache],
                                    outputs=[doc_patch_md, doc_patch_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_patchtst"),
                                    inputs=[language, doc_patch_cache],
                                    outputs=[doc_patch_md, doc_patch_cache],
                                )
                            with gr.TabItem("🌍 World Model / Scénario 6"):
                                doc_wm_cache = gr.State({})
                                doc_wm_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_scenario6_world_model"),
                                    inputs=[language, doc_wm_cache],
                                    outputs=[doc_wm_md, doc_wm_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_scenario6_world_model"),
                                    inputs=[language, doc_wm_cache],
                                    outputs=[doc_wm_md, doc_wm_cache],
                                )
                            with gr.TabItem("📋 Scénarios / Scenarios"):
                                doc_all_cache = gr.State({})
                                doc_all_md = gr.Markdown()
                                language.change(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_scenarios"),
                                    inputs=[language, doc_all_cache],
                                    outputs=[doc_all_md, doc_all_cache],
                                )
                                demo.load(
                                    lambda lang, cache: _doc_loader(lang, cache, "src.utils_ui_doc", "render_doc_scenarios"),
                                    inputs=[language, doc_all_cache],
                                    outputs=[doc_all_md, doc_all_cache],
                                )

                    with gr.TabItem("🌱 Scénario 1 / Scenario 1"):
                        rule_inputs = _build_rule_inputs(env_components["max_irrigation"], "fr")
                        run_btn = gr.Button("🌱 Simuler le scénario 1 / Run Scenario 1")
                        plot_out = gr.Plot()
                        metrics_out = gr.Markdown()

                        run_btn.click(
                            _run_scenario1,
                            inputs=[
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                rule_inputs["rule_type"],
                                rule_inputs["threshold_cbar"],
                                rule_inputs["dose_mm"],
                                rule_inputs["rain_threshold_mm"],
                                rule_inputs["reduce_factor"],
                                rule_inputs["psi_low"],
                                rule_inputs["psi_high"],
                                rule_inputs["dose_mm_band"],
                                rule_inputs["psi_target"],
                                rule_inputs["k_I"],
                                language,
                            ],
                            outputs=[plot_out, metrics_out, scenario1_result_state],
                        )

                    with gr.TabItem("🎓 Scénario 2 / Scenario 2"):
                        if not PPO_AVAILABLE:
                            gr.Markdown("⚠️ PPO indisponible. Installer `gymnasium` et `stable-baselines3`.")
                        apply_goal_btn = goal_components["apply"]
                        goal_status = gr.Markdown()

                        def _apply_goal_spec(
                            enable,
                            stress_max,
                            irrig_max,
                            drain_max,
                            events_max,
                            P1,
                            P2,
                            P3,
                            style,
                        ):
                            spec = build_goal_spec(
                                enable,
                                stress_max,
                                irrig_max,
                                drain_max,
                                events_max,
                                P1,
                                P2,
                                P3,
                                style,
                            )
                            msg = "✅ Objectifs appliqués." if spec else "ℹ️ Objectifs désactivés."
                            return spec, msg

                        apply_goal_btn.click(
                            _apply_goal_spec,
                            inputs=[
                                goal_components["enable"],
                                goal_components["stress_max"],
                                goal_components["irrig_max"],
                                goal_components["drain_max"],
                                goal_components["events_max"],
                                goal_components["P1"],
                                goal_components["P2"],
                                goal_components["P3"],
                                goal_components["style"],
                            ],
                            outputs=[goal_spec_state, goal_status],
                        )

                        train_btn = gr.Button("🚀 Démarrer l'entraînement PPO (Scénario 2) / Start PPO training (Scenario 2)")
                        train_status = gr.Markdown()
                        train_btn.click(
                            _start_training_status,
                            outputs=train_status,
                            queue=False,
                        ).then(
                            _train_scenario2,
                            inputs=[
                                ppo_components["total_timesteps"],
                                ppo_components["policy_type"],
                                ppo_components["n_steps"],
                                ppo_components["batch_size"],
                                ppo_components["learning_rate"],
                                ppo_components["gamma"],
                                ppo_components["gae_lambda"],
                                ppo_components["clip_range"],
                                ppo_components["ent_coef"],
                                ppo_components["vf_coef"],
                                mlp_components["hidden_layer_1"] if mlp_components else gr.State(64),
                                mlp_components["hidden_layer_2"] if mlp_components else gr.State(64),
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                weather_components["weather_source"],
                                weather_components["era5_path"],
                                weather_components["era5_freq"],
                                goal_spec_state,
                            ],
                            outputs=[scenario2_model_state, train_status],
                            show_progress=True,
                        )

                        eval_btn = gr.Button("📈 Évaluer le modèle PPO (Scénario 2) / Evaluate PPO model (Scenario 2)")
                        eval_plot = gr.Plot()
                        eval_metrics = gr.Markdown()
                        eval_btn.click(
                            _eval_scenario2,
                            inputs=[
                                scenario2_model_state,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                weather_components["weather_source"],
                                weather_components["era5_path"],
                                weather_components["era5_freq"],
                                language,
                            ],
                            outputs=[eval_plot, eval_metrics],
                        )
                    with gr.TabItem("🔬 Scénario 3 / Scenario 3"):
                        gr.Markdown("### 🔬 Scenario 3 — Neural ODE (hybride)")
                        if not TORCH_AVAILABLE_ODE:
                            gr.Markdown("⚠️ PyTorch indisponible pour le pré-entraînement.")
                        with gr.Row():
                            ode_n_traj = gr.Number(value=32, label="N_traj", precision=0)
                            ode_epochs = gr.Number(value=10, label="n_epochs", precision=0)
                            ode_batch = gr.Number(value=256, label="batch_size", precision=0)
                            ode_lr = gr.Number(value=1e-3, label="lr", precision=5)
                        pretrain_btn = gr.Button("🧠 Pré-entraîner le résiduel ODE / Pretrain ODE residual")
                        pretrain_status = gr.Markdown()
                        pretrain_btn.click(
                            _start_training_status,
                            outputs=pretrain_status,
                            queue=False,
                        ).then(
                            _pretrain_ode,
                            inputs=[
                                ode_n_traj,
                                ode_epochs,
                                ode_batch,
                                ode_lr,
                                env_components["max_irrigation"],
                                env_components["season_length"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                            ],
                            outputs=[scenario3_residual_state, pretrain_status],
                            show_progress=True,
                        )
                        train_btn = gr.Button("🚀 Entraîner PPO hybride (Scénario 3) / Train hybrid PPO (Scenario 3)")
                        train_status = gr.Markdown()
                        train_btn.click(
                            _start_training_status,
                            outputs=train_status,
                            queue=False,
                        ).then(
                            _train_scenario3,
                            inputs=[
                                scenario3_residual_state,
                                ppo_components["policy_type"],
                                ppo_components["total_timesteps"],
                                ppo_components["n_steps"],
                                ppo_components["batch_size"],
                                ppo_components["learning_rate"],
                                ppo_components["gamma"],
                                ppo_components["gae_lambda"],
                                ppo_components["clip_range"],
                                ppo_components["ent_coef"],
                                ppo_components["vf_coef"],
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                goal_spec_state,
                            ],
                            outputs=[scenario3_model_state, train_status],
                            show_progress=True,
                        )
                        eval_btn = gr.Button("📈 Évaluer (Scénario 3) / Evaluate (Scenario 3)")
                        eval_plot = gr.Plot()
                        eval_status = gr.Markdown()
                        eval_btn.click(
                            _start_eval_status,
                            outputs=eval_status,
                            queue=False,
                        ).then(
                            _eval_scenario3,
                            inputs=[
                                scenario3_model_state,
                                scenario3_residual_state,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                language,
                            ],
                            outputs=[eval_plot, eval_status],
                            show_progress=True,
                        )
                    with gr.TabItem("🧠 Scénario 3b / Scenario 3b"):
                        gr.Markdown("### 🧠 Scenario 3b — Neural ODE continu")
                        if not TORCH_AVAILABLE_ODE_CONT:
                            gr.Markdown("⚠️ PyTorch indisponible pour le pré-entraînement.")
                        with gr.Row():
                            odec_n_traj = gr.Number(value=32, label="N_traj", precision=0)
                            odec_epochs = gr.Number(value=10, label="n_epochs", precision=0)
                            odec_batch = gr.Number(value=256, label="batch_size", precision=0)
                            odec_lr = gr.Number(value=1e-3, label="lr", precision=5)
                        pretrain_btn = gr.Button("🧠 Pré-entraîner le résiduel ODE continu / Pretrain continuous ODE residual")
                        pretrain_status = gr.Markdown()
                        pretrain_btn.click(
                            _start_training_status,
                            outputs=pretrain_status,
                            queue=False,
                        ).then(
                            _pretrain_ode_cont,
                            inputs=[
                                odec_n_traj,
                                odec_epochs,
                                odec_batch,
                                odec_lr,
                                env_components["max_irrigation"],
                                env_components["season_length"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                            ],
                            outputs=[scenario3b_residual_state, pretrain_status],
                            show_progress=True,
                        )
                        train_btn = gr.Button("🚀 Entraîner PPO hybride (Scénario 3b) / Train hybrid PPO (Scenario 3b)")
                        train_status = gr.Markdown()
                        train_btn.click(
                            _start_training_status,
                            outputs=train_status,
                            queue=False,
                        ).then(
                            _train_scenario3b,
                            inputs=[
                                scenario3b_residual_state,
                                ppo_components["policy_type"],
                                ppo_components["total_timesteps"],
                                ppo_components["n_steps"],
                                ppo_components["batch_size"],
                                ppo_components["learning_rate"],
                                ppo_components["gamma"],
                                ppo_components["gae_lambda"],
                                ppo_components["clip_range"],
                                ppo_components["ent_coef"],
                                ppo_components["vf_coef"],
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                goal_spec_state,
                            ],
                            outputs=[scenario3b_model_state, train_status],
                            show_progress=True,
                        )
                        eval_btn = gr.Button("📈 Évaluer (Scénario 3b) / Evaluate (Scenario 3b)")
                        eval_plot = gr.Plot()
                        eval_status = gr.Markdown()
                        eval_btn.click(
                            _start_eval_status,
                            outputs=eval_status,
                            queue=False,
                        ).then(
                            _eval_scenario3b,
                            inputs=[
                                scenario3b_model_state,
                                scenario3b_residual_state,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                language,
                            ],
                            outputs=[eval_plot, eval_status],
                            show_progress=True,
                        )
                    with gr.TabItem("🌀 Scénario 4 / Scenario 4"):
                        gr.Markdown("### 🌀 Scenario 4 — Neural CDE")
                        if not TORCH_AVAILABLE_CDE:
                            gr.Markdown("⚠️ PyTorch indisponible pour le pré-entraînement.")
                        with gr.Row():
                            cde_n_traj = gr.Number(value=32, label="N_traj", precision=0)
                            cde_seq = gr.Number(value=5, label="seq_len", precision=0)
                            cde_epochs = gr.Number(value=10, label="n_epochs", precision=0)
                            cde_batch = gr.Number(value=256, label="batch_size", precision=0)
                            cde_lr = gr.Number(value=1e-3, label="lr", precision=5)
                        pretrain_btn = gr.Button("🧠 Pré-entraîner le résiduel CDE / Pretrain CDE residual")
                        pretrain_status = gr.Markdown()
                        pretrain_btn.click(
                            _start_training_status,
                            outputs=pretrain_status,
                            queue=False,
                        ).then(
                            _pretrain_cde,
                            inputs=[
                                cde_n_traj,
                                cde_seq,
                                cde_epochs,
                                cde_batch,
                                cde_lr,
                                env_components["max_irrigation"],
                                env_components["season_length"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                            ],
                            outputs=[scenario4_residual_state, pretrain_status],
                            show_progress=True,
                        )
                        train_btn = gr.Button("🚀 Entraîner PPO hybride (Scénario 4) / Train hybrid PPO (Scenario 4)")
                        train_status = gr.Markdown()
                        train_btn.click(
                            _start_training_status,
                            outputs=train_status,
                            queue=False,
                        ).then(
                            _train_scenario4,
                            inputs=[
                                scenario4_residual_state,
                                cde_seq,
                                ppo_components["policy_type"],
                                ppo_components["total_timesteps"],
                                ppo_components["n_steps"],
                                ppo_components["batch_size"],
                                ppo_components["learning_rate"],
                                ppo_components["gamma"],
                                ppo_components["gae_lambda"],
                                ppo_components["clip_range"],
                                ppo_components["ent_coef"],
                                ppo_components["vf_coef"],
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                goal_spec_state,
                            ],
                            outputs=[scenario4_model_state, train_status],
                            show_progress=True,
                        )
                        eval_btn = gr.Button("📈 Évaluer (Scénario 4) / Evaluate (Scenario 4)")
                        eval_plot = gr.Plot()
                        eval_status = gr.Markdown()
                        eval_btn.click(
                            _start_eval_status,
                            outputs=eval_status,
                            queue=False,
                        ).then(
                            _eval_scenario4,
                            inputs=[
                                scenario4_model_state,
                                scenario4_residual_state,
                                cde_seq,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                language,
                            ],
                            outputs=[eval_plot, eval_status],
                            show_progress=True,
                        )
                    with gr.TabItem("🔮 Scénario 5 / Scenario 5"):
                        gr.Markdown("### 🔮 Scenario 5 — PatchTST")
                        if not TORCH_AVAILABLE_PATCHTST:
                            gr.Markdown("⚠️ PyTorch indisponible pour PatchTST.")
                        with gr.Row():
                            pt_n_traj = gr.Number(value=32, label="N_traj", precision=0)
                            pt_seq_len = gr.Number(value=30, label="seq_len", precision=0)
                            pt_epochs = gr.Number(value=10, label="n_epochs", precision=0)
                            pt_batch = gr.Number(value=64, label="batch_size", precision=0)
                            pt_lr = gr.Number(value=1e-3, label="lr", precision=5)
                        with gr.Row():
                            pt_feature_dim = gr.Number(value=16, label="feature_dim", precision=0)
                            pt_task = gr.Dropdown(choices=["auto", "supervised"], value="auto", label="task")
                        pretrain_btn = gr.Button("🧠 Pré-entraîner PatchTST / Pretrain PatchTST")
                        pretrain_status = gr.Markdown()
                        pretrain_btn.click(
                            _pretrain_patchtst,
                            inputs=[
                                pt_n_traj,
                                pt_seq_len,
                                pt_epochs,
                                pt_batch,
                                pt_lr,
                                pt_feature_dim,
                                pt_task,
                                env_components["max_irrigation"],
                                env_components["season_length"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                            ],
                            outputs=[patchtst_model_state, pretrain_status],
                        )
                        train_btn = gr.Button("🚀 Entraîner PPO PatchTST (Scénario 5) / Train PPO PatchTST (Scenario 5)")
                        train_status = gr.Markdown()
                        train_btn.click(
                            _train_scenario5,
                            inputs=[
                                patchtst_model_state,
                                pt_seq_len,
                                ppo_components["policy_type"],
                                ppo_components["total_timesteps"],
                                ppo_components["n_steps"],
                                ppo_components["batch_size"],
                                ppo_components["learning_rate"],
                                ppo_components["gamma"],
                                ppo_components["gae_lambda"],
                                ppo_components["clip_range"],
                                ppo_components["ent_coef"],
                                ppo_components["vf_coef"],
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                goal_spec_state,
                            ],
                            outputs=[scenario5_model_state, train_status],
                        )
                        eval_btn = gr.Button("📈 Évaluer (Scénario 5) / Evaluate (Scenario 5)")
                        eval_plot = gr.Plot()
                        eval_status = gr.Markdown()
                        eval_btn.click(
                            _eval_scenario5,
                            inputs=[
                                scenario5_model_state,
                                patchtst_model_state,
                                pt_seq_len,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                language,
                            ],
                            outputs=[eval_plot, eval_status],
                        )
                    with gr.TabItem("🌍 Scénario 6 / Scenario 6"):
                        gr.Markdown("### 🌍 Scenario 6 — World Model")
                        if not TORCH_AVAILABLE_WM:
                            gr.Markdown("⚠️ PyTorch indisponible pour le World Model.")
                        with gr.Row():
                            wm_seq_len = gr.Number(value=30, label="seq_len", precision=0)
                            wm_latent_dim = gr.Number(value=16, label="latent_dim", precision=0)
                            wm_n_traj = gr.Number(value=32, label="N_traj", precision=0)
                            wm_epochs = gr.Number(value=20, label="n_epochs", precision=0)
                            wm_batch = gr.Number(value=64, label="batch_size", precision=0)
                            wm_lr = gr.Number(value=1e-3, label="lr", precision=5)
                        wm_cde_seq = gr.Number(value=10, label="cde_seq_len", precision=0)
                        phase1_btn = gr.Button("⚙️ Entraîner World Model Phase 1 / Train World Model Phase 1")
                        phase1_status = gr.Markdown()
                        phase1_btn.click(
                            _train_wm_phase1,
                            inputs=[
                                patchtst_model_state,
                                wm_n_traj,
                                wm_seq_len,
                                wm_epochs,
                                wm_batch,
                                wm_lr,
                                wm_latent_dim,
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                            ],
                            outputs=[wm_phase1_transition_state, phase1_status],
                        )
                        phase2_btn = gr.Button("⚙️ Entraîner World Model Phase 2 / Train World Model Phase 2")
                        phase2_status = gr.Markdown()
                        phase2_btn.click(
                            _train_wm_phase2,
                            inputs=[
                                patchtst_model_state,
                                wm_n_traj,
                                wm_seq_len,
                                wm_cde_seq,
                                wm_epochs,
                                wm_batch,
                                wm_lr,
                                wm_latent_dim,
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                            ],
                            outputs=[wm_phase2_transition_state, wm_phase2_decoder_state, phase2_status],
                        )
                        gr.Markdown("#### PPO avec World Model")
                        with gr.Row():
                            imag_h = gr.Number(value=5, label="imagination_horizon", precision=0)
                            imag_ratio = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="imagination_ratio")
                        phase1ppo_btn = gr.Button("🚀 PPO Phase 1 / PPO Phase 1")
                        phase1ppo_status = gr.Markdown()
                        phase1ppo_btn.click(
                            _train_ppo_wm_phase1,
                            inputs=[
                                patchtst_model_state,
                                wm_phase1_transition_state,
                                imag_h,
                                imag_ratio,
                                ppo_components["policy_type"],
                                ppo_components["total_timesteps"],
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                            ],
                            outputs=[wm_phase1_ppo_state, phase1ppo_status],
                        )
                        phase2ppo_btn = gr.Button("🚀 PPO Phase 2 / PPO Phase 2")
                        phase2ppo_status = gr.Markdown()
                        phase2ppo_btn.click(
                            _train_ppo_wm_phase2,
                            inputs=[
                                patchtst_model_state,
                                wm_phase2_transition_state,
                                wm_phase2_decoder_state,
                                imag_h,
                                imag_ratio,
                                ppo_components["policy_type"],
                                ppo_components["total_timesteps"],
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                            ],
                            outputs=[wm_phase2_ppo_state, phase2ppo_status],
                        )
                        with gr.Row():
                            wm_seq_len_eval = gr.Number(value=30, label="wm_seq_len", precision=0)
                            wm_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="hybrid_alpha")
                        phase3ppo_btn = gr.Button("🚀 PPO Phase 3 (hybride) / PPO Phase 3 (hybrid)")
                        phase3ppo_status = gr.Markdown()
                        phase3ppo_btn.click(
                            _train_ppo_wm_phase3,
                            inputs=[
                                patchtst_model_state,
                                wm_phase2_transition_state,
                                wm_phase2_decoder_state,
                                wm_seq_len_eval,
                                wm_alpha,
                                ppo_components["policy_type"],
                                ppo_components["total_timesteps"],
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                weather_components["weather_source"],
                                weather_components["era5_path"],
                                weather_components["era5_freq"],
                            ],
                            outputs=[wm_phase3_ppo_state, phase3ppo_status],
                        )
                        eval_btn = gr.Button("📈 Évaluer (Scénario 6) / Evaluate (Scenario 6)")
                        eval_plot = gr.Plot()
                        eval_status = gr.Markdown()
                        eval_btn.click(
                            _eval_scenario6,
                            inputs=[
                                wm_phase3_ppo_state,
                                patchtst_model_state,
                                wm_phase2_transition_state,
                                wm_phase2_decoder_state,
                                wm_seq_len_eval,
                                wm_alpha,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                weather_components["weather_source"],
                                weather_components["era5_path"],
                                weather_components["era5_freq"],
                                language,
                            ],
                            outputs=[eval_plot, eval_status],
                        )
                    with gr.TabItem("📈 Évaluation / Evaluation"):
                        gr.Markdown("### 📈 Evaluation")
                        eval2_btn = gr.Button("Évaluer Scénario 2 / Evaluate Scenario 2")
                        eval2_status = gr.Markdown()
                        eval2_btn.click(
                            _rollout_scenario2,
                            inputs=[
                                scenario2_model_state,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                                weather_components["weather_source"],
                                weather_components["era5_path"],
                                weather_components["era5_freq"],
                            ],
                            outputs=[scenario2_rollout_state, eval2_status],
                        )

                        eval3_btn = gr.Button("Évaluer Scénario 3 / Evaluate Scenario 3")
                        eval3_status = gr.Markdown()
                        eval3_btn.click(
                            _rollout_scenario3,
                            inputs=[
                                scenario3_model_state,
                                scenario3_residual_state,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                            ],
                            outputs=[scenario3_rollout_state, eval3_status],
                        )

                        eval3b_btn = gr.Button("Évaluer Scénario 3b / Evaluate Scenario 3b")
                        eval3b_status = gr.Markdown()
                        eval3b_btn.click(
                            _rollout_scenario3,
                            inputs=[
                                scenario3b_model_state,
                                scenario3b_residual_state,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                            ],
                            outputs=[scenario3b_rollout_state, eval3b_status],
                        )

                        eval4_btn = gr.Button("Évaluer Scénario 4 / Evaluate Scenario 4")
                        eval4_status = gr.Markdown()
                        eval4_btn.click(
                            _rollout_scenario4,
                            inputs=[
                                scenario4_model_state,
                                scenario4_residual_state,
                                cde_seq,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                            ],
                            outputs=[scenario4_rollout_state, eval4_status],
                        )

                        eval5_btn = gr.Button("Évaluer Scénario 5 / Evaluate Scenario 5")
                        eval5_status = gr.Markdown()
                        eval5_btn.click(
                            _rollout_scenario5,
                            inputs=[
                                scenario5_model_state,
                                patchtst_model_state,
                                pt_seq_len,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                            ],
                            outputs=[scenario5_rollout_state, eval5_status],
                        )

                        eval6_btn = gr.Button("Évaluer Scénario 6 / Evaluate Scenario 6")
                        eval6_status = gr.Markdown()
                        eval6_btn.click(
                            _rollout_scenario6,
                            inputs=[
                                wm_phase3_ppo_state,
                                patchtst_model_state,
                                wm_phase2_transition_state,
                                wm_phase2_decoder_state,
                                wm_seq_len_eval,
                                wm_alpha,
                                env_components["season_length"],
                                env_components["max_irrigation"],
                                env_components["seed"],
                                soil_components["Z_r"],
                                soil_components["theta_s"],
                                soil_components["theta_fc"],
                                soil_components["theta_wp"],
                                soil_components["psi_sat"],
                                soil_components["psi_fc"],
                                soil_components["psi_wp"],
                                soil_components["k_d"],
                                soil_components["eta_I"],
                                soil_components["psi_ET_crit"],
                                weather_components["et0_base"],
                                weather_components["et0_amp"],
                                weather_components["et0_noise"],
                                weather_components["p_rain_early"],
                                weather_components["p_rain_mid"],
                                weather_components["p_rain_late"],
                                weather_components["rain_min"],
                                weather_components["rain_max"],
                            ],
                            outputs=[scenario6_rollout_state, eval6_status],
                        )
                    with gr.TabItem("📊 Visualisation / Visualization"):
                        gr.Markdown("### 📊 Visualisation")
                        viz_choice = gr.Dropdown(
                            choices=["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 3b", "Scenario 4", "Scenario 5", "Scenario 6"],
                            value="Scenario 1",
                            label="Choisir un scénario / Select scenario",
                        )
                        viz_btn = gr.Button("Afficher / Show")
                        viz_plot = gr.Plot()
                        viz_status = gr.Markdown()
                        viz_btn.click(
                            _render_visualization,
                            inputs=[
                                viz_choice,
                                scenario1_result_state,
                                scenario2_rollout_state,
                                scenario3_rollout_state,
                                scenario3b_rollout_state,
                                scenario4_rollout_state,
                                scenario5_rollout_state,
                                scenario6_rollout_state,
                                language,
                            ],
                            outputs=[viz_plot, viz_status],
                        )
                    with gr.TabItem("⚖️ Comparaison / Comparison"):
                        gr.Markdown("### ⚖️ Comparaison")
                        compare_btn = gr.Button("Construire tableau comparatif / Build comparison table")
                        compare_table = gr.Dataframe()
                        compare_plot = gr.Plot()
                        compare_btn.click(
                            lambda s1, s2, s3, s3b, s4, s5, s6: (
                                _build_comparison_table(s1, s2, s3, s3b, s4, s5, s6),
                                _plot_rollout_summary(_build_comparison_table(s1, s2, s3, s3b, s4, s5, s6)),
                            ),
                            inputs=[
                                scenario1_result_state,
                                scenario2_rollout_state,
                                scenario3_rollout_state,
                                scenario3b_rollout_state,
                                scenario4_rollout_state,
                                scenario5_rollout_state,
                                scenario6_rollout_state,
                            ],
                            outputs=[compare_table, compare_plot],
                        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch()
