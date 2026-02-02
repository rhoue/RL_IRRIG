#!/usr/bin/env python3
"""
Pure Python CLI runner for RL Intelligent Irrigation.

Examples:
  python main.py run --scenario scenario1
  python main.py run --scenario scenario2 --model-path models/scenario2_ppo.zip
  python main.py run --scenario scenario3 --model-path models/scenario3_ppo.zip --residual-path models/scenario3_residual.pt
  python main.py compare --scenarios scenario1 scenario2 scenario3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils_physical_model import (
    PhysicalBucket,
    evaluate_episode,
    rule_bande_confort,
    rule_proportionnelle,
    rule_seuil_unique,
    simulate_scenario1,
)
from src.utils_physics_config import (
    get_default_era5_land_config,
    get_default_soil_config,
    get_default_weather_config,
)


SCENARIOS = ("scenario1", "scenario2", "scenario3", "scenario3b")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_np(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _standardize_rollout(scenario: str, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if scenario == "scenario1":
        return {
            "psi": _to_np(data["psi"]),
            "S": _to_np(data["S"]),
            "I": _to_np(data["I"]),
            "R": _to_np(data["rain"]),
            "ETc": _to_np(data["ETc"]),
            "D": _to_np(data["D"]),
        }
    return {
        "psi": _to_np(data["psi"]),
        "S": _to_np(data["S"]),
        "I": _to_np(data["I"]),
        "R": _to_np(data["R"]),
        "ETc": _to_np(data["ETc"]),
        "D": _to_np(data["D"]),
    }


def _compute_metrics(std: Dict[str, np.ndarray]) -> Dict[str, float]:
    psi = std["psi"][1:] if len(std["psi"]) == len(std["I"]) + 1 else std["psi"]
    I = std["I"]
    R = std["R"]
    ETc = std["ETc"]
    D = std["D"]
    water = float(np.sum(I + R))
    return {
        "total_irrigation_mm": float(np.sum(I)),
        "total_rain_mm": float(np.sum(R)),
        "total_etc_mm": float(np.sum(ETc)),
        "total_drainage_mm": float(np.sum(D)),
        "mean_psi_cbar": float(np.mean(psi)),
        "comfort_days_pct": float(100.0 * np.mean((psi >= 20.0) & (psi <= 60.0))),
        "water_efficiency": float(np.sum(ETc) / water) if water > 0 else 0.0,
    }


def _save_run_artifact(out_dir: Path, scenario: str, std: Dict[str, np.ndarray], metrics: Dict[str, float]) -> None:
    _ensure_dir(out_dir)
    np.savez_compressed(out_dir / f"{scenario}.npz", **std)
    (out_dir / f"{scenario}_metrics.json").write_text(json.dumps(metrics, indent=2))


def _load_run_artifact(input_dir: Path, scenario: str) -> Dict[str, np.ndarray]:
    path = input_dir / f"{scenario}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _build_rule(args: argparse.Namespace):
    if args.rule == "threshold":
        return rule_seuil_unique, {
            "threshold_cbar": args.threshold_cbar,
            "dose_mm": args.dose_mm,
            "rain_threshold_mm": args.rain_threshold_mm,
            "reduce_factor": args.reduce_factor,
        }
    if args.rule == "band":
        return rule_bande_confort, {
            "psi_low": args.psi_low,
            "psi_high": args.psi_high,
            "dose_mm": args.dose_mm,
        }
    return rule_proportionnelle, {
        "psi_target": args.psi_target,
        "k_I": args.k_i,
    }


def _load_models_for_scenario(args: argparse.Namespace):
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for scenario2/3/3b.") from exc

    model = PPO.load(args.model_path)
    residual = None
    if args.scenario in ("scenario3", "scenario3b"):
        if not args.residual_path:
            raise ValueError("--residual-path is required for scenario3/scenario3b.")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required for residual model loading.") from exc

        loaded = torch.load(args.residual_path, map_location="cpu")
        if hasattr(loaded, "eval"):
            residual = loaded
        else:
            if args.scenario == "scenario3":
                from src.utils_neuro_ode import ResidualODEModel

                residual = ResidualODEModel(in_dim=4, hidden=64)
            else:
                from src.utils_neuro_ode_cont import ContinuousResidualODE

                residual = ContinuousResidualODE(hidden=64)
            if isinstance(loaded, dict) and "state_dict" in loaded:
                residual.load_state_dict(loaded["state_dict"], strict=False)
            elif isinstance(loaded, dict):
                residual.load_state_dict(loaded, strict=False)
            residual.eval()
    return model, residual


def run_command(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    soil_params = get_default_soil_config()
    weather_params = get_default_weather_config()

    if args.scenario == "scenario1":
        rule_fn, rule_kwargs = _build_rule(args)
        sim = simulate_scenario1(
            T=args.season_length,
            seed=args.seed,
            I_max=args.max_irrigation,
            soil=PhysicalBucket(**soil_params),
            rule_fn=rule_fn,
            rule_kwargs=rule_kwargs,
            weather_params=weather_params,
        )
        std = _standardize_rollout("scenario1", sim)
    else:
        model, residual = _load_models_for_scenario(args)
        data_source = "era5_land" if args.era5_path else "synthetic"
        era_cfg = None
        if data_source == "era5_land":
            era_cfg = get_default_era5_land_config(
                use_era5_land=True,
                data_path=args.era5_path,
                resample_freq=args.era5_freq,
            )
        rollout = evaluate_episode(
            model=model,
            season_length=args.season_length,
            max_irrigation=args.max_irrigation,
            seed=args.seed,
            soil_params=soil_params,
            weather_params=weather_params,
            residual_ode=residual if args.scenario in ("scenario3", "scenario3b") else None,
            data_source=data_source,
            data_path=args.era5_path,
            era5_land_cfg=era_cfg,
        )
        std = _standardize_rollout(args.scenario, rollout)

    metrics = _compute_metrics(std)
    _save_run_artifact(out_dir, args.scenario, std, metrics)
    print(f"[ok] Saved: {out_dir / (args.scenario + '.npz')}")
    print(json.dumps(metrics, indent=2))
    return 0


def compare_command(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    data: Dict[str, Dict[str, np.ndarray]] = {}
    rows: List[Dict[str, Any]] = []
    for scenario in args.scenarios:
        std = _load_run_artifact(input_dir, scenario)
        data[scenario] = std
        rows.append({"scenario": scenario, **_compute_metrics(std)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_metrics.csv", index=False)
    print(df.to_string(index=False))
    print(f"[ok] Saved metrics: {out_dir / 'comparison_metrics.csv'}")

    # Plot 1: tension
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for scenario, std in data.items():
        y = std["psi"][1:] if len(std["psi"]) == len(std["I"]) + 1 else std["psi"]
        ax1.plot(np.arange(len(y)), y, linewidth=2, label=scenario)
    ax1.axhspan(20, 60, color="green", alpha=0.12, label="optimal 20-60 cbar")
    ax1.set_title("Matric tension comparison")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("psi (cbar)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "comparison_tension.png", dpi=150)

    # Plot 2: reserve
    soil = get_default_soil_config()
    s_fc = soil["theta_fc"] * soil["Z_r"]
    s_wp = soil["theta_wp"] * soil["Z_r"]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for scenario, std in data.items():
        y = std["S"][1:] if len(std["S"]) == len(std["I"]) + 1 else std["S"]
        ax2.plot(np.arange(len(y)), y, linewidth=2, label=scenario)
    ax2.axhline(s_fc, color="gray", linestyle="--", label="S_fc")
    ax2.axhline(s_wp, color="#b5651d", linestyle="--", label="S_wp")
    ax2.set_title("Soil storage comparison")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("S (mm)")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "comparison_storage.png", dpi=150)

    # Plot 3: irrigation + rain totals
    labels = list(data.keys())
    irr = [float(np.sum(v["I"])) for v in data.values()]
    rain = [float(np.sum(v["R"])) for v in data.values()]
    x = np.arange(len(labels))
    w = 0.4
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(x - w / 2, irr, width=w, label="total irrigation")
    ax3.bar(x + w / 2, rain, width=w, label="total rain", color="#58d1d8")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=20)
    ax3.set_ylabel("mm")
    ax3.set_title("Water volumes (irrigation + rain)")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "comparison_irrigation_rain.png", dpi=150)

    print(f"[ok] Saved plots in: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for RL Intelligent Irrigation.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one scenario and save artifact.")
    run.add_argument("--scenario", choices=SCENARIOS, required=True)
    run.add_argument("--season-length", type=int, default=120)
    run.add_argument("--seed", type=int, default=123)
    run.add_argument("--max-irrigation", type=float, default=20.0)
    run.add_argument("--output-dir", default="outputs")
    run.add_argument("--model-path", default=None, help="PPO zip path (required for scenario2/3/3b).")
    run.add_argument("--residual-path", default=None, help="Residual model path (required for scenario3/3b).")
    run.add_argument("--era5-path", default=None, help="Optional ERA5-Land .nc path.")
    run.add_argument("--era5-freq", default="1D")

    # Scenario1 rule options
    run.add_argument("--rule", choices=("threshold", "band", "proportional"), default="threshold")
    run.add_argument("--threshold-cbar", type=float, default=80.0)
    run.add_argument("--dose-mm", type=float, default=15.0)
    run.add_argument("--rain-threshold-mm", type=float, default=2.0)
    run.add_argument("--reduce-factor", type=float, default=0.5)
    run.add_argument("--psi-low", type=float, default=20.0)
    run.add_argument("--psi-high", type=float, default=60.0)
    run.add_argument("--psi-target", type=float, default=40.0)
    run.add_argument("--k-i", type=float, default=0.1)
    run.set_defaults(func=run_command)

    cmp_cmd = sub.add_parser("compare", help="Compare already saved scenario artifacts.")
    cmp_cmd.add_argument("--scenarios", nargs="+", choices=SCENARIOS, required=True)
    cmp_cmd.add_argument("--input-dir", default="outputs")
    cmp_cmd.add_argument("--output-dir", default="outputs")
    cmp_cmd.set_defaults(func=compare_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

