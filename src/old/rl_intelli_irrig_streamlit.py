"""
Streamlit web application for RL Intelligent Irrigation Environment.

This module provides an interactive web interface to visualize and experiment
with the irrigation environment. Users can adjust parameters, run episodes,
and see real-time visualizations of soil moisture, irrigation actions, and rewards.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Optional

from src.utils_env_modeles import IrrigationEnvPhysical
from rl_intelli_irrig_basic import BasicIntelligentIrrigation

# Try to import PPO functionality
try:
    from rl_intelli_irrig_ppo import PPOIrrigationTrainer, integrate_with_basic
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    PPOIrrigationTrainer = None
    integrate_with_basic = None


# Configure page
st.set_page_config(
    page_title="RL Intelligent Irrigation",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8")


def initialize_session_state():
    """
    Initialize session state variables if they don't exist.
    
    Streamlit's session_state persists data across reruns of the app.
    This function ensures all required state variables are initialized
    to avoid KeyError exceptions.
    
    State variables:
    - episode_history: List of all completed episode data (for comparison/analysis)
    - current_episode: Most recently run episode data (for display)
    - ppo_model: Trained PPO model (if training was performed)
    - ppo_trainer: PPO trainer instance (for evaluation/retraining)
    - training_progress: Current training progress (if training is in progress)
    """
    if "episode_history" not in st.session_state:
        st.session_state.episode_history = []  # Store history of all episodes
    if "current_episode" not in st.session_state:
        st.session_state.current_episode = None  # Current episode being displayed
    if "ppo_model" not in st.session_state:
        st.session_state.ppo_model = None  # Trained PPO model for AI policy
    if "ppo_trainer" not in st.session_state:
        st.session_state.ppo_trainer = None  # PPO trainer instance
    if "training_progress" not in st.session_state:
        st.session_state.training_progress = None  # Training progress tracking


def run_episode_with_policy(
    demo: BasicIntelligentIrrigation,
    policy: str,
    **policy_kwargs
) -> Dict[str, Any]:
    """
    Run an episode with specified policy and return episode data.
    
    This is a wrapper function that calls the demo's run_episode method.
    It simulates one complete irrigation season using the specified policy.
    
    Args:
        demo: BasicIntelligentIrrigation instance with configured environment
        policy: Policy name ("random", "threshold", "reactive", "none", or "ppo")
        **policy_kwargs: Additional parameters specific to the policy
            - For "threshold": threshold_psi, irrigation_amount
            - For "reactive": min_psi, max_psi
            - For "ppo": deterministic (bool)
    
    Returns:
        Dictionary containing:
            - observations: List of state observations at each step
            - actions: List of irrigation actions taken
            - rewards: List of rewards received
            - info: List of info dictionaries with detailed metrics
            - total_reward: Sum of all rewards
    """
    return demo.run_episode(policy=policy, **policy_kwargs)


def plot_episode_metrics_notebook_style(episode_data: Dict[str, Any], S_fc: float = 90.0, S_wp: float = 30.0):
    """
    Create notebook-style 4-panel visualization similar to the research plots.
    
    This function generates a comprehensive 4-panel vertical layout showing:
    1. Soil tension (ψ) over time with optimal zones and hazard markers
    2. Soil moisture (S) over time with field capacity and wilting point
    3. Irrigation and rainfall (stacked bars) with hazard indicators
    4. ETc and drainage fluxes over time
    
    Args:
        episode_data: Dictionary containing episode observations, actions, rewards, and info
        S_fc: Field capacity (mm) - used for soil moisture visualization
        S_wp: Wilting point (mm) - used for soil moisture visualization
    
    Returns:
        matplotlib.figure.Figure: Figure object with 4 subplots
    """
    # Extract episode data arrays
    observations = np.array(episode_data["observations"])  # State observations: [psi, S, R, ET0]
    actions = np.array(episode_data["actions"])  # Irrigation actions (mm) taken at each step
    info_list = episode_data.get("info", [])  # Additional info (ETc, D, hazards, etc.)
    
    # Observations include initial state + T steps = T+1 values
    # Actions and info have T values (one per step)
    T = len(actions)  # actual number of steps (season length)
    num_obs = len(observations)  # actual number of observations
    
    # Extract state variables first to see what we have
    psi = observations[:, 0]  # Tension
    S = observations[:, 1]    # Soil moisture
    
    # Use actual length of observations for state timeline
    # t_state should match the length of psi (which is len(observations))
    t_state = np.arange(len(psi))  # for state variables (psi, S) - matches actual observation count
    t_flux = np.arange(T)  # for flux variables (I, R, ETc, D) - T values, days 0 to T-1
    
    # Extract flux variables - observations[0] is initial state, observations[1:] are after steps
    # For rain, we use the observation at each step (which includes forecast/current rain)
    if len(observations) > 1:
        # Take observations[1:] up to T elements (one per action)
        num_rain_points = min(len(observations) - 1, T)
        R = observations[1:1+num_rain_points, 2]  # Rain
    else:
        R = np.zeros(T)
    
    # Extract ETc and Drainage from info (one value per step)
    ETc = np.array([info.get("ETc", 0.0) for info in info_list[:T]]) if info_list else np.zeros(T)
    D = np.array([info.get("D", 0.0) for info in info_list[:T]]) if info_list else np.zeros(T)
    
    # Extract hazard events for visualization
    hazard_days = {}  # {day: [hazard_types]}
    for i, info in enumerate(info_list[:T]):
        if info.get("active_hazards"):
            hazard_days[i + 1] = info["active_hazards"]
    
    # Ensure all flux arrays have exactly T elements to match actions
    if len(R) < T:
        R = np.pad(R, (0, T - len(R)), 'constant')
    elif len(R) > T:
        R = R[:T]
    
    if len(ETc) < T:
        ETc = np.pad(ETc, (0, T - len(ETc)), 'constant')
    elif len(ETc) > T:
        ETc = ETc[:T]
    
    if len(D) < T:
        D = np.pad(D, (0, T - len(D)), 'constant')
    elif len(D) > T:
        D = D[:T]
    
    # Ensure actions has exactly T elements
    if len(actions) != T:
        actions = actions[:T] if len(actions) > T else np.pad(actions, (0, T - len(actions)), 'constant')
    
    # t_flux should match T (one value per action/step)
    # Use 0-based indexing (0 to T-1) for flux arrays
    t_flux = np.arange(T)
    
    # Create 4-panel vertical layout
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    title_suffix = " (with Hazards)" if hazard_days else ""
    fig.suptitle(f"Episode Analysis - Water Balance & Dynamics{title_suffix}", fontsize=16, fontweight="bold", y=0.995)
    
    # 1. Soil Tension (ψ) with hazard markers
    axs[0].plot(t_state, psi, color="tab:red", linewidth=2, label="ψ (tension)", zorder=3)
    axs[0].axhspan(20, 60, color="tab:green", alpha=0.15, label="Confort (20-60 cbar)")
    axs[0].axhline(y=30, color="green", linestyle="--", alpha=0.5, linewidth=1)
    axs[0].axhline(y=60, color="green", linestyle="--", alpha=0.5, linewidth=1)
    
    # Add hazard event markers
    hazard_colors = {
        "drought": "brown",
        "flood": "cyan",
        "heatwave": "orange",
        "equipment_failure": "red",
        "water_restriction": "purple"
    }
    for day, hazards in hazard_days.items():
        if day < len(psi):
            for hazard in hazards:
                if hazard != "equipment_failure_warning":
                    color = hazard_colors.get(hazard, "gray")
                    axs[0].axvspan(day - 0.5, day + 0.5, alpha=0.2, color=color, zorder=1)
                    # Add marker
                    if day < len(psi):
                        axs[0].plot(day, psi[day], 'o', color=color, markersize=8, alpha=0.7, zorder=4)
    
    axs[0].set_ylabel("ψ (cbar)", fontsize=11)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='best', fontsize=9)
    axs[0].set_ylim(0, max(150, psi.max() * 1.1))
    
    # 2. Soil Moisture (S)
    axs[1].plot(t_state, S, color="tab:blue", linewidth=2, label="S (réserve d'eau)")
    axs[1].axhline(S_fc, ls="--", color="gray", alpha=0.7, linewidth=1.5, label="S_fc (Field Capacity)")
    axs[1].axhline(S_wp, ls="--", color="brown", alpha=0.7, linewidth=1.5, label="S_wp (Wilting Point)")
    axs[1].set_ylabel("S (mm)", fontsize=11)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='best', fontsize=9)
    
    # 3. Irrigation and Rain (stacked bars) with hazard indicators
    # At this point, actions and R should both have exactly T elements
    I_values = actions[:T]  # Should be exactly T elements
    R_values = R[:T]  # Should be exactly T elements
    
    # Color bars based on hazards
    bar_colors_I = ["tab:blue"] * T
    for day, hazards in hazard_days.items():
        day_idx = day - 1  # Convert to 0-indexed
        if 0 <= day_idx < T:
            if "equipment_failure" in hazards:
                bar_colors_I[day_idx] = "red"
            elif "water_restriction" in hazards:
                bar_colors_I[day_idx] = "purple"
            elif any(h in hazards for h in ["drought", "heatwave"]):
                bar_colors_I[day_idx] = "orange"
    
    bars_I = axs[2].bar(t_flux, I_values, width=0.8, label="Irrigation", color=bar_colors_I, alpha=0.7)
    bars_R = axs[2].bar(t_flux, R_values, width=0.8, bottom=I_values, label="Pluie", color="tab:cyan", alpha=0.5)
    
    # Add hazard labels
    hazard_y_pos = max(I_values + R_values) * 1.05 if len(I_values) > 0 else 10
    for day, hazards in hazard_days.items():
        day_idx = day - 1
        if 0 <= day_idx < T:
            for hazard in hazards:
                if hazard != "equipment_failure_warning":
                    color = hazard_colors.get(hazard, "gray")
                    marker = {"drought": "🌵", "flood": "🌊", "heatwave": "🔥", 
                             "equipment_failure": "🔧", "water_restriction": "🚱"}.get(hazard, "⚠️")
                    axs[2].text(day_idx, hazard_y_pos, marker, ha='center', fontsize=10, alpha=0.7)
    
    axs[2].set_ylabel("mm", fontsize=11)
    axs[2].grid(True, alpha=0.3, axis="y")
    axs[2].legend(loc='best', fontsize=9)
    
    # 4. ETc and Drainage (water fluxes)
    # At this point, ETc and D should both have exactly T elements
    ETc_vals = ETc[:T]  # Should be exactly T elements
    D_vals = D[:T]  # Should be exactly T elements
    
    axs[3].plot(t_flux, ETc_vals, label="ETc", color="tab:orange", linewidth=2, marker='o', markersize=2)
    axs[3].plot(t_flux, D_vals, label="Drainage", color="tab:purple", linewidth=2, marker='s', markersize=2)
    axs[3].set_ylabel("Flux (mm)", fontsize=11)
    axs[3].set_xlabel("Jour (Day)", fontsize=11)
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_episode_metrics(episode_data: Dict[str, Any], S_fc: float = 90.0, S_wp: float = 30.0):
    """
    Create visualizations for episode metrics with decision zones.
    
    Generates a comprehensive 6-panel (2x3 grid) visualization showing:
    1. Soil tension with decision zones (optimal/warning/critical)
    2. Soil moisture with decision zones
    3. Irrigation actions with color coding
    4. Cumulative reward over time
    5. Weather conditions (rain and ET0)
    6. Reward per step
    
    Args:
        episode_data: Dictionary containing episode data
        S_fc: Field capacity (mm) for visualization thresholds
        S_wp: Wilting point (mm) for visualization thresholds
    
    Returns:
        matplotlib.figure.Figure: Figure object with 6 subplots arranged in 2x3 grid
    """
    # Extract arrays from episode data
    observations = np.array(episode_data["observations"])  # [psi, S, R, ET0] at each step
    actions = np.array(episode_data["actions"])  # Irrigation amounts (mm)
    rewards = np.array(episode_data["rewards"])  # Rewards received at each step
    
    # Observations include initial state + T steps = T+1 values
    # Actions and rewards have T values (one per step)
    T = len(actions)  # actual number of steps
    days_state = np.arange(len(observations))  # For state variables (psi, S) - T+1 values
    days_flux = np.arange(1, T + 1)  # For flux variables (actions, rewards, R, ET0) - T values, days 1 to T

    # Extract state variables (T+1 values: initial + after each step)
    psi = observations[:, 0]  # Tension
    S = observations[:, 1]    # Soil moisture
    
    # Extract flux variables - observations[0] is initial state, observations[1:] are after steps
    # For rain and ET0, we use the observation at each step (which includes forecast/current rain)
    R = observations[1:, 2] if len(observations) > 1 else np.zeros(T)  # Rain (T values)
    ET0 = observations[1:, 3] if len(observations) > 1 else np.zeros(T)  # ET0 (T values)
    
    # Ensure all flux arrays have exactly T elements to match actions and days_flux
    if len(R) < T:
        R = np.pad(R, (0, T - len(R)), 'constant')
    elif len(R) > T:
        R = R[:T]
    if len(ET0) < T:
        ET0 = np.pad(ET0, (0, T - len(ET0)), 'constant')
    elif len(ET0) > T:
        ET0 = ET0[:T]
    
    # Ensure actions and rewards have exactly T elements
    if len(actions) != T:
        actions = actions[:T] if len(actions) > T else np.pad(actions, (0, T - len(actions)), 'constant')
    if len(rewards) != T:
        rewards = rewards[:T] if len(rewards) > T else np.pad(rewards, (0, T - len(rewards)), 'constant')
    
    # Final safety check: ensure days_flux matches actions length
    if len(days_flux) != len(actions):
        days_flux = np.arange(1, len(actions) + 1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Episode Metrics Visualization with Decision Zones", fontsize=16, fontweight="bold")

    # 1. Soil Tension (ψ) with decision zones
    ax1 = axes[0, 0]
    # Color code based on zones
    colors = ['green' if 30 <= p <= 60 else 'yellow' if 20 <= p < 30 or 60 < p <= 80 else 'red' for p in psi]
    ax1.scatter(days_state, psi, c=colors, alpha=0.6, s=30, zorder=3)
    ax1.plot(days_state, psi, "b-", linewidth=2, alpha=0.7, label="Soil Tension (ψ)", zorder=2)
    
    # Decision zones
    ax1.axhspan(30, 60, alpha=0.2, color="green", label="Optimal Zone")
    ax1.axhspan(20, 30, alpha=0.15, color="yellow", label="Warning Zone")
    ax1.axhspan(60, 80, alpha=0.15, color="yellow")
    ax1.axhspan(0, 20, alpha=0.1, color="red", label="Critical Zone")
    ax1.axhspan(80, 200, alpha=0.1, color="red")
    
    ax1.axhline(y=30, color="g", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.axhline(y=60, color="g", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.axhline(y=20, color="orange", linestyle=":", alpha=0.5)
    ax1.axhline(y=80, color="orange", linestyle=":", alpha=0.5)
    
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Tension (cbar)")
    ax1.set_title("Soil Tension with Decision Zones")
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(150, psi.max() * 1.1))

    # 2. Soil Moisture (S) with decision zones
    ax2 = axes[0, 1]
    optimal_min = S_wp + 10
    # Color code based on zones
    s_colors = ['green' if optimal_min <= s <= S_fc else 'yellow' if S_wp <= s < optimal_min or S_fc < s <= S_fc + 10 else 'red' for s in S]
    ax2.scatter(days_state, S, c=s_colors, alpha=0.6, s=30, zorder=3)
    ax2.plot(days_state, S, "brown", linewidth=2, alpha=0.7, label="Soil Moisture (S)", zorder=2)
    
    # Decision zones
    ax2.axhspan(optimal_min, S_fc, alpha=0.2, color="green", label="Optimal Zone")
    ax2.axhspan(S_wp, optimal_min, alpha=0.15, color="yellow", label="Warning Zone")
    ax2.axhspan(S_fc, S_fc + 10, alpha=0.15, color="yellow")
    
    ax2.axhline(y=S_fc, color="g", linestyle="--", alpha=0.7, linewidth=1.5, label="Field Capacity")
    ax2.axhline(y=S_wp, color="r", linestyle="--", alpha=0.7, linewidth=1.5, label="Wilting Point")
    ax2.axhline(y=optimal_min, color="orange", linestyle=":", alpha=0.5)
    
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Moisture (mm)")
    ax2.set_title("Soil Moisture with Decision Zones")
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Irrigation Actions with recommendations overlay
    ax3 = axes[0, 2]
    # Color bars based on amount
    bar_colors = ['lightblue' if a < 5 else 'blue' if a < 15 else 'darkblue' for a in actions]
    ax3.bar(days_flux, actions, color=bar_colors, alpha=0.7, label="Irrigation", edgecolor='navy', linewidth=0.5)
    
    # Add recommendation threshold line
    avg_action = np.mean(actions) if len(actions) > 0 else 0
    ax3.axhline(y=avg_action, color="orange", linestyle="--", alpha=0.6, linewidth=1.5, label=f"Average: {avg_action:.1f} mm")
    
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Irrigation (mm)")
    ax3.set_title("Irrigation Actions")
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Cumulative Reward
    ax4 = axes[1, 0]
    cumulative_reward = np.cumsum(rewards)
    ax4.plot(days_flux, cumulative_reward, "g-", linewidth=2)
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Cumulative Reward")
    ax4.set_title("Cumulative Reward Over Time")
    ax4.grid(True, alpha=0.3)

    # 5. Weather Conditions
    ax5 = axes[1, 1]
    ax5_twin = ax5.twinx()
    line1 = ax5.bar(days_flux, R, color="cyan", alpha=0.6, label="Rain (R)")
    line2 = ax5_twin.plot(days_flux, ET0, "orange", linewidth=2, label="ET0")
    ax5.set_xlabel("Day")
    ax5.set_ylabel("Rain (mm)", color="cyan")
    ax5_twin.set_ylabel("ET0 (mm/day)", color="orange")
    ax5.set_title("Weather Conditions")
    ax5.tick_params(axis="y", labelcolor="cyan")
    ax5_twin.tick_params(axis="y", labelcolor="orange")
    lines = [line1] + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc="upper left")
    ax5.grid(True, alpha=0.3)

    # 6. Reward per Step
    ax6 = axes[1, 2]
    ax6.plot(days_flux, rewards, "purple", linewidth=2, marker="o", markersize=3)
    ax6.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax6.set_xlabel("Day")
    ax6.set_ylabel("Reward")
    ax6.set_title("Reward per Step")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def get_status_color(value: float, optimal_min: float, optimal_max: float, warning_min: float, warning_max: float) -> tuple:
    """
    Get color and status based on value ranges.
    
    Used for visual indicators in the UI. Categorizes values into:
    - Optimal (green): Within optimal range
    - Warning (yellow): Outside optimal but within warning range
    - Critical (red): Outside warning range
    
    Args:
        value: Current value to evaluate
        optimal_min: Lower bound of optimal range
        optimal_max: Upper bound of optimal range
        warning_min: Lower bound of warning range
        warning_max: Upper bound of warning range
    
    Returns:
        Tuple of (color_icon, status_text, hex_color)
        Example: ("🟢", "Optimal", "#28a745")
    """
    if optimal_min <= value <= optimal_max:
        return ("🟢", "Optimal", "#28a745")
    elif warning_min <= value < optimal_min or optimal_max < value <= warning_max:
        return ("🟡", "Warning", "#ffc107")
    elif value < warning_min:
        return ("🔴", "Critical Low", "#dc3545")
    else:
        return ("🔴", "Critical High", "#dc3545")


def analyze_state_and_recommend(psi: float, S: float, R: float, ET0: float, S_fc: float = 90.0, S_wp: float = 30.0, active_hazards: list = None) -> Dict[str, Any]:
    """
    Analyze current state and provide recommendations for irrigation decisions.
    
    This is the core decision support function. It evaluates the current soil and
    weather conditions and provides:
    - Status assessment (optimal/warning/critical)
    - Visual indicators (color-coded)
    - Action recommendations
    - Urgency level
    - Alerts for active hazards
    
    Args:
        psi: Soil water tension (cbar) - lower = wetter, higher = drier
        S: Soil water storage (mm)
        R: Expected rainfall (mm) - natural water input
        ET0: Reference evapotranspiration (mm/day) - water demand indicator
        S_fc: Field capacity (mm) - optimal upper limit of soil moisture
        S_wp: Wilting point (mm) - critical lower limit of soil moisture
        active_hazards: List of active hazard types (e.g., ["drought", "flood"])
    
    Returns:
        Dictionary containing:
            - psi_status, psi_color, psi_hex: Soil tension status and colors
            - S_status, S_color, S_hex: Soil moisture status and colors
            - recommendations: List of recommendation strings
            - alerts: List of alert messages (for hazards)
            - action_suggestion: Main recommended action
            - urgency: Urgency level ("normal", "medium", "high")
            - water_percentage: Available water as percentage of usable range
    """
    # Initialize return values
    recommendations = []
    alerts = []
    action_suggestion = None
    urgency = "normal"
    
    active_hazards = active_hazards or []
    
    # Optimal zones
    psi_optimal_min, psi_optimal_max = 30.0, 60.0
    psi_warning_min, psi_warning_max = 20.0, 80.0
    
    S_optimal_min, S_optimal_max = S_wp + 10, S_fc
    S_warning_min, S_warning_max = S_wp, S_fc + 10
    
    # Analyze soil tension (ψ)
    psi_color, psi_status, psi_hex = get_status_color(psi, psi_optimal_min, psi_optimal_max, psi_warning_min, psi_warning_max)
    
    if psi < psi_optimal_min:
        deficit = psi_optimal_min - psi
        if psi < 20:
            urgency = "high"
            alerts.append(f"⚠️ CRITICAL: Very low soil tension ({psi:.1f} cbar). Immediate irrigation needed!")
            action_suggestion = f"Irrigate {min(20, deficit * 0.5):.1f}-{min(20, deficit * 0.8):.1f} mm immediately"
        else:
            alerts.append(f"⚠️ Low soil tension ({psi:.1f} cbar). Consider irrigation.")
            action_suggestion = f"Consider irrigating {min(15, deficit * 0.4):.1f}-{min(20, deficit * 0.6):.1f} mm"
        recommendations.append(f"• Increase irrigation to raise soil tension to optimal range (30-60 cbar)")
    elif psi > psi_optimal_max:
        excess = psi - psi_optimal_max
        if psi > 100:
            urgency = "high"
            alerts.append(f"⚠️ CRITICAL: Very high soil tension ({psi:.1f} cbar). Crop stress likely!")
            recommendations.append(f"• Urgent irrigation needed to reduce stress")
        else:
            alerts.append(f"⚠️ High soil tension ({psi:.1f} cbar). Monitor closely.")
        recommendations.append(f"• Soil tension above optimal. Monitor crop stress levels.")
        if action_suggestion is None:
            action_suggestion = "Monitor - may need irrigation soon"
    else:
        recommendations.append(f"• Soil tension is optimal ({psi:.1f} cbar)")
        action_suggestion = "No immediate action needed - maintain current irrigation schedule"
    
    # Analyze soil moisture (S)
    S_color, S_status, S_hex = get_status_color(S, S_optimal_min, S_optimal_max, S_warning_min, S_warning_max)
    
    if S < S_optimal_min:
        deficit = S_optimal_min - S
        recommendations.append(f"• Soil moisture is {S:.1f} mm, below optimal range ({S_optimal_min:.1f}-{S_optimal_max:.1f} mm)")
        if S < S_wp:
            urgency = "high" if urgency != "high" else urgency
            alerts.append(f"🚨 CRITICAL: Soil moisture below wilting point ({S_wp:.1f} mm)!")
    elif S > S_fc:
        recommendations.append(f"• Soil moisture above field capacity. Risk of drainage/drowning.")
        if action_suggestion and "irrigat" not in action_suggestion.lower():
            action_suggestion = "Reduce irrigation - soil is saturated"
    else:
        recommendations.append(f"• Soil moisture is within acceptable range ({S:.1f} mm)")
    
    # Analyze active hazards
    hazard_icons = {
        "drought": "🌵",
        "flood": "🌊",
        "heatwave": "🔥",
        "equipment_failure": "🔧",
        "equipment_failure_warning": "⚠️",
        "water_restriction": "🚱"
    }
    
    if active_hazards:
        urgency = "high"  # Hazards always increase urgency
        for hazard in active_hazards:
            icon = hazard_icons.get(hazard, "⚠️")
            hazard_name = hazard.replace("_", " ").title()
            
            if hazard == "drought":
                alerts.append(f"{icon} DROUGHT ACTIVE: Reduced rainfall, increased ET0. Critical irrigation needed!")
                recommendations.append(f"• {icon} Drought conditions: Increase irrigation frequency and amounts")
                if action_suggestion is None or "irrigat" not in action_suggestion.lower():
                    action_suggestion = "URGENT: Maximize irrigation - drought conditions active"
            
            elif hazard == "flood":
                alerts.append(f"{icon} FLOOD WARNING: Excessive rainfall expected. Risk of waterlogging!")
                recommendations.append(f"• {icon} Flood conditions: Stop irrigation, prepare for drainage")
                action_suggestion = "STOP irrigation - flood conditions active"
            
            elif hazard == "heatwave":
                alerts.append(f"{icon} HEATWAVE: Extreme ET0. High water demand - increase irrigation!")
                recommendations.append(f"• {icon} Heatwave: Significantly increase irrigation to compensate for high ET0")
                if "irrigat" in action_suggestion.lower():
                    action_suggestion = action_suggestion.replace("irrigat", "URGENTLY irrigat")
            
            elif hazard == "equipment_failure":
                alerts.append(f"{icon} EQUIPMENT FAILURE: Irrigation system unavailable!")
                recommendations.append(f"• {icon} Equipment failure: System cannot irrigate. Monitor closely and prepare alternative solutions")
                action_suggestion = "⚠️ CANNOT IRRIGATE - equipment failure active"
            
            elif hazard == "equipment_failure_warning":
                alerts.append(f"{icon} EQUIPMENT WARNING: System failure expected in 2 days. Plan ahead!")
                recommendations.append(f"• {icon} Equipment warning: Increase irrigation now before system failure")
                if action_suggestion is None:
                    action_suggestion = "⚠️ Increase irrigation now - equipment failure coming"
            
            elif hazard == "water_restriction":
                alerts.append(f"{icon} WATER RESTRICTIONS: Limited irrigation capacity available")
                recommendations.append(f"• {icon} Water restrictions: Irrigation limited to 50%. Prioritize critical periods")
                if action_suggestion and "irrigat" in action_suggestion.lower():
                    action_suggestion += " (restricted to 50% capacity)"
    
    # Analyze weather conditions
    if R > 5.0:
        recommendations.append(f"• Significant rainfall expected ({R:.1f} mm). Reduce scheduled irrigation.")
        if action_suggestion and "irrigat" in action_suggestion.lower() and "flood" not in active_hazards:
            action_suggestion = f"Delay irrigation - {R:.1f} mm rain expected"
    
    if ET0 > 5.0:
        recommendations.append(f"• High evapotranspiration ({ET0:.1f} mm/day). Increased water demand expected.")
        if urgency == "normal" and psi < psi_optimal_min:
            urgency = "medium"
    
    # Calculate water balance
    water_balance = S - S_wp
    water_percentage = (water_balance / (S_fc - S_wp)) * 100 if (S_fc - S_wp) > 0 else 0
    
    return {
        "psi_status": psi_status,
        "psi_color": psi_color,
        "psi_hex": psi_hex,
        "S_status": S_status,
        "S_color": S_color,
        "S_hex": S_hex,
        "recommendations": recommendations,
        "alerts": alerts,
        "action_suggestion": action_suggestion,
        "urgency": urgency,
        "water_percentage": water_percentage,
    }


def display_episode_summary(episode_data: Dict[str, Any], S_fc: float = 90.0, S_wp: float = 30.0):
    """
    Display episode summary in Streamlit with decision support.
    
    Creates a comprehensive summary display showing:
    - Key metrics (total reward, yield, irrigation)
    - Final state analysis with recommendations
    - Visual status cards
    - Detailed metrics table
    
    Args:
        episode_data: Dictionary containing complete episode data
        S_fc: Field capacity (mm) for analysis thresholds
        S_wp: Wilting point (mm) for analysis thresholds
    """
    # Create 4-column layout for key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_reward = episode_data["total_reward"]
    total_irrigation = sum(episode_data["actions"])

    with col1:
        st.metric("Total Reward", f"{total_reward:.2f}")

    with col2:
        st.metric("Total Irrigation", f"{total_irrigation:.2f} mm")

    if episode_data["info"]:
        final_info = episode_data["info"][-1]
        
        with col3:
            if "yield" in final_info:
                st.metric("Final Yield", f"{final_info['yield']:.4f}")
            else:
                st.metric("Final Yield", "N/A")

        with col4:
            st.metric("Cumulative Stress", f"{final_info['cum_stress']:.2f}")

    # Decision Support Section
    st.divider()
    st.subheader("🎯 Decision Support & Recommendations")
    
    final_obs = episode_data["observations"][-1]
    final_info = episode_data["info"][-1] if episode_data["info"] else {}
    
    psi, S, R, ET0 = final_obs[0], final_obs[1], final_obs[2], final_obs[3]
    active_hazards = final_info.get("active_hazards", [])
    
    # Analyze state and get recommendations (including hazards)
    analysis = analyze_state_and_recommend(psi, S, R, ET0, S_fc, S_wp, active_hazards=active_hazards)
    
    # Display status indicators
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown(f"### {analysis['psi_color']} Soil Tension")
        st.markdown(f"**{psi:.1f} cbar** - *{analysis['psi_status']}*")
        st.markdown(f"<div style='background-color: {analysis['psi_hex']}20; padding: 10px; border-radius: 5px; border-left: 4px solid {analysis['psi_hex']};'>Optimal: 30-60 cbar</div>", unsafe_allow_html=True)
    
    with status_col2:
        st.markdown(f"### {analysis['S_color']} Soil Moisture")
        st.markdown(f"**{S:.1f} mm** - *{analysis['S_status']}*")
        st.markdown(f"<div style='background-color: {analysis['S_hex']}20; padding: 10px; border-radius: 5px; border-left: 4px solid {analysis['S_hex']};'>Available: {analysis['water_percentage']:.1f}%</div>", unsafe_allow_html=True)
    
    with status_col3:
        urgency_icon = "🔴" if analysis['urgency'] == "high" else "🟡" if analysis['urgency'] == "medium" else "🟢"
        st.markdown(f"### {urgency_icon} Action Urgency")
        urgency_text = analysis['urgency'].upper().replace("_", " ")
        st.markdown(f"**{urgency_text}**")
        st.markdown(f"<div style='padding: 10px; border-radius: 5px;'>Current conditions require <strong>{urgency_text}</strong> attention</div>", unsafe_allow_html=True)
    
    # Action Recommendation Box
    if analysis['action_suggestion']:
        urgency_color = "#dc3545" if analysis['urgency'] == "high" else "#ffc107" if analysis['urgency'] == "medium" else "#28a745"
        st.markdown(
            f"<div style='background-color: {urgency_color}15; padding: 15px; border-radius: 8px; border-left: 5px solid {urgency_color}; margin: 10px 0;'>"
            f"<h4 style='margin-top: 0; color: {urgency_color};'>💡 Recommended Action</h4>"
            f"<p style='font-size: 1.1em; margin-bottom: 0;'><strong>{analysis['action_suggestion']}</strong></p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Alerts
    if analysis['alerts']:
        st.markdown("### 🚨 Alerts")
        for alert in analysis['alerts']:
            st.warning(alert)
    
    # Recommendations
    st.markdown("### 📋 Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        for i, rec in enumerate(analysis['recommendations'][:len(analysis['recommendations'])//2 + 1]):
            st.markdown(rec)
    with rec_col2:
        for rec in analysis['recommendations'][len(analysis['recommendations'])//2 + 1:]:
            st.markdown(rec)
    
    st.divider()
    
    # Detailed metrics table
    st.subheader("📊 Final State Details")
    metrics_df = pd.DataFrame({
        "Metric": [
            "Soil Tension (ψ)",
            "Soil Moisture (S)",
            "Rain (R)",
            "ET0",
            "Day",
            "Cumulative Irrigation",
            "Cumulative Stress",
        ],
        "Value": [
            f"{psi:.2f} cbar",
            f"{S:.2f} mm",
            f"{R:.2f} mm",
            f"{ET0:.2f} mm/day",
            f"{final_info.get('day', len(episode_data['observations']))}",
            f"{final_info.get('cum_irrig', total_irrigation):.2f} mm",
            f"{final_info.get('cum_stress', 0):.2f}",
        ]
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ==================================================================
# MAIN STREAMLIT APPLICATION
# ==================================================================
def main():
    """
    Main Streamlit application entry point.
    
    This function orchestrates the entire web application:
    1. Initializes session state (persistent data across reruns)
    2. Sets up UI layout (sidebar for config, main area for results)
    3. Handles user interactions (button clicks, parameter changes)
    4. Displays results (visualizations, metrics, recommendations)
    
    Streamlit automatically reruns this function when:
    - User changes a widget (slider, button, etc.)
    - Code files are modified
    - Session state is updated programmatically
    """
    # Initialize persistent state variables (survives reruns)
    initialize_session_state()

    st.title("💧 RL Intelligent Irrigation Environment")
    st.markdown("Interactive visualization and experimentation with the irrigation RL environment")
    
    # Add info box explaining the difference
    with st.expander("ℹ️ Understanding Training vs Running Episodes", expanded=False):
        st.markdown("""
        ### 🤖 **Training PPO** (Machine Learning)
        - **What it does:** Teaches an AI agent (using PPO algorithm) how to make smart irrigation decisions
        - **How:** The agent tries many different actions, learns from rewards/penalties, and improves over time
        - **When:** One-time process that takes time (thousands of timesteps)
        - **Result:** Creates a trained model that can make intelligent decisions
        
        ### ▶️ **Running Episodes** (Simulation)
        - **What it does:** Simulates one irrigation season using a chosen decision-making policy
        - **How:** Uses a policy (rule-based or AI) to make daily irrigation decisions over 120 days
        - **When:** Quick process - simulates one season in seconds
        - **Result:** Shows how well a policy performs (rewards, yield, water usage)
        
        ### 🔗 **The Relationship:**
        - ✅ **You can run episodes WITHOUT training PPO** - use rule-based policies like:
          - `random`: Random irrigation amounts
          - `threshold`: Irrigate when soil tension exceeds threshold
          - `reactive`: Adjust irrigation based on current stress levels
          - `none`: No irrigation (baseline)
        
        - ✅ **Training PPO is OPTIONAL** - it creates a smarter AI policy
        - ✅ **If you train PPO**, you get an additional `ppo` policy option
        - ❌ **To use the `ppo` policy, you MUST train a PPO model first**
        
        ### 💡 **Typical Workflow:**
        1. **Start simple:** Run episodes with rule-based policies (no training needed)
        2. **Compare:** See how different policies perform
        3. **Train PPO:** If you want AI-powered decisions (optional, takes time)
        4. **Compare again:** See if the trained PPO model performs better
        """)

    # ==================================================================
    # SIDEBAR: Configuration Panel
    # ==================================================================
    # All user-configurable parameters are in the sidebar
    # This keeps the main area clean for visualizations
    
    # Display logo in sidebar header
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "logo.jpg")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    else:
        # Fallback: try relative path from src directory
        logo_path = os.path.join("images", "logo.jpg")
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, use_container_width=True)
    
    st.sidebar.header("⚙️ Configuration")

    # Environment parameters - core simulation settings
    st.sidebar.subheader("Environment Parameters")
    season_length = st.sidebar.slider("Season Length (days)", 30, 200, 120)
    # Season length determines how many days to simulate (typical: 120 days for a growing season)
    
    max_irrigation = st.sidebar.slider("Max Irrigation (mm/day)", 5.0, 50.0, 20.0)
    # Maximum irrigation per day - sets the action space upper bound
    
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    # Random seed for reproducibility - same seed produces same weather/random events
    # 
    # The seed is used in the following scenarios:
    # 1. Environment initialization: Sets up the random number generator (RNG)
    # 2. Weather generation: Determines the sequence of rain, ET0, and Kc values
    #    - Same seed = same weather pattern (useful for comparing policies)
    #    - Different seed = different weather (useful for robustness testing)
    # 3. Hazard generation: If hazards are enabled, seed affects which hazards occur
    #    and when they are scheduled
    # 4. Episode reset: If reset() is called with a seed, it regenerates weather
    # 
    # IMPORTANT: When comparing scenarios (e.g., Scenario 1 vs Scenario 3), use the
    # same seed to ensure fair comparison with identical weather conditions.

    # Soil parameters (optional)
    with st.sidebar.expander("Soil Parameters (Advanced)"):
        S_max = st.number_input("S_max (mm)", value=120.0)
        S_fc = st.number_input("S_fc (mm)", value=90.0)
        S_wp = st.number_input("S_wp (mm)", value=30.0)
        psi_sat = st.number_input("ψ_sat (cbar)", value=10.0)
        psi_fc = st.number_input("ψ_fc (cbar)", value=30.0)
        psi_wp = st.number_input("ψ_wp (cbar)", value=150.0)
    
    # Hazard events configuration
    with st.sidebar.expander("⚠️ Hazard Events (Disturbances)", expanded=False):
        enable_hazards = st.checkbox("Enable Hazard Events", value=False, help="Add realistic disturbances like droughts, floods, and equipment failures")
        
        if enable_hazards:
            st.markdown("**Hazard Probabilities (per season):**")
            
            drought_prob = st.slider("Drought", 0.0, 1.0, 0.2, 0.05, help="Probability of drought period")
            drought_duration = st.number_input("Drought Duration (days)", 3, 30, 7, key="drought_dur")
            
            flood_prob = st.slider("Flood", 0.0, 1.0, 0.15, 0.05, help="Probability of flood event")
            flood_duration = st.number_input("Flood Duration (days)", 1, 10, 3, key="flood_dur")
            
            heatwave_prob = st.slider("Heatwave", 0.0, 1.0, 0.1, 0.05, help="Probability of heatwave")
            heatwave_duration = st.number_input("Heatwave Duration (days)", 3, 15, 5, key="heatwave_dur")
            
            equipment_failure_prob = st.slider("Equipment Failure", 0.0, 1.0, 0.05, 0.05, help="Probability of irrigation system failure")
            equipment_failure_duration = st.number_input("Failure Duration (days)", 2, 14, 5, key="equip_dur")
            
            water_restriction_prob = st.slider("Water Restrictions", 0.0, 1.0, 0.1, 0.05, help="Probability of water use restrictions")
            water_restriction_duration = st.number_input("Restriction Duration (days)", 5, 30, 10, key="restrict_dur")
            
            hazard_cfg = {
                "enable_hazards": True,
                "drought_prob": drought_prob,
                "drought_duration": drought_duration,
                "flood_prob": flood_prob,
                "flood_duration": flood_duration,
                "heatwave_prob": heatwave_prob,
                "heatwave_duration": heatwave_duration,
                "equipment_failure_prob": equipment_failure_prob,
                "equipment_failure_duration": equipment_failure_duration,
                "water_restriction_prob": water_restriction_prob,
                "water_restriction_duration": water_restriction_duration,
            }
        else:
            hazard_cfg = {"enable_hazards": False}

    # ==================================================================
    # Policy Selection - Choose decision-making strategy
    # ==================================================================
    # Policies determine how irrigation decisions are made:
    # - Rule-based: Simple heuristics (random, threshold, reactive)
    # - Learned: PPO model trained through reinforcement learning
    
    st.sidebar.subheader("Policy Selection")
    policy_options = ["random", "none", "threshold", "reactive"]  # Base policies (no training needed)
    policy_descriptions = {
        "random": "Random irrigation amounts (no training needed) - baseline for comparison",
        "none": "No irrigation - baseline to show natural rainfall effects (no training needed)",
        "threshold": "Irrigate when tension > threshold (no training needed) - simple rule-based",
        "reactive": "Reactive irrigation based on stress (no training needed) - adaptive rule-based",
    }
    
    if PPO_AVAILABLE and st.session_state.ppo_model is not None:
        policy_options.append("ppo")
        policy_descriptions["ppo"] = "AI-trained PPO policy (requires trained model)"
    
    def format_policy_name(x):
        if x == 'ppo':
            if not PPO_AVAILABLE or st.session_state.ppo_model is None:
                return f"{x} (AI - needs training)"
            else:
                return f"{x} (AI-trained)"
        else:
            return f"{x} (rule-based)"
    
    policy_type = st.sidebar.selectbox(
        "Policy Type",
        policy_options,
        help="Select the decision-making policy. Rule-based policies (random, threshold, reactive) don't require training. PPO requires training first.",
        format_func=format_policy_name
    )
    
    # Show policy description
    if policy_type in policy_descriptions:
        st.sidebar.caption(f"💡 {policy_descriptions[policy_type]}")

    # Policy-specific parameters
    policy_kwargs = {}
    if policy_type == "threshold":
        st.sidebar.subheader("Threshold Policy Parameters")
        policy_kwargs["threshold_psi"] = st.sidebar.slider("Threshold ψ (cbar)", 30.0, 100.0, 50.0)
        policy_kwargs["irrigation_amount"] = st.sidebar.slider("Irrigation Amount (mm)", 5.0, 30.0, 15.0)

    elif policy_type == "reactive":
        st.sidebar.subheader("Reactive Policy Parameters")
        policy_kwargs["min_psi"] = st.sidebar.slider("Min ψ (cbar)", 10.0, 50.0, 30.0)
        policy_kwargs["max_psi"] = st.sidebar.slider("Max ψ (cbar)", 40.0, 100.0, 60.0)

    # ==================================================================
    # Create and Cache Environment Instance
    # ==================================================================
    # Build soil parameters dictionary from sidebar inputs
    soil_params = {
        "S_max": S_max,      # Maximum soil water storage capacity
        "S_fc": S_fc,        # Field capacity - optimal upper limit
        "S_wp": S_wp,        # Wilting point - critical lower limit
        "psi_sat": psi_sat,  # Tension at saturation (very wet)
        "psi_fc": psi_fc,    # Tension at field capacity (optimal)
        "psi_wp": psi_wp,    # Tension at wilting point (very dry)
    }

    # Cache demo instance in session state to avoid expensive recreation
    # Key includes all parameters that affect the environment
    # If parameters change, a new instance is created automatically
    demo_key = f"demo_{season_length}_{max_irrigation}_{seed}_{str(soil_params)}_{str(hazard_cfg)}"
    if demo_key not in st.session_state:
        # Create new environment instance with current configuration
        # Use shared weather generator for consistency across scenarios
        # weather_params={} ensures use of utils_weather.generate_weather (matches notebooks)
        weather_params = {}  # Empty dict triggers use of shared weather generator
        
        st.session_state[demo_key] = BasicIntelligentIrrigation(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            soil=soil_params,
            hazard_cfg=hazard_cfg,
            weather_params=weather_params  # Ensures consistent weather generation
        )
    demo = st.session_state[demo_key]  # Retrieve cached or newly created instance
    
    # Set PPO model if available
    if st.session_state.ppo_model is not None:
        demo.ppo_model = st.session_state.ppo_model

    # ==================================================================
    # MAIN CONTENT: Real-Time Decision Dashboard
    # ==================================================================
    # This section only appears if an episode has been run
    # Provides interactive decision support and analysis tools
    
    if st.session_state.current_episode:
        st.subheader("🎯 Real-Time Decision Dashboard")
        
        # Extract episode data for analysis
        observations = np.array(st.session_state.current_episode["observations"])
        # observations: [psi, S, R, ET0] at each time step
        
        actions = np.array(st.session_state.current_episode["actions"])
        # actions: Irrigation amounts (mm) applied at each step
        
        # Create tabs for different analysis views
        # Each tab provides a different perspective on the episode data
        tab1, tab2, tab3 = st.tabs(["📊 Current State", "📈 Timeline Analysis", "💡 Action Analysis"])
        
        with tab1:
            # Show latest state analysis
            if len(observations) > 0:
                info_list = st.session_state.current_episode.get("info", [])
                latest_obs = observations[-1]
                latest_action = actions[-1] if len(actions) > 0 else 0
                latest_info = info_list[-1] if info_list else {}
                latest_hazards = latest_info.get("active_hazards", [])
                
                analysis = analyze_state_and_recommend(
                    latest_obs[0], latest_obs[1], latest_obs[2], latest_obs[3], 
                    S_fc, S_wp, active_hazards=latest_hazards
                )
                
                # Create visual status cards
                card_col1, card_col2, card_col3, card_col4 = st.columns(4)
                
                with card_col1:
                    st.markdown(
                        f"<div style='text-align: center; padding: 15px; background-color: {analysis['psi_hex']}20; "
                        f"border-radius: 10px; border: 2px solid {analysis['psi_hex']};'>"
                        f"<h3>{analysis['psi_color']}</h3>"
                        f"<h2>{latest_obs[0]:.1f}</h2>"
                        f"<p><strong>Soil Tension</strong><br>{analysis['psi_status']}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                with card_col2:
                    st.markdown(
                        f"<div style='text-align: center; padding: 15px; background-color: {analysis['S_hex']}20; "
                        f"border-radius: 10px; border: 2px solid {analysis['S_hex']};'>"
                        f"<h3>{analysis['S_color']}</h3>"
                        f"<h2>{latest_obs[1]:.1f}</h2>"
                        f"<p><strong>Soil Moisture (mm)</strong><br>{analysis['S_status']}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                with card_col3:
                    st.markdown(
                        f"<div style='text-align: center; padding: 15px; background-color: #17a2b820; "
                        f"border-radius: 10px; border: 2px solid #17a2b8;'>"
                        f"<h3>💧</h3>"
                        f"<h2>{latest_action:.1f}</h2>"
                        f"<p><strong>Last Irrigation (mm)</strong><br>Day {len(observations)}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                with card_col4:
                    urgency_color = "#dc3545" if analysis['urgency'] == "high" else "#ffc107" if analysis['urgency'] == "medium" else "#28a745"
                    urgency_icon = "🔴" if analysis['urgency'] == "high" else "🟡" if analysis['urgency'] == "medium" else "🟢"
                    st.markdown(
                        f"<div style='text-align: center; padding: 15px; background-color: {urgency_color}20; "
                        f"border-radius: 10px; border: 2px solid {urgency_color};'>"
                        f"<h3>{urgency_icon}</h3>"
                        f"<h2>{analysis['urgency'].upper()}</h2>"
                        f"<p><strong>Urgency Level</strong><br>Action Required</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                # Action recommendation
                if analysis['action_suggestion']:
                    st.markdown("### 💡 Recommended Action")
                    st.info(f"**{analysis['action_suggestion']}**")
        
        with tab2:
            # Timeline of state changes
            if len(observations) > 1:
                st.markdown("### State Changes Over Time")
                info_list = st.session_state.current_episode.get("info", [])
                
                # Create a simplified timeline view
                timeline_data = []
                for i, obs in enumerate(observations):
                    day_info = info_list[i] if i < len(info_list) else {}
                    day_hazards = day_info.get("active_hazards", [])
                    day_analysis = analyze_state_and_recommend(obs[0], obs[1], obs[2], obs[3], S_fc, S_wp, active_hazards=day_hazards)
                    
                    hazard_str = ", ".join(day_hazards) if day_hazards else "None"
                    timeline_data.append({
                        "Day": i + 1,
                        "ψ (cbar)": f"{obs[0]:.1f}",
                        "S (mm)": f"{obs[1]:.1f}",
                        "Status": f"{day_analysis['psi_color']} {day_analysis['S_color']}",
                        "Action (mm)": f"{actions[i]:.1f}" if i < len(actions) else "0.0",
                        "Urgency": day_analysis['urgency'].upper(),
                        "Hazards": hazard_str
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        
        with tab3:
            # Analyze action effectiveness
            st.markdown("### Irrigation Action Analysis")
            
            if len(actions) > 0:
                total_irrigation = sum(actions)
                days_with_irrigation = sum(1 for a in actions if a > 0)
                avg_irrigation = np.mean([a for a in actions if a > 0]) if days_with_irrigation > 0 else 0
                
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    st.metric("Total Irrigation", f"{total_irrigation:.1f} mm")
                
                with action_col2:
                    st.metric("Days Irrigated", f"{days_with_irrigation} / {len(actions)}")
                
                with action_col3:
                    st.metric("Average per Day", f"{avg_irrigation:.1f} mm" if avg_irrigation > 0 else "N/A")
                
                # Action recommendations
                info_list = st.session_state.current_episode.get("info", [])
                final_obs = observations[-1]
                final_info = info_list[-1] if info_list else {}
                final_hazards = final_info.get("active_hazards", [])
                final_analysis = analyze_state_and_recommend(
                    final_obs[0], final_obs[1], final_obs[2], final_obs[3], S_fc, S_wp, active_hazards=final_hazards
                )
                
                st.markdown("### 📋 Recommendations")
                for rec in final_analysis['recommendations']:
                    st.markdown(f"• {rec}")
        
        st.divider()
    
    # ==================================================================
    # MAIN CONTENT AREA: Two-Column Layout
    # ==================================================================
    # Left column (3/4 width): Episode visualization and results
    # Right column (1/4 width): Action buttons and controls
    
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Current Episode")
        
        # Show placeholder/instructions if no episode has been run yet
        # This provides guidance to first-time users
        if not st.session_state.current_episode:
            st.info("👈 **Get Started:** Use the 'Actions' panel on the right to run an episode. You can choose from rule-based policies (no training needed) or train a PPO model for AI-powered decisions.")
            st.markdown("""
            **Quick Guide:**
            1. **Select a Policy** in the sidebar (random, threshold, reactive, or none - no training needed!)
            2. **Click "▶️ Run Episode"** in the Actions panel to simulate one irrigation season
            3. **View Results** here - episode metrics, plots, and decision insights will appear
            4. **Optional:** Train a PPO model for AI-powered irrigation decisions
            """)

    with col2:
        st.subheader("Actions")
        
        # ==================================================================
        # PPO TRAINING SECTION
        # ==================================================================
        # Allows users to train an AI agent using Proximal Policy Optimization
        # Training creates a learned policy that can outperform rule-based policies
        
        # PPO Training Section (only shown if PPO dependencies are available)
        if PPO_AVAILABLE:
            with st.expander("🤖 PPO Training (Optional - Creates AI Policy)", expanded=False):
                st.caption("💡 Train an AI agent to learn optimal irrigation decisions. This is optional - you can run episodes with rule-based policies without training.")
                total_timesteps = st.number_input(
                    "Training Timesteps",
                    min_value=1000,
                    max_value=500000,
                    value=30000,
                    step=1000,
                    help="Number of timesteps for PPO training. More timesteps = better learning but takes longer."
                )
                
                if st.button("🚀 Train PPO Model", use_container_width=True, type="primary"):
                    training_placeholder = st.empty()
                    with training_placeholder.container():
                        st.info("⏳ Starting PPO training... This may take a while.")
                        
                    try:
                        # Create trainer
                        trainer = integrate_with_basic(demo)
                        trainer.create_environment()
                        
                        # Train model (verbose output will be captured)
                        with training_placeholder.container():
                            st.info(f"⏳ Training PPO model for {total_timesteps:,} timesteps...")
                        
                        model = trainer.train(
                            total_timesteps=int(total_timesteps),
                            verbose=1  # Show training progress
                        )
                        
                        st.session_state.ppo_model = model
                        st.session_state.ppo_trainer = trainer
                        
                        training_placeholder.empty()
                        st.success(f"✅ PPO model trained successfully with {total_timesteps:,} timesteps!")
                        st.balloons()
                        
                        # Evaluate the model
                        with st.spinner("Evaluating trained model..."):
                            eval_results = trainer.evaluate(n_episodes=5, deterministic=True)
                            
                            # Format evaluation results
                            yield_text = f"{eval_results['mean_yield']:.4f}" if eval_results['mean_yield'] else "N/A"
                            if eval_results['mean_yield']:
                                yield_text += f" ± {eval_results['std_yield']:.4f}"
                            
                            eval_text = (
                                f"**Evaluation Results:**\n\n"
                                f"• Mean Reward: {eval_results['mean_total_reward']:.2f} ± {eval_results['std_total_reward']:.2f}\n"
                                f"• Mean Yield: {yield_text}\n"
                                f"• Mean Irrigation: {eval_results['mean_total_irrigation']:.2f} ± {eval_results['std_total_irrigation']:.2f} mm"
                            )
                            st.info(eval_text)
                        
                        st.rerun()  # Refresh to update policy options
                            
                    except Exception as e:
                        training_placeholder.empty()
                        st.error(f"❌ Training failed: {str(e)}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
                
                if st.session_state.ppo_model is not None:
                    st.success("✅ PPO model available")
                    if st.button("🗑️ Clear PPO Model", use_container_width=True):
                        st.session_state.ppo_model = None
                        st.session_state.ppo_trainer = None
                        demo.ppo_model = None
                        st.rerun()
                else:
                    st.info("No PPO model trained yet")
        
        st.divider()
        
        # ==================================================================
        # EPISODE RUNNING BUTTONS
        # ==================================================================
        # These buttons trigger episode simulations
        # Episodes run quickly (seconds) and show results immediately
        
        st.caption("💡 Simulate one irrigation season. Works with any policy - training PPO is optional!")
        
        # Primary button: Run a single episode
        # This is the main action users take to see policy performance
        if st.button("▶️ Run Episode", type="primary", use_container_width=True):
            # Ensure PPO model is set if using PPO policy
            if policy_type == "ppo":
                if st.session_state.ppo_model is None:
                    st.error("Please train a PPO model first!")
                    st.stop()
                demo.ppo_model = st.session_state.ppo_model
            
            with st.spinner("Running episode..."):
                episode_data = run_episode_with_policy(
                    demo, policy_type, **policy_kwargs
                )
                st.session_state.current_episode = episode_data
                st.session_state.episode_history.append(episode_data.copy())

        # Secondary button: Run multiple episodes for statistical analysis
        # Runs 10 episodes and shows aggregated statistics
        # Useful for evaluating policy performance with variance
        if st.button("🔄 Run 10 Episodes", use_container_width=True):
            # Ensure PPO model is set if using PPO policy
            if policy_type == "ppo":
                if st.session_state.ppo_model is None:
                    st.error("Please train a PPO model first!")
                    st.stop()
                demo.ppo_model = st.session_state.ppo_model
                
            with st.spinner("Running 10 episodes..."):
                stats = demo.run_multiple_episodes(
                    n_episodes=10, policy=policy_type, **policy_kwargs
                )
                st.session_state.stats = stats

        # Utility button: Clear all episode history
        # Useful for starting fresh or when switching between different configurations
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.episode_history = []  # Clear all stored episodes
            st.session_state.current_episode = None  # Clear current episode display

        # Display current episode (col1 is already set up above)
    # ==================================================================
    # EPISODE DISPLAY: Show results when episode data exists
    # ==================================================================
    if st.session_state.current_episode:
        # Check for and display hazard events that occurred during the episode
        # Hazards are shown as visual indicators and alerts
        episode_info = st.session_state.current_episode.get("info", [])
        if episode_info and any(info.get("active_hazards") for info in episode_info):
            hazard_days = []
            for i, info in enumerate(episode_info):
                if info.get("active_hazards"):
                    hazard_days.append((i + 1, info["active_hazards"]))
            
            if hazard_days:
                st.subheader("⚠️ Hazard Events Occurred")
                hazard_cols = st.columns(min(4, len(set([h[0] for h in hazard_days]))))
                for idx, (day, hazards) in enumerate(hazard_days[:8]):  # Show first 8
                    with hazard_cols[idx % len(hazard_cols)]:
                        hazard_icons = {
                            "drought": "🌵",
                            "flood": "🌊",
                            "heatwave": "🔥",
                            "equipment_failure": "🔧",
                            "equipment_failure_warning": "⚠️",
                            "water_restriction": "🚱"
                        }
                        hazard_names = ", ".join([f"{hazard_icons.get(h, '⚠️')} {h.replace('_', ' ').title()}" for h in hazards])
                        st.markdown(f"**Day {day}:** {hazard_names}")
                
                # Show final summary
                final_info = episode_info[-1] if episode_info else {}
                if "hazard_history" in final_info:
                    st.info(f"📋 **Hazard Summary:** {len(final_info['hazard_history'])} hazard event(s) occurred during this season")
        
        display_episode_summary(st.session_state.current_episode, S_fc=S_fc, S_wp=S_wp)
        
        # Plot visualizations with tabs for different views
        st.subheader("📈 Episode Visualization")
        
        viz_tab1, viz_tab2 = st.tabs(["📊 Detailed Analysis", "🌊 Water Balance View"])
        
        with viz_tab1:
            st.caption("Comprehensive view with decision zones, rewards, and weather analysis")
            fig = plot_episode_metrics(
                st.session_state.current_episode,
                S_fc=S_fc,
                S_wp=S_wp
            )
            st.pyplot(fig)
            plt.close()
        
        with viz_tab2:
            st.caption("Research-style visualization focusing on water balance dynamics (ψ, S, I+R, ETc+D)")
            fig2 = plot_episode_metrics_notebook_style(
                st.session_state.current_episode,
                S_fc=S_fc,
                S_wp=S_wp
            )
            st.pyplot(fig2)
            plt.close()

    # Display statistics if available
    if "stats" in st.session_state:
        st.subheader("📊 Statistics (10 Episodes)")
        stats = st.session_state.stats
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric(
                "Mean Reward",
                f"{stats['mean_total_reward']:.2f}",
                delta=f"±{stats['std_total_reward']:.2f}"
            )
        with stat_col2:
            if stats['mean_yield'] is not None:
                st.metric(
                    "Mean Yield",
                    f"{stats['mean_yield']:.4f}",
                    delta=f"±{stats['std_yield']:.4f}"
                )
            else:
                st.metric("Mean Yield", "N/A")
        with stat_col3:
            st.metric(
                "Mean Irrigation",
                f"{stats['mean_total_irrigation']:.2f} mm",
                delta=f"±{stats['std_total_irrigation']:.2f} mm"
            )

    # Episode history with insights
    if st.session_state.episode_history:
        st.subheader("📜 Episode History & Insights")
        
        history_data = []
        for i, ep in enumerate(st.session_state.episode_history):
            final_obs = ep["observations"][-1] if ep["observations"] else [0, 0, 0, 0]
            final_info = ep["info"][-1] if ep["info"] else {}
            
            # Quick analysis for each episode
            ep_analysis = analyze_state_and_recommend(
                final_obs[0], final_obs[1], final_obs[2], final_obs[3], S_fc, S_wp
            )
            
            history_data.append({
                "Episode": i + 1,
                "Total Reward": f"{ep['total_reward']:.2f}",
                "Total Irrigation": f"{sum(ep['actions']):.1f} mm",
                "Final Yield": f"{final_info.get('yield', 0):.4f}" if final_info.get('yield') else "N/A",
                "Cumulative Stress": f"{final_info.get('cum_stress', 0):.2f}",
                "Final Status": f"{ep_analysis['psi_color']} {ep_analysis['S_color']}",
                "Urgency": ep_analysis['urgency'].upper(),
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Performance insights
        if len(st.session_state.episode_history) > 1:
            st.markdown("### 📊 Performance Insights")
            
            rewards = [ep["total_reward"] for ep in st.session_state.episode_history]
            irrigations = [sum(ep["actions"]) for ep in st.session_state.episode_history]
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                best_ep = np.argmax(rewards) + 1
                st.info(f"🏆 **Best Episode:** #{best_ep} with reward {max(rewards):.2f}")
            
            with insight_col2:
                avg_irrigation = np.mean(irrigations)
                efficiency_ep = np.argmin(irrigations) + 1
                st.info(f"💧 **Most Efficient:** Episode #{efficiency_ep} used {min(irrigations):.1f} mm")
            
            with insight_col3:
                avg_reward = np.mean(rewards)
                improvement = max(rewards) - min(rewards)
                st.info(f"📈 **Improvement Range:** {improvement:.2f} (avg: {avg_reward:.2f})")
            
            # Trend analysis
            if len(rewards) >= 3:
                trend = "📈 Improving" if rewards[-1] > rewards[0] else "📉 Declining" if rewards[-1] < rewards[0] else "➡️ Stable"
                st.markdown(f"**Trend:** {trend} - Latest reward is {abs(rewards[-1] - rewards[0]):.2f} {'higher' if rewards[-1] > rewards[0] else 'lower'} than first")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        RL Intelligent Irrigation Environment | Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
