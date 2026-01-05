"""
Basic standalone script for RL Intelligent Irrigation Environment.

This module provides a simplified interface to run and test the irrigation
environment without additional dependencies beyond the core environment.
Useful for quick testing, debugging, and understanding basic functionality.
"""

import numpy as np
from typing import Optional, Dict, Any
from src.utils_env_modeles import IrrigationEnvPhysical


class BasicIntelligentIrrigation:
    """
    Basic demonstration and testing interface for the irrigation environment.
    
    This class provides a simplified wrapper around IrrigationEnvPhysical that makes
    it easy to run episodes with various policies (rule-based or learned) without
    needing to understand the full Gymnasium API.
    
    Key Features:
    - Run single or multiple episodes with different policies
    - Support for rule-based policies (random, threshold, reactive, none)
    - Support for learned policies (PPO model)
    - Episode history tracking
    - Statistics aggregation for multiple episodes
    
    Typical Usage:
        demo = BasicIntelligentIrrigation(season_length=120, max_irrigation=20.0)
        episode_data = demo.run_episode(policy="threshold", threshold_psi=50.0)
        demo.print_episode_summary(episode_data)
    """

    def __init__(
        self,
        season_length: int = 120,
        max_irrigation: float = 20.0,
        seed: int = 0,
        ppo_model=None,
        hazard_cfg: Optional[Dict[str, Any]] = None,
        **env_kwargs
    ):
        """
        Initialize the basic irrigation.

        Args:
            season_length: Length of irrigation season in days (default: 120)
                Typical growing season length for most crops
            max_irrigation: Maximum irrigation per day (mm) (default: 20.0)
                Sets the upper bound of the action space
            seed: Random seed for reproducibility (default: 0)
                Same seed produces same weather patterns and random events
            ppo_model: Optional trained PPO model for "ppo" policy (default: None)
                If provided, enables use of learned AI policy
            hazard_cfg: Optional hazard events configuration (default: None)
                Dictionary configuring drought, flood, heatwave, etc.
            **env_kwargs: Additional arguments passed to IrrigationEnvPhysical
                Can include soil parameters, reward config, residual models, etc.
        """
        # Handle hazard_cfg parameter: can be passed directly or via env_kwargs
        # This allows flexibility in how the configuration is provided
        if hazard_cfg is None:
            hazard_cfg = env_kwargs.pop("hazard_cfg", None)
        
        # Create the underlying Gymnasium environment
        # This is the core simulation engine that models soil-plant-water dynamics
        self.env = IrrigationEnvPhysical(
            season_length=season_length,
            max_irrigation=max_irrigation,
            seed=seed,
            hazard_cfg=hazard_cfg,
            **env_kwargs  # Pass any additional configuration (soil, reward, etc.)
        )
        
        # Episode history: stores all completed episodes for later analysis
        # Useful for comparing different policies or parameter settings
        self.episode_history = []
        
        # PPO model: trained reinforcement learning agent (optional)
        # If set, enables "ppo" policy option for intelligent decision-making
        self.ppo_model = ppo_model

    def run_episode(self, policy: str = "random", **policy_kwargs) -> Dict[str, Any]:
        """
        Run a single episode with a specified policy.
        
        Simulates one complete irrigation season from day 0 to season_length.
        At each day, the policy selects an irrigation action, and the environment
        updates the soil state based on weather and plant water use.

        Args:
            policy: Policy type determining how irrigation decisions are made
                - "random": Random irrigation amounts (baseline)
                - "none": No irrigation (natural baseline)
                - "threshold": Irrigate when tension exceeds threshold
                - "reactive": Adaptive irrigation based on stress level
                - "ppo": Learned policy from trained PPO model
            **policy_kwargs: Additional arguments specific to each policy
                - For "threshold": threshold_psi, irrigation_amount
                - For "reactive": min_psi, max_psi
                - For "ppo": deterministic (bool)

        Returns:
            Dictionary containing complete episode data:
                - observations: List of [psi, S, R, ET0] at each step
                - actions: List of irrigation amounts (mm) applied
                - rewards: List of daily rewards received
                - info: List of info dicts with metrics (ETc, D, hazards, etc.)
                - total_reward: Sum of all daily rewards plus terminal reward
        """
        # Reset environment to initial state (day 0, soil at field capacity)
        # Returns initial observation [psi_0, S_0, R_0, ET0_0]
        obs, info = self.env.reset()
        done = False  # Episode continues until season ends
        
        # Initialize data structure to store episode trajectory
        episode_data = {
            "observations": [],  # Store state at each step
            "actions": [],       # Store actions taken at each step
            "rewards": [],       # Store rewards received at each step
            "info": [],          # Store additional metrics (ETc, D, hazards, etc.)
            "total_reward": 0.0, # Cumulative reward (updated as we go)
        }

        # Main episode loop: run until season is complete
        while not done:
            # Select action based on current observation and policy
            # This is where the policy logic is applied (see _select_action method)
            action = self._select_action(obs, policy, **policy_kwargs)

            # Execute action in environment
            # Environment simulates: irrigation → water balance → plant response → new state
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated  # Episode ends when season completes

            # Store step data for analysis and visualization
            episode_data["observations"].append(obs.copy())  # Current state before action
            # Convert action to float (handles array or scalar input)
            action_value = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            episode_data["actions"].append(action_value)
            episode_data["rewards"].append(float(reward))  # Daily reward
            episode_data["info"].append(info.copy())  # Detailed step information
            episode_data["total_reward"] += reward  # Accumulate total reward

            # Update observation for next iteration
            obs = next_obs

        return episode_data

    def _select_action(
        self, obs: np.ndarray, policy: str, **kwargs
    ) -> np.ndarray:
        """
        Select action based on policy type.

        Args:
            obs: Current observation [psi, S, R, ET0]
            policy: Policy type
            **kwargs: Policy-specific parameters

        Returns:
            Action array [irrigation_mm]
        """
        psi = obs[0]
        S = obs[1]
        R = obs[2]
        ET0 = obs[3]

        if policy == "random":
            # Random irrigation policy
            # Applies random irrigation between 0 and I_max (uniform distribution)
            # Used as baseline for comparison - no intelligence, just random actions
            action = np.array([self.env.I_max * np.random.random()])

        elif policy == "none":
            # No irrigation policy (baseline)
            # Always applies zero irrigation - shows what happens with natural rainfall only
            # Useful for establishing baseline performance
            action = np.array([0.0])

        elif policy == "threshold":
            # Threshold policy: Simple rule-based irrigation
            # Logic: If soil tension exceeds threshold (soil is too dry), irrigate
            # This is a simple "on/off" control strategy
            threshold_psi = kwargs.get("threshold_psi", 50.0)  # Tension threshold (cbar)
            irrigation_amount = kwargs.get("irrigation_amount", 15.0)  # Fixed irrigation (mm)
            
            if psi > threshold_psi:
                # Soil is dry (high tension) - apply irrigation
                action = np.array([irrigation_amount])
            else:
                # Soil is wet enough (low tension) - no irrigation needed
                action = np.array([0.0])

        elif policy == "reactive":
            # Reactive policy: Adaptive rule-based irrigation
            # More sophisticated than threshold - adjusts irrigation amount based on stress level
            # Uses proportional control: more stress → more irrigation
            min_psi = kwargs.get("min_psi", 30.0)  # Lower bound of optimal zone (cbar)
            max_psi = kwargs.get("max_psi", 60.0)  # Upper bound of optimal zone (cbar)
            
            if psi > max_psi:
                # High stress (too dry): Apply irrigation proportional to deficit
                # deficit = how far above optimal zone
                # irrigation = deficit * 0.3 (proportional response factor)
                # Cap deficit at 50.0 to prevent excessive irrigation
                deficit = min(psi - max_psi, 50.0)
                action = np.array([min(self.env.I_max, deficit * 0.3)])
            elif psi < min_psi:
                # Over-irrigated (too wet): No irrigation needed
                action = np.array([0.0])
            else:
                # Comfort zone (optimal): No irrigation needed
                # Soil conditions are already in the optimal range
                action = np.array([0.0])

        elif policy == "ppo":
            # PPO policy: Learned intelligent policy from reinforcement learning
            # Uses a trained Proximal Policy Optimization (PPO) model
            # The model has learned optimal irrigation decisions through trial and error
            if self.ppo_model is None:
                raise ValueError("PPO model not provided. Set ppo_model in __init__ or use a different policy.")
            
            # deterministic=True: Use mean action (consistent, good for evaluation)
            # deterministic=False: Sample from policy distribution (stochastic, more exploration)
            deterministic = kwargs.get("deterministic", True)
            action, _ = self.ppo_model.predict(obs, deterministic=deterministic)
            
            # Ensure action is in correct format (numpy array with shape (1,))
            # Handles different return types from different PPO implementations
            if not isinstance(action, np.ndarray):
                action = np.array([float(action)])
            elif action.ndim == 0:
                action = np.array([float(action)])

        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Clip action to valid range [0, I_max] to ensure it's within environment bounds
        # This is a safety check in case policy returns invalid values
        return np.clip(action, 0.0, self.env.I_max)

    def run_multiple_episodes(
        self, n_episodes: int = 10, policy: str = "random", **policy_kwargs
    ) -> Dict[str, Any]:
        """
        Run multiple episodes and collect aggregated statistics.
        
        Useful for evaluating policy performance with statistical significance.
        Each episode uses different random weather (via seed variation) to test
        robustness across varying conditions.

        Args:
            n_episodes: Number of episodes to run (default: 10)
                More episodes = more reliable statistics but takes longer
            policy: Policy type to evaluate
            **policy_kwargs: Policy-specific parameters

        Returns:
            Dictionary containing:
                - episodes: List of all episode data dictionaries
                - mean_total_reward: Average total reward across episodes
                - std_total_reward: Standard deviation of total rewards
                - mean_yield: Average final crop yield (if available)
                - std_yield: Standard deviation of yields
                - mean_total_irrigation: Average total irrigation applied
                - std_total_irrigation: Standard deviation of irrigation
        """
        # Initialize lists to collect data from all episodes
        all_episodes = []  # Store complete data for each episode
        total_rewards = []  # Collect total rewards for statistics
        final_yields = []  # Collect final yields for statistics
        total_irrigation = []  # Collect total irrigation for statistics

        # Run each episode
        # Note: Each episode gets a different random seed (implicitly via environment reset)
        # This ensures varied weather conditions across episodes
        for i in range(n_episodes):
            # Run a single episode with the specified policy
            episode_data = self.run_episode(policy=policy, **policy_kwargs)
            all_episodes.append(episode_data)  # Store complete episode data
            
            # Extract metrics for statistical analysis
            total_rewards.append(episode_data["total_reward"])
            
            # Extract final yield from episode info (only available at season end)
            if episode_data["info"] and "yield" in episode_data["info"][-1]:
                final_yields.append(episode_data["info"][-1]["yield"])
            
            # Calculate total irrigation: sum of all daily irrigation actions
            total_irrigation.append(sum(episode_data["actions"]))

        # Calculate and return aggregated statistics
        return {
            "episodes": all_episodes,  # All episode data for detailed analysis
            "mean_total_reward": np.mean(total_rewards),  # Average performance
            "std_total_reward": np.std(total_rewards),  # Performance variability
            "mean_yield": np.mean(final_yields) if final_yields else None,  # Average yield
            "std_yield": np.std(final_yields) if final_yields else None,  # Yield variability
            "mean_total_irrigation": np.mean(total_irrigation),  # Average water usage
            "std_total_irrigation": np.std(total_irrigation),  # Water usage variability
        }

    def print_episode_summary(self, episode_data: Dict[str, Any]):
        """
        Print a human-readable summary of an episode.
        
        Displays key metrics including:
        - Total reward (performance indicator)
        - Total irrigation (water usage)
        - Final yield (crop productivity)
        - Final soil state (moisture and tension)
        - Cumulative stress (plant health indicator)

        Args:
            episode_data: Episode data dictionary returned by run_episode()
                Must contain: total_reward, actions, info
        """
        # Print header separator
        print("\n" + "=" * 60)
        print("EPISODE SUMMARY")
        print("=" * 60)
        
        # Basic metrics available in all episodes
        print(f"Total Reward: {episode_data['total_reward']:.2f}")
        # Total reward combines daily rewards (stress, irrigation cost) + terminal reward (yield)
        
        print(f"Total Irrigation: {sum(episode_data['actions']):.2f} mm")
        # Total water applied over the entire season
        
        # Extract final state information (available at episode end)
        if episode_data["info"]:
            final_info = episode_data["info"][-1]  # Last info dict contains final state
            
            # Final yield (only calculated at season end)
            if "yield" in final_info:
                print(f"Final Yield: {final_info['yield']:.4f}")
                # Yield is normalized (0-1) and depends on cumulative stress
            
            # Final soil state
            print(f"Final Soil Moisture (S): {final_info['S']:.2f} mm")
            # Soil water content at end of season
            
            print(f"Final Tension (ψ): {final_info['psi']:.2f} cbar")
            # Soil water tension at end of season (indicates water availability)
            
            print(f"Cumulative Stress: {final_info['cum_stress']:.2f}")
            # Sum of daily stress values - higher stress = lower yield
        
        # Print footer separator
        print("=" * 60 + "\n")


def main():
    """
    Example usage of the basic irrigation demo.
    
    This function demonstrates how to use the BasicIntelligentIrrigation class
    to test different irrigation policies and compare their performance.
    
    It shows:
    1. Creating a demo instance
    2. Running episodes with different policies
    3. Viewing episode summaries
    4. Running multiple episodes for statistical analysis
    """
    print("Running Basic Irrigation Environment Demo")
    print("-" * 60)

    # Create demo instance with standard parameters
    # season_length=120: Typical growing season (4 months)
    # max_irrigation=20.0: Maximum 20mm/day irrigation capacity
    # seed=42: Fixed seed for reproducibility (same weather every run)
    demo = BasicIntelligentIrrigation(
        season_length=120,
        max_irrigation=20.0,
        seed=42
    )

    # Test different policies to compare their performance
    # Each policy represents a different decision-making strategy
    policies = [
        ("none", "No Irrigation"),        # Baseline: natural rainfall only
        ("random", "Random Policy"),      # Random actions (no intelligence)
        ("threshold", "Threshold Policy"), # Simple rule-based
        ("reactive", "Reactive Policy"),  # Adaptive rule-based
    ]

    # Run one episode with each policy and display results
    for policy_name, policy_desc in policies:
        print(f"\nTesting: {policy_desc} ({policy_name})")
        print("-" * 60)
        
        # Run episode with policy-specific parameters
        if policy_name == "threshold":
            # Threshold policy: irrigate when psi > 50 cbar, apply 15mm
            episode_data = demo.run_episode(
                policy=policy_name,
                threshold_psi=50.0,       # Tension threshold (cbar)
                irrigation_amount=15.0    # Fixed irrigation amount (mm)
            )
        elif policy_name == "reactive":
            # Reactive policy: maintain optimal zone between 30-60 cbar
            episode_data = demo.run_episode(
                policy=policy_name,
                min_psi=30.0,  # Lower bound of optimal zone (cbar)
                max_psi=60.0   # Upper bound of optimal zone (cbar)
            )
        else:
            # Simple policies (none, random) don't need additional parameters
            episode_data = demo.run_episode(policy=policy_name)
        
        # Display summary of episode results
        demo.print_episode_summary(episode_data)

    # Run multiple episodes for statistical analysis
    # This provides more reliable performance estimates by averaging across
    # multiple weather scenarios
    print("\n" + "=" * 60)
    print("RUNNING MULTIPLE EPISODES (Statistics)")
    print("=" * 60)
    
    # Run 10 episodes with threshold policy and calculate statistics
    stats = demo.run_multiple_episodes(
        n_episodes=10,
        policy="threshold",
        threshold_psi=50.0,
        irrigation_amount=15.0
    )
    
    # Display statistical summary
    # Mean ± Std shows average performance and variability
    print(f"Mean Total Reward: {stats['mean_total_reward']:.2f} ± {stats['std_total_reward']:.2f}")
    # Higher reward is better (less stress, good yield, efficient water use)
    
    if stats['mean_yield'] is not None:
        print(f"Mean Yield: {stats['mean_yield']:.4f} ± {stats['std_yield']:.4f}")
        # Yield is normalized (0-1), higher is better
    
    print(f"Mean Total Irrigation: {stats['mean_total_irrigation']:.2f} ± {stats['std_total_irrigation']:.2f} mm")
    # Lower irrigation with same yield = more efficient policy


if __name__ == "__main__":
    main()
