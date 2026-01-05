"""
Gradio UI configuration sections (parallel to Streamlit UI modules).
"""

from .environment_config import build_environment_config
from .soil_config import build_soil_and_tension_config
from .weather_config import build_weather_config
from .ppo_config import build_ppo_training_section
from .mlp_config import build_mlp_policy_config
from .goal_programming_config import build_goal_programming_config

__all__ = [
    "build_environment_config",
    "build_soil_and_tension_config",
    "build_weather_config",
    "build_ppo_training_section",
    "build_mlp_policy_config",
    "build_goal_programming_config",
]
