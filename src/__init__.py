"""
Macronutrient Balancing Planner using Reinforcement Learning

A comprehensive system for meal planning that uses deep reinforcement learning
to optimize macronutrient balance while considering user preferences and nutritional quality.
"""

__version__ = "1.0.0"
__author__ = "ML Research Team"

from .config import Config
from .data_loader import OpenFoodFactsAPI, NutritionData, FoodItem, RealFoodDatabase
from .user_profile import UserProfile
from .evaluation import EvaluationMetrics, BaselineComparison

# Import optional modules if available
try:
    from .environment import MealPlanningEnvironment
    ENVIRONMENT_AVAILABLE = True
except ImportError:
    MealPlanningEnvironment = None
    ENVIRONMENT_AVAILABLE = False

try:
    from .agent import PPOAgent
    AGENT_AVAILABLE = True
except ImportError:
    PPOAgent = None
    AGENT_AVAILABLE = False

__all__ = [
    "Config",
    "OpenFoodFactsAPI",
    "NutritionData", 
    "FoodItem",
    "RealFoodDatabase",
    "UserProfile",
    "EvaluationMetrics",
    "BaselineComparison"
]

# Add optional components if available
if ENVIRONMENT_AVAILABLE:
    __all__.append("MealPlanningEnvironment")

if AGENT_AVAILABLE:
    __all__.append("PPOAgent") 