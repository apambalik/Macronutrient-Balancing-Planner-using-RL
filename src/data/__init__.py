"""
Data management for meal planning.
"""

from .user_profile import UserProfile, UserProfileManager, ActivityLevel, DietaryRestriction, NutritionalGoals
from .nutritional_database import NutritionalDatabase, FoodItem

__all__ = [
    'UserProfile',
    'UserProfileManager',
    'ActivityLevel',
    'DietaryRestriction',
    'NutritionalGoals',
    'NutritionalDatabase',
    'FoodItem'
] 