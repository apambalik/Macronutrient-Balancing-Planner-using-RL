"""
Reinforcement Learning Environment for Meal Planning.

This module implements the RL environment for the macronutrient balancing planner.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MealState:
    """Represents the current state of a meal planning episode"""
    current_nutrition: Dict[str, float]
    meals_consumed: List[Dict]
    step: int
    target_nutrition: Dict[str, float]
    remaining_meals: int


class MealPlanningEnvironment(gym.Env):
    """Gymnasium-compatible RL environment for meal planning"""
    
    def __init__(self, food_database, user_profile, config):
        super().__init__()
        
        self.food_database = food_database
        self.user_profile = user_profile
        self.config = config
        
        # Environment parameters
        self.max_steps_per_episode = config.env_max_steps_per_episode
        self.max_daily_meals = config.env_max_daily_meals
        
        # Get available foods (limit to reasonable number for action space)
        self.available_foods = list(self.food_database.foods.values())[:1000]
        self.food_ids = [food.id for food in self.available_foods]
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Initialize state
        self.state = None
        self.current_step = 0
        self.episode_meals = []
        self.total_nutrition = {'calories': 0, 'protein': 0, 'carbohydrates': 0, 'fat': 0}
        
        # Reward weights
        self.reward_weights = config.env_reward_weights
        self.penalties = config.env_penalties
        
        logger.info(f"Environment initialized with {len(self.available_foods)} foods")
    
    def _define_spaces(self):
        """Define observation and action spaces"""
        # Observation space: [current_nutrition(4), target_nutrition(4), progress(3), constraints(3), history(6)]
        # Total: 20 dimensions
        obs_dim = 20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: [food_selection_index(normalized), portion_size(normalized)]
        # food_selection: 0-1 (will be mapped to food index)
        # portion_size: 0-1 (will be mapped to 50-200g)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        logger.debug(f"Observation space: {self.observation_space}")
        logger.debug(f"Action space: {self.action_space}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.current_step = 0
        self.episode_meals = []
        self.total_nutrition = {'calories': 0, 'protein': 0, 'carbohydrates': 0, 'fat': 0}
        
        # Get target nutrition
        target_calories = self.user_profile.goals['target_calories']
        target_protein, target_carbs, target_fat = self.user_profile.get_target_macro_grams()
        
        self.target_nutrition = {
            'calories': target_calories,
            'protein': target_protein,
            'carbohydrates': target_carbs,
            'fat': target_fat
        }
        
        # Create initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Parse action
        food_selection, portion_size = action
        
        # Map action to actual food and portion
        food_idx = int(food_selection * len(self.available_foods))
        food_idx = max(0, min(food_idx, len(self.available_foods) - 1))
        selected_food = self.available_foods[food_idx]
        
        # Map portion size (0-1 -> 50-200g)
        actual_portion = 50 + portion_size * 150  # 50-200g range
        
        # Check if food is compatible with user preferences
        if not self.user_profile.is_food_compatible(selected_food):
            # Penalty for selecting incompatible food
            reward = -0.5
            observation = self._get_observation()
            info = {'selected_food': selected_food, 'portion_size': actual_portion, 'penalty': 'incompatible_food'}
            
            # Check termination
            done = self.current_step >= self.max_steps_per_episode
            truncated = len(self.episode_meals) >= self.max_daily_meals
            
            return observation, reward, done, truncated, info
        
        # Add nutrition from selected food
        portion_multiplier = actual_portion / 100  # Convert to per-100g multiplier
        
        meal_nutrition = {
            'calories': selected_food.nutrition.calories * portion_multiplier,
            'protein': selected_food.nutrition.protein * portion_multiplier,
            'carbohydrates': selected_food.nutrition.carbohydrates * portion_multiplier,
            'fat': selected_food.nutrition.fat * portion_multiplier
        }
        
        # Update total nutrition
        for key in self.total_nutrition:
            self.total_nutrition[key] += meal_nutrition[key]
        
        # Add meal to episode history
        self.episode_meals.append({
            'food': selected_food,
            'portion_size': actual_portion,
            'nutrition': meal_nutrition,
            'step': self.current_step
        })
        
        # Calculate reward
        reward = self._calculate_reward(selected_food, actual_portion)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check termination conditions
        done = self.current_step >= self.max_steps_per_episode
        truncated = len(self.episode_meals) >= self.max_daily_meals
        
        # Prepare info
        info = {
            'selected_food': selected_food,
            'portion_size': actual_portion,
            'meal_nutrition': meal_nutrition,
            'total_nutrition': self.total_nutrition.copy(),
            'meals_count': len(self.episode_meals)
        }
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        # Current nutrition (normalized by targets)
        current_nutrition = np.array([
            self.total_nutrition['calories'] / self.target_nutrition['calories'],
            self.total_nutrition['protein'] / self.target_nutrition['protein'],
            self.total_nutrition['carbohydrates'] / self.target_nutrition['carbohydrates'],
            self.total_nutrition['fat'] / self.target_nutrition['fat']
        ])
        
        # Target nutrition (normalized)
        target_nutrition = np.array([1.0, 1.0, 1.0, 1.0])  # Always 1.0 since we normalized current by target
        
        # Progress indicators
        progress = np.array([
            self.current_step / self.max_steps_per_episode,  # Episode progress
            len(self.episode_meals) / self.max_daily_meals,   # Meals progress
            min(1.0, sum(current_nutrition) / 4.0)           # Overall nutrition progress
        ])
        
        # Constraint indicators
        constraints = np.array([
            max(0, current_nutrition[0] - 1.0),  # Calorie excess
            max(0, 1.0 - current_nutrition[0]),  # Calorie deficit
            np.std(current_nutrition[:4])        # Macro imbalance
        ])
        
        # Recent history (last 2 meals nutrition impact)
        history = np.zeros(6)  # 2 meals × 3 macros each
        recent_meals = self.episode_meals[-2:] if len(self.episode_meals) >= 2 else self.episode_meals
        
        for i, meal in enumerate(recent_meals):
            if i < 2:  # Only consider last 2 meals
                meal_ratios = [
                    meal['nutrition']['protein'] / self.target_nutrition['protein'],
                    meal['nutrition']['carbohydrates'] / self.target_nutrition['carbohydrates'],
                    meal['nutrition']['fat'] / self.target_nutrition['fat']
                ]
                history[i*3:(i+1)*3] = meal_ratios
        
        # Concatenate all observation components
        observation = np.concatenate([
            current_nutrition,  # 4 dims
            target_nutrition,   # 4 dims  
            progress,           # 3 dims
            constraints,        # 3 dims
            history            # 6 dims
        ])
        
        # Ensure observation is float32 and has correct shape
        observation = observation.astype(np.float32)
        assert observation.shape == (20,), f"Observation shape mismatch: {observation.shape}"
        
        return observation
    
    def _calculate_reward(self, selected_food, portion_size: float) -> float:
        """Calculate reward for the current action"""
        rewards = {}
        penalties = {}
        
        # 1. Macro balance reward
        target_ratios = np.array([
            self.target_nutrition['protein'],
            self.target_nutrition['carbohydrates'], 
            self.target_nutrition['fat']
        ])
        
        current_ratios = np.array([
            self.total_nutrition['protein'],
            self.total_nutrition['carbohydrates'],
            self.total_nutrition['fat']
        ])
        
        # Calculate how close we are to target ratios
        ratio_errors = np.abs(current_ratios - target_ratios) / target_ratios
        macro_balance_score = max(0, 1 - np.mean(ratio_errors))
        rewards['macro_balance'] = macro_balance_score
        
        # 2. Calorie target reward
        calorie_error = abs(self.total_nutrition['calories'] - self.target_nutrition['calories'])
        calorie_accuracy = max(0, 1 - (calorie_error / self.target_nutrition['calories']))
        rewards['calorie_target'] = calorie_accuracy
        
        # 3. Variety reward
        unique_foods = len(set(meal['food'].name for meal in self.episode_meals))
        total_meals = len(self.episode_meals)
        variety_score = unique_foods / max(1, total_meals)
        rewards['variety'] = variety_score
        
        # 4. Nutritional quality reward
        quality_score = selected_food.quality_score
        preference_score = self.user_profile.get_preference_score(selected_food)
        nutritional_quality = (quality_score + preference_score) / 2
        rewards['nutritional_quality'] = nutritional_quality
        
        # Calculate penalties
        
        # 1. Calorie excess penalty
        if self.total_nutrition['calories'] > self.target_nutrition['calories'] * 1.1:  # 10% threshold
            excess_ratio = (self.total_nutrition['calories'] - self.target_nutrition['calories']) / self.target_nutrition['calories']
            penalties['calorie_excess'] = excess_ratio
        
        # 2. Calorie deficit penalty (less severe)
        if self.total_nutrition['calories'] < self.target_nutrition['calories'] * 0.8:  # 20% threshold
            deficit_ratio = (self.target_nutrition['calories'] - self.total_nutrition['calories']) / self.target_nutrition['calories']
            penalties['calorie_deficit'] = deficit_ratio * 0.5  # Less severe than excess
        
        # 3. Macro imbalance penalty
        macro_imbalance = np.std(ratio_errors)
        if macro_imbalance > 0.3:  # Threshold for significant imbalance
            penalties['macro_imbalance'] = macro_imbalance
        
        # Combine rewards and penalties
        total_reward = sum(rewards[key] * self.reward_weights[key] for key in rewards)
        total_penalty = sum(penalties[key] * abs(self.penalties[key]) for key in penalties)
        
        final_reward = total_reward - total_penalty
        
        return final_reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        return {
            'current_step': self.current_step,
            'meals_consumed': len(self.episode_meals),
            'total_nutrition': self.total_nutrition.copy(),
            'target_nutrition': self.target_nutrition.copy(),
            'available_foods_count': len(self.available_foods)
        }
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Meals: {len(self.episode_meals)}")
            print(f"Nutrition: {self.total_nutrition}")
            print(f"Target: {self.target_nutrition}")
            print("-" * 40)
    
    def close(self):
        """Clean up environment resources"""
        pass 