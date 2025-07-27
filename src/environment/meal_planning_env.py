"""
Reinforcement Learning Environment for Meal Planning.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..data.user_profile import UserProfile
from ..data.nutritional_database import NutritionalDatabase, FoodItem


@dataclass
class MealState:
    """Represents the current state of the meal planning environment."""
    # Nutritional targets
    target_calories: float
    target_protein: float
    target_carbs: float
    target_fats: float
    
    # Current nutritional intake
    current_calories: float
    current_protein: float
    current_carbs: float
    current_fats: float
    
    # Meal planning state
    meals_planned: int
    max_meals: int
    
    # User preferences (encoded)
    dietary_restrictions: List[float]  # One-hot encoded
    preferred_categories: List[float]  # One-hot encoded
    excluded_allergens: List[float]    # One-hot encoded
    
    # Available food items (encoded)
    available_foods: List[float]  # Binary vector of available foods
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL agent."""
        return np.concatenate([
            [self.target_calories, self.target_protein, self.target_carbs, self.target_fats],
            [self.current_calories, self.current_protein, self.current_carbs, self.current_fats],
            [self.meals_planned, self.max_meals],
            self.dietary_restrictions,
            self.preferred_categories,
            self.excluded_allergens,
            self.available_foods
        ])


class MealPlanningEnvironment(gym.Env):
    """
    Gym environment for meal planning using reinforcement learning.
    """
    
    def __init__(self, nutritional_db: NutritionalDatabase, config: Any):
        super().__init__()
        
        self.nutritional_db = nutritional_db
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Environment parameters
        self.max_steps = config.max_steps
        self.max_meals_per_day = config.max_meals_per_day
        self.min_calories = config.min_calories
        self.max_calories = config.max_calories
        self.target_protein_ratio = config.target_protein_ratio
        self.target_carbs_ratio = config.target_carbs_ratio
        self.target_fats_ratio = config.target_fats_ratio
        
        # Reward weights
        self.nutrition_weight = config.nutrition_weight
        self.satisfaction_weight = config.satisfaction_weight
        self.variety_weight = config.variety_weight
        self.variety_penalty = config.variety_penalty
        
        # Current state
        self.current_user: Optional[UserProfile] = None
        self.current_state: Optional[MealState] = None
        self.selected_foods: List[str] = []
        self.step_count = 0
        
        # Available food categories
        self.food_categories = self.nutritional_db.get_categories()
        self.all_foods = list(self.nutritional_db.foods.keys())
        
        # Encode dietary restrictions and allergens
        self._setup_encodings()
        
        # Define action and observation spaces
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: select food items (discrete actions)
        self.action_space = gym.spaces.Discrete(len(self.nutritional_db.foods))
        
        # Observation space: state vector
        state_size = (
            4 +  # Nutritional targets
            4 +  # Current nutritional intake
            2 +  # Meal planning state
            len(self.nutritional_db.foods) +  # Dietary restrictions (one-hot)
            len(self.food_categories) +       # Preferred categories (one-hot)
            len(self.nutritional_db.foods) +  # Excluded allergens (one-hot)
            len(self.nutritional_db.foods)    # Available foods (binary)
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
    
    def _setup_encodings(self):
        """Setup encodings for dietary restrictions and allergens."""
        # Get all unique allergens and dietary restrictions
        all_allergens = set()
        all_dietary_restrictions = set()
        
        for food in self.nutritional_db.foods.values():
            all_allergens.update(food.allergens)
            # Add dietary restriction tags
            if 'vegetarian' in food.tags:
                all_dietary_restrictions.add('vegetarian')
            if 'vegan' in food.tags:
                all_dietary_restrictions.add('vegan')
            if 'gluten_free' in food.tags:
                all_dietary_restrictions.add('gluten_free')
        
        self.all_allergens = list(all_allergens)
        self.all_dietary_restrictions = list(all_dietary_restrictions)
    
    def set_user_profile(self, user_profile: UserProfile):
        """Set the user profile for the current episode."""
        self.current_user = user_profile
        self.selected_foods = []
        self.step_count = 0
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        if self.current_user is None:
            raise ValueError("User profile must be set before reset")
        
        # Initialize state
        self.selected_foods = []
        self.step_count = 0
        
        # Create initial state
        self.current_state = self._create_initial_state()
        
        return self.current_state.to_array(), {}
    
    def _create_initial_state(self) -> MealState:
        """Create the initial state for the episode."""
        goals = self.current_user.nutritional_goals
        
        # Encode dietary restrictions
        dietary_restrictions = self._encode_dietary_restrictions()
        
        # Encode preferred categories
        preferred_categories = self._encode_preferred_categories()
        
        # Encode excluded allergens
        excluded_allergens = self._encode_excluded_allergens()
        
        # Encode available foods
        available_foods = self._encode_available_foods()
        
        return MealState(
            target_calories=goals.target_calories,
            target_protein=goals.target_protein,
            target_carbs=goals.target_carbs,
            target_fats=goals.target_fats,
            current_calories=0.0,
            current_protein=0.0,
            current_carbs=0.0,
            current_fats=0.0,
            meals_planned=0,
            max_meals=self.max_meals_per_day,
            dietary_restrictions=dietary_restrictions,
            preferred_categories=preferred_categories,
            excluded_allergens=excluded_allergens,
            available_foods=available_foods
        )
    
    def _encode_dietary_restrictions(self) -> List[float]:
        """Encode dietary restrictions as one-hot vector."""
        encoding = [0.0] * len(self.nutritional_db.foods)
        
        for restriction in self.current_user.dietary_restrictions:
            for i, food in enumerate(self.nutritional_db.foods.values()):
                if restriction.value == 'vegetarian' and 'vegetarian' in food.tags:
                    encoding[i] = 1.0
                elif restriction.value == 'vegan' and 'vegan' in food.tags:
                    encoding[i] = 1.0
                elif restriction.value == 'gluten_free' and 'gluten_free' in food.tags:
                    encoding[i] = 1.0
        
        return encoding
    
    def _encode_preferred_categories(self) -> List[float]:
        """Encode preferred food categories as one-hot vector."""
        encoding = [0.0] * len(self.food_categories)
        
        # For now, assume all categories are preferred
        # In a real implementation, this would be based on user preferences
        for i in range(len(self.food_categories)):
            encoding[i] = 1.0
        
        return encoding
    
    def _encode_excluded_allergens(self) -> List[float]:
        """Encode excluded allergens as binary vector."""
        encoding = [0.0] * len(self.nutritional_db.foods)
        
        for i, food in enumerate(self.nutritional_db.foods.values()):
            if any(allergen in food.allergens for allergen in self.current_user.allergies):
                encoding[i] = 1.0
        
        return encoding
    
    def _encode_available_foods(self) -> List[float]:
        """Encode available foods as binary vector."""
        encoding = [1.0] * len(self.nutritional_db.foods)  # All foods available initially
        
        # Mark already selected foods as unavailable
        for food_id in self.selected_foods:
            if food_id in self.nutritional_db.foods:
                idx = list(self.nutritional_db.foods.keys()).index(food_id)
                encoding[idx] = 0.0
        
        return encoding
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Index of the food item to select
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Get the selected food
        food_ids = list(self.nutritional_db.foods.keys())
        if action >= len(food_ids):
            action = len(food_ids) - 1  # Clamp action
        
        selected_food_id = food_ids[action]
        selected_food = self.nutritional_db.foods[selected_food_id]
        
        # Add food to selection
        self.selected_foods.append(selected_food_id)
        
        # Update current nutritional intake
        self.current_state.current_calories += selected_food.calories
        self.current_state.current_protein += selected_food.protein
        self.current_state.current_carbs += selected_food.carbs
        self.current_state.current_fats += selected_food.fats
        self.current_state.meals_planned += 1
        
        # Update available foods
        self.current_state.available_foods = self._encode_available_foods()
        
        # Calculate reward
        reward = self._calculate_reward(selected_food)
        
        # Check if episode is done
        done = self._is_done()
        
        # Create info dictionary
        info = {
            'selected_food': selected_food.name,
            'nutritional_summary': {
                'calories': self.current_state.current_calories,
                'protein': self.current_state.current_protein,
                'carbs': self.current_state.current_carbs,
                'fats': self.current_state.current_fats
            },
            'meals_planned': self.current_state.meals_planned
        }
        
        return self.current_state.to_array(), reward, done, False, info
    
    def _calculate_reward(self, selected_food: FoodItem) -> float:
        """Calculate reward for selecting a food item."""
        # Nutrition reward
        nutrition_reward = self._calculate_nutrition_reward(selected_food)
        
        # Satisfaction reward
        satisfaction_reward = self._calculate_satisfaction_reward(selected_food)
        
        # Variety reward
        variety_reward = self._calculate_variety_reward(selected_food)
        
        # Combine rewards
        total_reward = (
            self.nutrition_weight * nutrition_reward +
            self.satisfaction_weight * satisfaction_reward +
            self.variety_weight * variety_reward
        )
        
        return total_reward
    
    def _calculate_nutrition_reward(self, selected_food: FoodItem) -> float:
        """Calculate reward based on nutritional contribution."""
        # Calculate how well this food contributes to targets
        remaining_calories = self.current_state.target_calories - self.current_state.current_calories
        remaining_protein = self.current_state.target_protein - self.current_state.current_protein
        remaining_carbs = self.current_state.target_carbs - self.current_state.current_carbs
        remaining_fats = self.current_state.target_fats - self.current_state.current_fats
        
        # Normalize contributions
        calorie_contribution = min(selected_food.calories / remaining_calories, 1.0) if remaining_calories > 0 else 0.0
        protein_contribution = min(selected_food.protein / remaining_protein, 1.0) if remaining_protein > 0 else 0.0
        carbs_contribution = min(selected_food.carbs / remaining_carbs, 1.0) if remaining_carbs > 0 else 0.0
        fats_contribution = min(selected_food.fats / remaining_fats, 1.0) if remaining_fats > 0 else 0.0
        
        # Penalize over-consumption
        if remaining_calories < 0:
            calorie_contribution = -abs(selected_food.calories / self.current_state.target_calories)
        if remaining_protein < 0:
            protein_contribution = -abs(selected_food.protein / self.current_state.target_protein)
        if remaining_carbs < 0:
            carbs_contribution = -abs(selected_food.carbs / self.current_state.target_carbs)
        if remaining_fats < 0:
            fats_contribution = -abs(selected_food.fats / self.current_state.target_fats)
        
        return (calorie_contribution + protein_contribution + carbs_contribution + fats_contribution) / 4
    
    def _calculate_satisfaction_reward(self, selected_food: FoodItem) -> float:
        """Calculate reward based on user satisfaction."""
        satisfaction = 0.0
        
        # Check if food is in preferred categories
        if selected_food.category in self.current_user.preferred_foods:
            satisfaction += 0.3
        
        # Check if food is not disliked
        if selected_food.name.lower() not in self.current_user.disliked_foods:
            satisfaction += 0.3
        
        # Check if food meets dietary restrictions
        dietary_compliance = 1.0
        for restriction in self.current_user.dietary_restrictions:
            if restriction.value == 'vegetarian' and 'vegetarian' not in selected_food.tags:
                dietary_compliance = 0.0
            elif restriction.value == 'vegan' and 'vegan' not in selected_food.tags:
                dietary_compliance = 0.0
            elif restriction.value == 'gluten_free' and 'gluten_free' not in selected_food.tags:
                dietary_compliance = 0.0
        
        satisfaction += 0.4 * dietary_compliance
        
        return satisfaction
    
    def _calculate_variety_reward(self, selected_food: FoodItem) -> float:
        """Calculate reward based on meal variety."""
        # Penalize selecting the same food multiple times
        if selected_food.food_id in self.selected_foods[:-1]:  # Exclude current selection
            return -self.variety_penalty
        
        # Reward for selecting different categories
        selected_categories = set()
        for food_id in self.selected_foods:
            food = self.nutritional_db.foods.get(food_id)
            if food:
                selected_categories.add(food.category)
        
        variety_bonus = len(selected_categories) / len(self.food_categories)
        
        return variety_bonus
    
    def _is_done(self) -> bool:
        """Check if the episode is done."""
        # Episode ends if:
        # 1. Maximum steps reached
        # 2. Maximum meals planned
        # 3. Nutritional targets met (with some tolerance)
        
        if self.step_count >= self.max_steps:
            return True
        
        if self.current_state.meals_planned >= self.max_meals_per_day:
            return True
        
        # Check if nutritional targets are met (within 10% tolerance)
        calorie_ratio = self.current_state.current_calories / self.current_state.target_calories
        protein_ratio = self.current_state.current_protein / self.current_state.target_protein
        carbs_ratio = self.current_state.current_carbs / self.current_state.target_carbs
        fats_ratio = self.current_state.current_fats / self.current_state.target_fats
        
        if (0.9 <= calorie_ratio <= 1.1 and 
            0.9 <= protein_ratio <= 1.1 and 
            0.9 <= carbs_ratio <= 1.1 and 
            0.9 <= fats_ratio <= 1.1):
            return True
        
        return False
    
    def get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return self.current_state.to_array()
    
    def get_meal_plan(self) -> Dict[str, Any]:
        """Get the current meal plan."""
        meals = []
        for food_id in self.selected_foods:
            food = self.nutritional_db.foods.get(food_id)
            if food:
                meals.append({
                    'food_id': food_id,
                    'name': food.name,
                    'category': food.category,
                    'calories': food.calories,
                    'protein': food.protein,
                    'carbs': food.carbs,
                    'fats': food.fats,
                    'serving_size': food.serving_size
                })
        
        nutritional_summary = self.nutritional_db.get_nutritional_summary(self.selected_foods)
        
        return {
            'meals': meals,
            'total_calories': nutritional_summary['calories'],
            'total_protein': nutritional_summary['protein'],
            'total_carbs': nutritional_summary['carbs'],
            'total_fats': nutritional_summary['fats'],
            'satisfaction_score': self._calculate_overall_satisfaction(),
            'variety_score': self._calculate_overall_variety()
        }
    
    def _calculate_overall_satisfaction(self) -> float:
        """Calculate overall satisfaction score for the meal plan."""
        if not self.selected_foods:
            return 0.0
        
        total_satisfaction = 0.0
        for food_id in self.selected_foods:
            food = self.nutritional_db.foods.get(food_id)
            if food:
                satisfaction = 0.0
                
                # Category preference
                if food.category in self.current_user.preferred_foods:
                    satisfaction += 0.3
                
                # Not disliked
                if food.name.lower() not in self.current_user.disliked_foods:
                    satisfaction += 0.3
                
                # Dietary compliance
                dietary_compliance = 1.0
                for restriction in self.current_user.dietary_restrictions:
                    if restriction.value == 'vegetarian' and 'vegetarian' not in food.tags:
                        dietary_compliance = 0.0
                    elif restriction.value == 'vegan' and 'vegan' not in food.tags:
                        dietary_compliance = 0.0
                    elif restriction.value == 'gluten_free' and 'gluten_free' not in food.tags:
                        dietary_compliance = 0.0
                
                satisfaction += 0.4 * dietary_compliance
                total_satisfaction += satisfaction
        
        return total_satisfaction / len(self.selected_foods)
    
    def _calculate_overall_variety(self) -> float:
        """Calculate overall variety score for the meal plan."""
        if not self.selected_foods:
            return 0.0
        
        selected_categories = set()
        for food_id in self.selected_foods:
            food = self.nutritional_db.foods.get(food_id)
            if food:
                selected_categories.add(food.category)
        
        return len(selected_categories) / len(self.food_categories) 