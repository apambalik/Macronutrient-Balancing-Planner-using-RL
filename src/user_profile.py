"""
User profile management for the Meal Planning RL system.

This module handles user-specific settings, preferences, and nutritional goals.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ActivityLevel(Enum):
    """Physical activity levels for BMR calculation"""
    SEDENTARY = 1.2
    LIGHTLY_ACTIVE = 1.375
    MODERATELY_ACTIVE = 1.55
    VERY_ACTIVE = 1.725
    EXTRA_ACTIVE = 1.9


class Gender(Enum):
    """Gender options for BMR calculation"""
    MALE = "male"
    FEMALE = "female"


class Goal(Enum):
    """Weight management goals"""
    LOSE_WEIGHT = "lose"
    MAINTAIN_WEIGHT = "maintain"
    GAIN_WEIGHT = "gain"


@dataclass
class UserProfile:
    """User profile containing personal information, goals, and preferences"""
    
    # Personal information
    age: int
    gender: Gender
    weight: float  # kg
    height: float  # cm
    activity_level: ActivityLevel
    
    # Goals and preferences
    goals: Dict = field(default_factory=lambda: {
        'weight_goal': Goal.MAINTAIN_WEIGHT,
        'target_calories': 2000,
        'protein_ratio': 0.25,
        'carb_ratio': 0.45,
        'fat_ratio': 0.30
    })
    
    preferences: Dict = field(default_factory=lambda: {
        'liked_categories': [],
        'disliked_foods': [],
        'cuisine_preferences': [],
        'dietary_restrictions': [],
        'allergens': []
    })
    
    health_conditions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate user profile data"""
        self._validate_profile()
        
        # Calculate and set target calories if not specified
        if self.goals.get('target_calories') is None:
            self.goals['target_calories'] = self.calculate_daily_calories()
    
    def _validate_profile(self):
        """Validate user profile data"""
        if self.age <= 0 or self.age > 150:
            raise ValueError(f"Invalid age: {self.age}")
        
        if self.weight <= 0 or self.weight > 1000:
            raise ValueError(f"Invalid weight: {self.weight}")
        
        if self.height <= 0 or self.height > 300:
            raise ValueError(f"Invalid height: {self.height}")
        
        # Validate macro ratios
        macro_sum = (self.goals.get('protein_ratio', 0) + 
                    self.goals.get('carb_ratio', 0) + 
                    self.goals.get('fat_ratio', 0))
        
        if abs(macro_sum - 1.0) > 0.01:
            logger.warning(f"Macro ratios sum to {macro_sum}, adjusting to sum to 1.0")
            # Normalize ratios
            self.goals['protein_ratio'] /= macro_sum
            self.goals['carb_ratio'] /= macro_sum
            self.goals['fat_ratio'] /= macro_sum
    
    def get_target_macros(self) -> Tuple[float, float, float]:
        """Get target macronutrient distribution (protein, carbs, fat)"""
        return (
            self.goals.get('protein_ratio', 0.25),
            self.goals.get('carb_ratio', 0.45),
            self.goals.get('fat_ratio', 0.30)
        )
    
    def get_target_macro_calories(self) -> Tuple[float, float, float]:
        """Get target macronutrient calories (protein, carbs, fat)"""
        target_calories = self.goals.get('target_calories', 2000)
        protein_ratio, carb_ratio, fat_ratio = self.get_target_macros()
        
        return (
            target_calories * protein_ratio,
            target_calories * carb_ratio,
            target_calories * fat_ratio
        )
    
    def get_target_macro_grams(self) -> Tuple[float, float, float]:
        """Get target macronutrient grams (protein, carbs, fat)"""
        protein_cal, carb_cal, fat_cal = self.get_target_macro_calories()
        
        # Convert calories to grams (4 cal/g for protein and carbs, 9 cal/g for fat)
        return (
            protein_cal / 4,  # grams of protein
            carb_cal / 4,     # grams of carbohydrates
            fat_cal / 9       # grams of fat
        )
    
    def calculate_bmr(self) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
        if self.gender == Gender.MALE:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        
        return bmr
    
    def calculate_daily_calories(self) -> float:
        """Calculate daily calorie needs based on BMR and activity level"""
        bmr = self.calculate_bmr()
        daily_calories = bmr * self.activity_level.value
        
        # Adjust based on weight goal
        goal = self.goals.get('weight_goal', Goal.MAINTAIN_WEIGHT)
        
        if goal == Goal.LOSE_WEIGHT:
            daily_calories *= 0.85  # 15% calorie deficit
        elif goal == Goal.GAIN_WEIGHT:
            daily_calories *= 1.15  # 15% calorie surplus
        
        return round(daily_calories)
    
    def calculate_bmi(self) -> float:
        """Calculate Body Mass Index"""
        height_m = self.height / 100  # convert cm to meters
        return self.weight / (height_m ** 2)
    
    def get_bmi_category(self) -> str:
        """Get BMI category"""
        bmi = self.calculate_bmi()
        
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def add_dietary_restriction(self, restriction: str):
        """Add a dietary restriction"""
        if restriction not in self.preferences['dietary_restrictions']:
            self.preferences['dietary_restrictions'].append(restriction)
            logger.info(f"Added dietary restriction: {restriction}")
    
    def remove_dietary_restriction(self, restriction: str):
        """Remove a dietary restriction"""
        if restriction in self.preferences['dietary_restrictions']:
            self.preferences['dietary_restrictions'].remove(restriction)
            logger.info(f"Removed dietary restriction: {restriction}")
    
    def add_allergen(self, allergen: str):
        """Add an allergen to avoid"""
        if allergen not in self.preferences['allergens']:
            self.preferences['allergens'].append(allergen)
            logger.info(f"Added allergen: {allergen}")
    
    def remove_allergen(self, allergen: str):
        """Remove an allergen"""
        if allergen in self.preferences['allergens']:
            self.preferences['allergens'].remove(allergen)
            logger.info(f"Removed allergen: {allergen}")
    
    def add_liked_category(self, category: str):
        """Add a liked food category"""
        if category not in self.preferences['liked_categories']:
            self.preferences['liked_categories'].append(category)
            logger.info(f"Added liked category: {category}")
    
    def add_disliked_food(self, food: str):
        """Add a disliked food"""
        if food not in self.preferences['disliked_foods']:
            self.preferences['disliked_foods'].append(food)
            logger.info(f"Added disliked food: {food}")
    
    def is_food_compatible(self, food_item) -> bool:
        """Check if a food item is compatible with user preferences and restrictions"""
        from .data_loader import FoodItem
        
        if not isinstance(food_item, FoodItem):
            return False
        
        # Check allergens
        user_allergens = set(self.preferences.get('allergens', []))
        food_allergens = set(food_item.allergens)
        if user_allergens.intersection(food_allergens):
            return False
        
        # Check disliked foods
        disliked = self.preferences.get('disliked_foods', [])
        if any(disliked_food.lower() in food_item.name.lower() for disliked_food in disliked):
            return False
        
        # Check dietary restrictions
        restrictions = self.preferences.get('dietary_restrictions', [])
        for restriction in restrictions:
            if not self._check_dietary_restriction(food_item, restriction):
                return False
        
        return True
    
    def _check_dietary_restriction(self, food_item, restriction: str) -> bool:
        """Check if food item meets a specific dietary restriction"""
        restriction_lower = restriction.lower()
        
        if restriction_lower == 'vegetarian':
            # Check if food contains meat-related categories
            meat_categories = ['meat', 'beef', 'pork', 'chicken', 'fish', 'seafood']
            return not any(cat in food_item.categories for cat in meat_categories)
        
        elif restriction_lower == 'vegan':
            # Check for animal products
            animal_categories = ['meat', 'dairy', 'eggs', 'honey', 'beef', 'pork', 'chicken', 'fish']
            return not any(cat in food_item.categories for cat in animal_categories)
        
        elif restriction_lower == 'gluten-free':
            # Check for gluten-containing ingredients
            gluten_sources = ['wheat', 'barley', 'rye', 'oats']  # Note: oats can be contaminated
            return not any(source in food_item.ingredients.lower() for source in gluten_sources)
        
        elif restriction_lower == 'dairy-free':
            # Check for dairy products
            dairy_terms = ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'lactose']
            return not any(term in food_item.ingredients.lower() for term in dairy_terms)
        
        elif restriction_lower == 'low-sodium':
            # Check sodium content (less than 140mg per 100g is considered low sodium)
            return food_item.nutrition.sodium < 140
        
        elif restriction_lower == 'low-fat':
            # Check fat content (less than 3g per 100g is considered low fat)
            return food_item.nutrition.fat < 3
        
        elif restriction_lower == 'low-sugar':
            # Check sugar content (less than 5g per 100g is considered low sugar)
            return food_item.nutrition.sugar < 5
        
        # If restriction not recognized, be conservative and allow the food
        logger.warning(f"Unknown dietary restriction: {restriction}")
        return True
    
    def get_preference_score(self, food_item) -> float:
        """Calculate a preference score for a food item (0-1 scale)"""
        from .data_loader import FoodItem
        
        if not isinstance(food_item, FoodItem):
            return 0.0
        
        score = 0.5  # Base score
        
        # Boost score for liked categories
        liked_categories = self.preferences.get('liked_categories', [])
        for category in liked_categories:
            if category.lower() in [cat.lower() for cat in food_item.categories]:
                score += 0.2
        
        # Reduce score for foods similar to disliked foods
        disliked = self.preferences.get('disliked_foods', [])
        for disliked_food in disliked:
            if disliked_food.lower() in food_item.name.lower():
                score -= 0.3
        
        # Boost score for better nutrition grades
        grade_scores = {'a': 0.2, 'b': 0.1, 'c': 0.0, 'd': -0.1, 'e': -0.2}
        grade = food_item.nutrition_grade.lower()
        score += grade_scores.get(grade, 0)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def update_goals(self, new_goals: Dict):
        """Update user goals"""
        for key, value in new_goals.items():
            if key in self.goals:
                self.goals[key] = value
                logger.info(f"Updated goal {key}: {value}")
        
        # Recalculate target calories if weight goal changed
        if 'weight_goal' in new_goals:
            self.goals['target_calories'] = self.calculate_daily_calories()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the user profile"""
        return {
            'personal_info': {
                'age': self.age,
                'gender': self.gender.value,
                'weight': self.weight,
                'height': self.height,
                'activity_level': self.activity_level.name,
                'bmi': round(self.calculate_bmi(), 1),
                'bmi_category': self.get_bmi_category()
            },
            'goals': {
                'weight_goal': self.goals['weight_goal'].value,
                'target_calories': self.goals['target_calories'],
                'bmr': round(self.calculate_bmr()),
                'macro_ratios': self.get_target_macros(),
                'macro_grams': tuple(round(g, 1) for g in self.get_target_macro_grams())
            },
            'preferences': self.preferences,
            'health_conditions': self.health_conditions
        } 