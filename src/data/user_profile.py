"""
User profile management for personalized meal planning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum
import json
import os


class DietaryRestriction(Enum):
    """Dietary restriction types."""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    KETO = "keto"
    PALEO = "paleo"
    LOW_CARB = "low_carb"
    HIGH_PROTEIN = "high_protein"


class ActivityLevel(Enum):
    """Activity level categories."""
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"


@dataclass
class NutritionalGoals:
    """Nutritional goals for the user."""
    target_calories: float
    target_protein: float  # grams
    target_carbs: float    # grams
    target_fats: float     # grams
    target_fiber: float    # grams
    target_sodium: float   # mg
    target_sugar: float    # grams
    
    def __post_init__(self):
        """Validate nutritional goals."""
        if self.target_calories <= 0:
            raise ValueError("Target calories must be positive")
        if any(x < 0 for x in [self.target_protein, self.target_carbs, self.target_fats]):
            raise ValueError("Macronutrient targets must be non-negative")


@dataclass
class UserProfile:
    """User profile containing dietary preferences and requirements."""
    user_id: str
    name: str
    age: int
    gender: str
    height: float  # cm
    weight: float  # kg
    activity_level: ActivityLevel
    dietary_restrictions: Set[DietaryRestriction] = field(default_factory=set)
    allergies: Set[str] = field(default_factory=set)
    disliked_foods: Set[str] = field(default_factory=set)
    preferred_foods: Set[str] = field(default_factory=set)
    nutritional_goals: Optional[NutritionalGoals] = None
    previous_feedback: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate nutritional goals if not provided."""
        if self.nutritional_goals is None:
            self.nutritional_goals = self._calculate_nutritional_goals()
    
    def _calculate_nutritional_goals(self) -> NutritionalGoals:
        """Calculate nutritional goals based on user profile."""
        # Calculate BMR using Mifflin-St Jeor Equation
        if self.gender.lower() in ['male', 'm']:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        
        # Apply activity multiplier
        activity_multipliers = {
            ActivityLevel.SEDENTARY: 1.2,
            ActivityLevel.LIGHTLY_ACTIVE: 1.375,
            ActivityLevel.MODERATELY_ACTIVE: 1.55,
            ActivityLevel.VERY_ACTIVE: 1.725,
            ActivityLevel.EXTREMELY_ACTIVE: 1.9
        }
        
        tdee = bmr * activity_multipliers[self.activity_level]
        
        # Calculate macronutrient targets
        target_protein = self.weight * 2.0  # 2g per kg body weight
        target_fats = (tdee * 0.25) / 9  # 25% of calories from fat
        target_carbs = (tdee - (target_protein * 4) - (target_fats * 9)) / 4
        
        return NutritionalGoals(
            target_calories=tdee,
            target_protein=target_protein,
            target_carbs=target_carbs,
            target_fats=target_fats,
            target_fiber=25.0,  # Recommended daily fiber
            target_sodium=2300.0,  # Recommended daily sodium
            target_sugar=50.0  # Recommended daily added sugar
        )
    
    def add_dietary_restriction(self, restriction: DietaryRestriction):
        """Add a dietary restriction."""
        self.dietary_restrictions.add(restriction)
    
    def remove_dietary_restriction(self, restriction: DietaryRestriction):
        """Remove a dietary restriction."""
        self.dietary_restrictions.discard(restriction)
    
    def add_allergy(self, allergen: str):
        """Add an allergy."""
        self.allergies.add(allergen.lower())
    
    def remove_allergy(self, allergen: str):
        """Remove an allergy."""
        self.allergies.discard(allergen.lower())
    
    def add_disliked_food(self, food: str):
        """Add a disliked food."""
        self.disliked_foods.add(food.lower())
    
    def remove_disliked_food(self, food: str):
        """Remove a disliked food."""
        self.disliked_foods.discard(food.lower())
    
    def add_preferred_food(self, food: str):
        """Add a preferred food."""
        self.preferred_foods.add(food.lower())
    
    def remove_preferred_food(self, food: str):
        """Remove a preferred food."""
        self.preferred_foods.discard(food.lower())
    
    def add_feedback(self, feedback: Dict[str, Any]):
        """Add user feedback."""
        self.previous_feedback.append(feedback)
    
    def get_recent_feedback(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent feedback within specified days."""
        # This would typically filter by timestamp
        # For now, return all feedback
        return self.previous_feedback[-10:]  # Last 10 feedback entries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'height': self.height,
            'weight': self.weight,
            'activity_level': self.activity_level.value,
            'dietary_restrictions': [r.value for r in self.dietary_restrictions],
            'allergies': list(self.allergies),
            'disliked_foods': list(self.disliked_foods),
            'preferred_foods': list(self.preferred_foods),
            'nutritional_goals': {
                'target_calories': self.nutritional_goals.target_calories,
                'target_protein': self.nutritional_goals.target_protein,
                'target_carbs': self.nutritional_goals.target_carbs,
                'target_fats': self.nutritional_goals.target_fats,
                'target_fiber': self.nutritional_goals.target_fiber,
                'target_sodium': self.nutritional_goals.target_sodium,
                'target_sugar': self.nutritional_goals.target_sugar,
            },
            'previous_feedback': self.previous_feedback
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create profile from dictionary."""
        # Convert activity level string to enum
        activity_level = ActivityLevel(data['activity_level'])
        
        # Convert dietary restrictions strings to enums
        dietary_restrictions = {DietaryRestriction(r) for r in data.get('dietary_restrictions', [])}
        
        # Create nutritional goals
        goals_data = data.get('nutritional_goals', {})
        nutritional_goals = NutritionalGoals(
            target_calories=goals_data.get('target_calories', 2000),
            target_protein=goals_data.get('target_protein', 150),
            target_carbs=goals_data.get('target_carbs', 200),
            target_fats=goals_data.get('target_fats', 65),
            target_fiber=goals_data.get('target_fiber', 25),
            target_sodium=goals_data.get('target_sodium', 2300),
            target_sugar=goals_data.get('target_sugar', 50),
        )
        
        return cls(
            user_id=data['user_id'],
            name=data['name'],
            age=data['age'],
            gender=data['gender'],
            height=data['height'],
            weight=data['weight'],
            activity_level=activity_level,
            dietary_restrictions=dietary_restrictions,
            allergies=set(data.get('allergies', [])),
            disliked_foods=set(data.get('disliked_foods', [])),
            preferred_foods=set(data.get('preferred_foods', [])),
            nutritional_goals=nutritional_goals,
            previous_feedback=data.get('previous_feedback', [])
        )
    
    def save(self, filepath: str):
        """Save profile to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'UserProfile':
        """Load profile from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class UserProfileManager:
    """Manager for user profiles."""
    
    def __init__(self, profiles_dir: str = "data/profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
    
    def save_profile(self, profile: UserProfile):
        """Save a user profile."""
        filepath = os.path.join(self.profiles_dir, f"{profile.user_id}.json")
        profile.save(filepath)
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load a user profile."""
        filepath = os.path.join(self.profiles_dir, f"{user_id}.json")
        if os.path.exists(filepath):
            return UserProfile.load(filepath)
        return None
    
    def list_profiles(self) -> List[str]:
        """List all user IDs."""
        profiles = []
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                user_id = filename[:-5]  # Remove .json extension
                profiles.append(user_id)
        return profiles
    
    def delete_profile(self, user_id: str):
        """Delete a user profile."""
        filepath = os.path.join(self.profiles_dir, f"{user_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath) 