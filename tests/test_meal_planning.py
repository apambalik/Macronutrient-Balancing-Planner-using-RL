"""
Tests for the RL Meal Planning System.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.user_profile import UserProfile, ActivityLevel, DietaryRestriction
from src.data.nutritional_database import NutritionalDatabase
from src.feedback.feedback_processor import FeedbackProcessor
from src.utils.config import Config


class TestUserProfile(unittest.TestCase):
    """Test user profile functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.user = UserProfile(
            user_id="test_user",
            name="Test User",
            age=25,
            gender="female",
            height=165.0,
            weight=60.0,
            activity_level=ActivityLevel.MODERATELY_ACTIVE
        )
    
    def test_user_creation(self):
        """Test user profile creation."""
        self.assertEqual(self.user.user_id, "test_user")
        self.assertEqual(self.user.name, "Test User")
        self.assertEqual(self.user.age, 25)
        self.assertEqual(self.user.gender, "female")
        self.assertEqual(self.user.height, 165.0)
        self.assertEqual(self.user.weight, 60.0)
        self.assertEqual(self.user.activity_level, ActivityLevel.MODERATELY_ACTIVE)
    
    def test_nutritional_goals_calculation(self):
        """Test nutritional goals calculation."""
        goals = self.user.nutritional_goals
        self.assertIsNotNone(goals)
        self.assertGreater(goals.target_calories, 0)
        self.assertGreater(goals.target_protein, 0)
        self.assertGreater(goals.target_carbs, 0)
        self.assertGreater(goals.target_fats, 0)
    
    def test_dietary_restrictions(self):
        """Test dietary restrictions."""
        self.user.add_dietary_restriction(DietaryRestriction.VEGETARIAN)
        self.assertIn(DietaryRestriction.VEGETARIAN, self.user.dietary_restrictions)
        
        self.user.remove_dietary_restriction(DietaryRestriction.VEGETARIAN)
        self.assertNotIn(DietaryRestriction.VEGETARIAN, self.user.dietary_restrictions)
    
    def test_allergies(self):
        """Test allergy management."""
        self.user.add_allergy("nuts")
        self.assertIn("nuts", self.user.allergies)
        
        self.user.remove_allergy("nuts")
        self.assertNotIn("nuts", self.user.allergies)
    
    def test_food_preferences(self):
        """Test food preferences."""
        self.user.add_preferred_food("quinoa")
        self.assertIn("quinoa", self.user.preferred_foods)
        
        self.user.add_disliked_food("brussels sprouts")
        self.assertIn("brussels sprouts", self.user.disliked_foods)


class TestNutritionalDatabase(unittest.TestCase):
    """Test nutritional database functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db = NutritionalDatabase()
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db.foods)
        self.assertGreater(len(self.db.foods), 0)
        self.assertIsNotNone(self.db.categories)
        self.assertGreater(len(self.db.categories), 0)
    
    def test_food_search(self):
        """Test food search functionality."""
        results = self.db.search_foods("chicken")
        self.assertIsInstance(results, list)
        
        if results:
            self.assertIn("chicken", results[0].name.lower())
    
    def test_food_filtering(self):
        """Test food filtering functionality."""
        filtered_foods = self.db.filter_foods(
            min_calories=100,
            max_calories=200,
            excluded_allergens=["nuts"]
        )
        self.assertIsInstance(filtered_foods, list)
        
        for food in filtered_foods:
            self.assertGreaterEqual(food.calories, 100)
            self.assertLessEqual(food.calories, 200)
            self.assertNotIn("nuts", food.allergens)
    
    def test_nutritional_summary(self):
        """Test nutritional summary calculation."""
        food_ids = ["chicken_breast", "brown_rice", "broccoli"]
        summary = self.db.get_nutritional_summary(food_ids)
        
        self.assertIn("calories", summary)
        self.assertIn("protein", summary)
        self.assertIn("carbs", summary)
        self.assertIn("fats", summary)
        self.assertGreater(summary["calories"], 0)


class TestFeedbackProcessor(unittest.TestCase):
    """Test feedback processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = FeedbackProcessor()
    
    def test_feedback_processing(self):
        """Test feedback processing."""
        feedback_data = {
            'meal_plan_id': 'test_plan',
            'satisfaction_score': 0.8,
            'nutrition_score': 0.9,
            'variety_score': 0.7,
            'overall_rating': 4.0,
            'comments': 'Great meal plan!',
            'specific_feedback': {
                'liked_foods': ['quinoa'],
                'disliked_foods': [],
                'nutritional_issues': [],
                'variety_issues': [],
                'preference_updates': {}
            }
        }
        
        processed_feedback = self.processor.process_feedback("test_user", feedback_data)
        
        self.assertIn("feedback_type", processed_feedback)
        self.assertIn("weighted_score", processed_feedback)
        self.assertIn("satisfaction_score", processed_feedback)
        self.assertIn("nutrition_score", processed_feedback)
        self.assertIn("variety_score", processed_feedback)
    
    def test_feedback_summary(self):
        """Test feedback summary generation."""
        # Add some test feedback first
        feedback_data = {
            'meal_plan_id': 'test_plan',
            'satisfaction_score': 0.8,
            'nutrition_score': 0.9,
            'variety_score': 0.7,
            'overall_rating': 4.0,
            'comments': 'Test feedback',
            'specific_feedback': {}
        }
        
        self.processor.process_feedback("test_user", feedback_data)
        
        summary = self.processor.get_feedback_summary("test_user")
        
        self.assertIn("total_feedback", summary)
        self.assertIn("avg_satisfaction", summary)
        self.assertIn("avg_nutrition", summary)
        self.assertIn("avg_variety", summary)
        self.assertIn("feedback_trend", summary)


class TestConfig(unittest.TestCase):
    """Test configuration functionality."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = Config()
        
        self.assertEqual(config.algorithm, "PPO")
        self.assertIsNotNone(config.environment)
        self.assertIsNotNone(config.agent)
        self.assertIsNotNone(config.database)
        self.assertIsNotNone(config.feedback)
    
    def test_algorithm_config(self):
        """Test algorithm-specific configuration."""
        config = Config()
        algorithm_config = config.get_algorithm_config()
        
        self.assertIn("learning_rate", algorithm_config)
        self.assertIn("batch_size", algorithm_config)
        self.assertIn("gamma", algorithm_config)
        
        # Test PPO-specific parameters
        if config.algorithm == "PPO":
            self.assertIn("clip_range", algorithm_config)
            self.assertIn("ent_coef", algorithm_config)
            self.assertIn("vf_coef", algorithm_config)


if __name__ == "__main__":
    unittest.main() 