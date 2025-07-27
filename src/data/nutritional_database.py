"""
Nutritional database management for food data.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta


@dataclass
class FoodItem:
    """Represents a food item with nutritional information."""
    food_id: str
    name: str
    category: str
    calories: float
    protein: float
    carbs: float
    fats: float
    fiber: float
    sugar: float
    sodium: float
    serving_size: str
    allergens: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.allergens is None:
            self.allergens = []
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'food_id': self.food_id,
            'name': self.name,
            'category': self.category,
            'calories': self.calories,
            'protein': self.protein,
            'carbs': self.carbs,
            'fats': self.fats,
            'fiber': self.fiber,
            'sugar': self.sugar,
            'sodium': self.sodium,
            'serving_size': self.serving_size,
            'allergens': self.allergens,
            'tags': self.tags
        }


class NutritionalDatabase:
    """
    Manages nutritional data from multiple sources including USDA and Open Food Facts.
    """
    
    def __init__(self, cache_dir: str = "data/cache", update_frequency: int = 7):
        self.cache_dir = cache_dir
        self.update_frequency = update_frequency
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize data storage
        self.foods: Dict[str, FoodItem] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # Load cached data
        self._load_cached_data()
        
        # Check if update is needed
        if self._should_update():
            self._update_database()
    
    def _load_cached_data(self):
        """Load data from cache files."""
        cache_file = os.path.join(self.cache_dir, "foods.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                for food_data in data:
                    food_item = FoodItem(**food_data)
                    self.foods[food_item.food_id] = food_item
                    
                    # Update categories
                    if food_item.category not in self.categories:
                        self.categories[food_item.category] = []
                    self.categories[food_item.category].append(food_item.food_id)
                
                self.logger.info(f"Loaded {len(self.foods)} foods from cache")
            except Exception as e:
                self.logger.error(f"Error loading cached data: {e}")
    
    def _should_update(self) -> bool:
        """Check if database should be updated."""
        timestamp_file = os.path.join(self.cache_dir, "last_update.txt")
        if not os.path.exists(timestamp_file):
            return True
        
        try:
            with open(timestamp_file, 'r') as f:
                last_update = datetime.fromisoformat(f.read().strip())
            return datetime.now() - last_update > timedelta(days=self.update_frequency)
        except Exception:
            return True
    
    def _update_database(self):
        """Update database from external sources."""
        self.logger.info("Updating nutritional database...")
        
        # Load sample data (in real implementation, this would fetch from APIs)
        self._load_sample_data()
        
        # Save to cache
        self._save_cache()
        
        # Update timestamp
        timestamp_file = os.path.join(self.cache_dir, "last_update.txt")
        with open(timestamp_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        self.logger.info("Database update completed")
    
    def _load_sample_data(self):
        """Load sample nutritional data for demonstration."""
        sample_foods = [
            {
                'food_id': 'chicken_breast',
                'name': 'Chicken Breast',
                'category': 'protein',
                'calories': 165,
                'protein': 31.0,
                'carbs': 0.0,
                'fats': 3.6,
                'fiber': 0.0,
                'sugar': 0.0,
                'sodium': 74.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['lean', 'high_protein']
            },
            {
                'food_id': 'salmon',
                'name': 'Salmon',
                'category': 'protein',
                'calories': 208,
                'protein': 25.0,
                'carbs': 0.0,
                'fats': 12.0,
                'fiber': 0.0,
                'sugar': 0.0,
                'sodium': 59.0,
                'serving_size': '100g',
                'allergens': ['fish'],
                'tags': ['omega3', 'healthy_fats']
            },
            {
                'food_id': 'brown_rice',
                'name': 'Brown Rice',
                'category': 'grains',
                'calories': 111,
                'protein': 2.6,
                'carbs': 23.0,
                'fats': 0.9,
                'fiber': 1.8,
                'sugar': 0.4,
                'sodium': 5.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['whole_grain', 'fiber']
            },
            {
                'food_id': 'quinoa',
                'name': 'Quinoa',
                'category': 'grains',
                'calories': 120,
                'protein': 4.4,
                'carbs': 22.0,
                'fats': 1.9,
                'fiber': 2.8,
                'sugar': 0.9,
                'sodium': 7.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['complete_protein', 'gluten_free']
            },
            {
                'food_id': 'broccoli',
                'name': 'Broccoli',
                'category': 'vegetables',
                'calories': 34,
                'protein': 2.8,
                'carbs': 7.0,
                'fats': 0.4,
                'fiber': 2.6,
                'sugar': 1.5,
                'sodium': 33.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['vitamin_c', 'fiber', 'antioxidants']
            },
            {
                'food_id': 'spinach',
                'name': 'Spinach',
                'category': 'vegetables',
                'calories': 23,
                'protein': 2.9,
                'carbs': 3.6,
                'fats': 0.4,
                'fiber': 2.2,
                'sugar': 0.4,
                'sodium': 79.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['iron', 'vitamin_k', 'low_calorie']
            },
            {
                'food_id': 'sweet_potato',
                'name': 'Sweet Potato',
                'category': 'vegetables',
                'calories': 86,
                'protein': 1.6,
                'carbs': 20.0,
                'fats': 0.1,
                'fiber': 3.0,
                'sugar': 4.2,
                'sodium': 55.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['vitamin_a', 'complex_carbs']
            },
            {
                'food_id': 'avocado',
                'name': 'Avocado',
                'category': 'fruits',
                'calories': 160,
                'protein': 2.0,
                'carbs': 9.0,
                'fats': 15.0,
                'fiber': 6.7,
                'sugar': 0.7,
                'sodium': 7.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['healthy_fats', 'fiber', 'potassium']
            },
            {
                'food_id': 'banana',
                'name': 'Banana',
                'category': 'fruits',
                'calories': 89,
                'protein': 1.1,
                'carbs': 23.0,
                'fats': 0.3,
                'fiber': 2.6,
                'sugar': 12.0,
                'sodium': 1.0,
                'serving_size': '100g',
                'allergens': [],
                'tags': ['potassium', 'natural_sugar']
            },
            {
                'food_id': 'greek_yogurt',
                'name': 'Greek Yogurt',
                'category': 'dairy',
                'calories': 59,
                'protein': 10.0,
                'carbs': 3.6,
                'fats': 0.4,
                'fiber': 0.0,
                'sugar': 3.2,
                'sodium': 36.0,
                'serving_size': '100g',
                'allergens': ['milk'],
                'tags': ['probiotics', 'high_protein']
            },
            {
                'food_id': 'eggs',
                'name': 'Eggs',
                'category': 'protein',
                'calories': 155,
                'protein': 13.0,
                'carbs': 1.1,
                'fats': 11.0,
                'fiber': 0.0,
                'sugar': 1.1,
                'sodium': 124.0,
                'serving_size': '100g',
                'allergens': ['eggs'],
                'tags': ['complete_protein', 'vitamin_d']
            },
            {
                'food_id': 'almonds',
                'name': 'Almonds',
                'category': 'nuts',
                'calories': 579,
                'protein': 21.0,
                'carbs': 22.0,
                'fats': 50.0,
                'fiber': 12.5,
                'sugar': 4.8,
                'sodium': 1.0,
                'serving_size': '100g',
                'allergens': ['tree_nuts'],
                'tags': ['healthy_fats', 'vitamin_e', 'magnesium']
            }
        ]
        
        for food_data in sample_foods:
            food_item = FoodItem(**food_data)
            self.foods[food_item.food_id] = food_item
            
            # Update categories
            if food_item.category not in self.categories:
                self.categories[food_item.category] = []
            self.categories[food_item.category].append(food_item.food_id)
    
    def _save_cache(self):
        """Save data to cache files."""
        cache_file = os.path.join(self.cache_dir, "foods.json")
        with open(cache_file, 'w') as f:
            json.dump([food.to_dict() for food in self.foods.values()], f, indent=2)
    
    def get_food(self, food_id: str) -> Optional[FoodItem]:
        """Get a specific food item."""
        return self.foods.get(food_id)
    
    def search_foods(self, query: str, limit: int = 10) -> List[FoodItem]:
        """Search for foods by name."""
        query = query.lower()
        results = []
        
        for food in self.foods.values():
            if query in food.name.lower():
                results.append(food)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_foods_by_category(self, category: str) -> List[FoodItem]:
        """Get all foods in a specific category."""
        if category not in self.categories:
            return []
        
        return [self.foods[food_id] for food_id in self.categories[category]]
    
    def get_categories(self) -> List[str]:
        """Get all available food categories."""
        return list(self.categories.keys())
    
    def filter_foods(self, 
                    min_calories: Optional[float] = None,
                    max_calories: Optional[float] = None,
                    min_protein: Optional[float] = None,
                    max_protein: Optional[float] = None,
                    min_carbs: Optional[float] = None,
                    max_carbs: Optional[float] = None,
                    min_fats: Optional[float] = None,
                    max_fats: Optional[float] = None,
                    excluded_allergens: Optional[List[str]] = None,
                    required_tags: Optional[List[str]] = None) -> List[FoodItem]:
        """
        Filter foods based on nutritional criteria and preferences.
        
        Args:
            min_calories: Minimum calories per serving
            max_calories: Maximum calories per serving
            min_protein: Minimum protein per serving
            max_protein: Maximum protein per serving
            min_carbs: Minimum carbs per serving
            max_carbs: Maximum carbs per serving
            min_fats: Minimum fats per serving
            max_fats: Maximum fats per serving
            excluded_allergens: List of allergens to exclude
            required_tags: List of tags that must be present
            
        Returns:
            List of FoodItem objects matching the criteria
        """
        filtered_foods = []
        
        for food in self.foods.values():
            # Check calorie range
            if min_calories is not None and food.calories < min_calories:
                continue
            if max_calories is not None and food.calories > max_calories:
                continue
            
            # Check protein range
            if min_protein is not None and food.protein < min_protein:
                continue
            if max_protein is not None and food.protein > max_protein:
                continue
            
            # Check carbs range
            if min_carbs is not None and food.carbs < min_carbs:
                continue
            if max_carbs is not None and food.carbs > max_carbs:
                continue
            
            # Check fats range
            if min_fats is not None and food.fats < min_fats:
                continue
            if max_fats is not None and food.fats > max_fats:
                continue
            
            # Check allergens
            if excluded_allergens:
                if any(allergen in food.allergens for allergen in excluded_allergens):
                    continue
            
            # Check required tags
            if required_tags:
                if not all(tag in food.tags for tag in required_tags):
                    continue
            
            filtered_foods.append(food)
        
        return filtered_foods
    
    def get_nutritional_summary(self, food_ids: List[str]) -> Dict[str, float]:
        """
        Get nutritional summary for a list of food items.
        
        Args:
            food_ids: List of food IDs
            
        Returns:
            Dictionary with total nutritional values
        """
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fats = 0
        total_fiber = 0
        total_sugar = 0
        total_sodium = 0
        
        for food_id in food_ids:
            food = self.foods.get(food_id)
            if food:
                total_calories += food.calories
                total_protein += food.protein
                total_carbs += food.carbs
                total_fats += food.fats
                total_fiber += food.fiber
                total_sugar += food.sugar
                total_sodium += food.sodium
        
        return {
            'calories': total_calories,
            'protein': total_protein,
            'carbs': total_carbs,
            'fats': total_fats,
            'fiber': total_fiber,
            'sugar': total_sugar,
            'sodium': total_sodium
        }
    
    def get_food_suggestions(self, 
                           target_calories: float,
                           target_protein: float,
                           target_carbs: float,
                           target_fats: float,
                           excluded_allergens: List[str] = None,
                           preferred_categories: List[str] = None,
                           max_suggestions: int = 20) -> List[FoodItem]:
        """
        Get food suggestions based on nutritional targets.
        
        Args:
            target_calories: Target calories
            target_protein: Target protein (g)
            target_carbs: Target carbs (g)
            target_fats: Target fats (g)
            excluded_allergens: Allergens to exclude
            preferred_categories: Preferred food categories
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested FoodItem objects
        """
        # Filter foods based on criteria
        filtered_foods = self.filter_foods(
            excluded_allergens=excluded_allergens
        )
        
        # Score foods based on how well they match targets
        scored_foods = []
        for food in filtered_foods:
            # Calculate how well this food fits the targets
            calorie_score = 1.0 - abs(food.calories - target_calories / 3) / target_calories
            protein_score = 1.0 - abs(food.protein - target_protein / 3) / target_protein
            carbs_score = 1.0 - abs(food.carbs - target_carbs / 3) / target_carbs
            fats_score = 1.0 - abs(food.fats - target_fats / 3) / target_fats
            
            # Category preference bonus
            category_bonus = 0.1 if preferred_categories and food.category in preferred_categories else 0.0
            
            # Calculate total score
            total_score = (calorie_score + protein_score + carbs_score + fats_score) / 4 + category_bonus
            
            scored_foods.append((food, total_score))
        
        # Sort by score and return top suggestions
        scored_foods.sort(key=lambda x: x[1], reverse=True)
        return [food for food, score in scored_foods[:max_suggestions]] 