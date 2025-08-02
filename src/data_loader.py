"""
Data loading and management components for the Meal Planning RL system.

This module handles:
- OpenFoodFacts API integration
- Food data structures and validation
- Database management with caching
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenFoodFactsAPI:
    """Interface for Open Food Facts API with caching and error handling"""
    
    def __init__(self, config):
        self.base_url = config.data_api_base_url
        self.timeout = config.data_api_timeout
        self.max_retries = config.data_api_max_retries
        self.batch_size = config.data_api_batch_size
        
        cache_path = Path(config.data_cache_dir) / config.data_food_cache_file
        self.cache_file = str(cache_path)
        self.cache = self._load_cache()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RL-MealPlanner/1.0 (Educational Research)'
        })
        
        # Create cache directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> Dict:
        """Load cached food data"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    logger.info(f"Loaded {len(cache_data.get('products', {}))} items from cache")
                    return cache_data
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
        
        return {'products': {}, 'last_updated': None}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            self.cache['last_updated'] = datetime.now().isoformat()
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def search_products(self, query: str, limit: int = 50, 
                       lang: str = 'en') -> List[Dict]:
        """Search for products using Open Food Facts API"""
        
        cache_key = f"search_{query}_{limit}_{lang}"
        if cache_key in self.cache['products']:
            logger.debug(f"Using cached results for query: {query}")
            return self.cache['products'][cache_key]
        
        url = f"{self.base_url}/cgi/search.pl"
        params = {
            'search_terms': query,
            'search_simple': 1,
            'action': 'process',
            'json': 1,
            'page_size': min(limit, 100),
            'fields': 'code,product_name,brands,categories_tags,nutriments,ingredients_text,allergens_tags,nutrition_grades'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                products = data.get('products', [])
                
                # Filter products with sufficient nutritional data
                filtered_products = []
                for product in products:
                    nutriments = product.get('nutriments', {})
                    if self._has_sufficient_nutrition_data(nutriments):
                        filtered_products.append(product)
                
                # Cache results
                self.cache['products'][cache_key] = filtered_products
                self._save_cache()
                
                logger.info(f"Found {len(filtered_products)} products for query: {query}")
                return filtered_products
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for query: {query}")
                    return []
        
        return []
    
    def get_product_by_barcode(self, barcode: str) -> Optional[Dict]:
        """Get product by barcode"""
        cache_key = f"barcode_{barcode}"
        if cache_key in self.cache['products']:
            return self.cache['products'][cache_key]
        
        url = f"{self.base_url}/api/v0/product/{barcode}.json"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 1:
                product = data.get('product', {})
                self.cache['products'][cache_key] = product
                self._save_cache()
                return product
        except Exception as e:
            logger.error(f"Error fetching product {barcode}: {e}")
        
        return None
    
    def _has_sufficient_nutrition_data(self, nutriments: Dict) -> bool:
        """Check if product has sufficient nutritional data"""
        required_fields = ['energy-kcal_100g', 'proteins_100g', 'carbohydrates_100g', 'fat_100g']
        return all(
            field in nutriments and 
            nutriments[field] is not None and 
            nutriments[field] != "" and
            isinstance(nutriments[field], (int, float))
            for field in required_fields
        )
    
    def batch_search_categories(self, categories: List[str], 
                              items_per_category: int = 20) -> List[Dict]:
        """Search for products across multiple categories"""
        all_products = []
        
        for category in categories:
            try:
                products = self.search_products(category, items_per_category)
                all_products.extend(products)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Error searching category {category}: {e}")
        
        # Remove duplicates based on product code
        seen_codes = set()
        unique_products = []
        for product in all_products:
            code = product.get('code')
            if code and code not in seen_codes:
                seen_codes.add(code)
                unique_products.append(product)
        
        logger.info(f"Retrieved {len(unique_products)} unique products from {len(categories)} categories")
        return unique_products


@dataclass
class NutritionData:
    """Standardized nutrition data per 100g"""
    calories: float
    protein: float
    carbohydrates: float
    fat: float
    fiber: float = 0.0
    sugar: float = 0.0
    sodium: float = 0.0
    saturated_fat: float = 0.0
    
    def __post_init__(self):
        """Validate and ensure non-negative values"""
        for field_name in ['calories', 'protein', 'carbohydrates', 'fat', 'fiber', 'sugar', 'sodium', 'saturated_fat']:
            value = getattr(self, field_name)
            if value < 0:
                setattr(self, field_name, 0.0)


@dataclass
class FoodItem:
    """Enhanced food item with real-world data"""
    id: str
    name: str
    brand: str
    nutrition: NutritionData
    categories: List[str]
    ingredients: str
    allergens: List[str]
    nutrition_grade: str  # A, B, C, D, E
    quality_score: float
    
    @classmethod
    def from_openfoodfacts(cls, product_data: Dict) -> 'FoodItem':
        """Create FoodItem from Open Food Facts data"""
        nutriments = product_data.get('nutriments', {})
        
        # Extract nutrition data with fallbacks
        nutrition = NutritionData(
            calories=nutriments.get('energy-kcal_100g', 0),
            protein=nutriments.get('proteins_100g', 0),
            carbohydrates=nutriments.get('carbohydrates_100g', 0),
            fat=nutriments.get('fat_100g', 0),
            fiber=nutriments.get('fiber_100g', 0),
            sugar=nutriments.get('sugars_100g', 0),
            sodium=nutriments.get('sodium_100g', 0),
            saturated_fat=nutriments.get('saturated-fat_100g', 0)
        )
        
        return cls(
            id=product_data.get('code', ''),
            name=product_data.get('product_name', 'Unknown'),
            brand=product_data.get('brands', 'Unknown'),
            nutrition=nutrition,
            categories=product_data.get('categories_tags', []),
            ingredients=product_data.get('ingredients_text', ''),
            allergens=product_data.get('allergens_tags', []),
            nutrition_grade=product_data.get('nutrition_grades', 'unknown'),
            quality_score=cls._calculate_quality_score(nutriments)
        )
    
    @staticmethod
    def _calculate_quality_score(nutriments: Dict) -> float:
        """Calculate a quality score based on nutritional completeness"""
        essential_fields = ['energy-kcal_100g', 'proteins_100g', 'carbohydrates_100g', 'fat_100g']
        optional_fields = ['fiber_100g', 'sugars_100g', 'sodium_100g', 'saturated-fat_100g']
        
        essential_score = sum(1 for field in essential_fields if field in nutriments) / len(essential_fields)
        optional_score = sum(1 for field in optional_fields if field in nutriments) / len(optional_fields)
        
        return 0.7 * essential_score + 0.3 * optional_score


class RealFoodDatabase:
    """Enhanced food database using real Open Food Facts data"""
    
    def __init__(self, api: OpenFoodFactsAPI, config):
        self.api = api
        self.cache_file = Path(config.data_cache_dir) / config.data_food_database_file
        self.foods: Dict[str, FoodItem] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # Create cache directory
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing database
        self._load_database()
    
    def _load_database(self):
        """Load food database from cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.foods = data.get('foods', {})
                    self.categories = data.get('categories', {})
                
                logger.info(f"Loaded {len(self.foods)} foods from cache")
            except Exception as e:
                logger.error(f"Error loading database: {e}")
    
    def _save_database(self):
        """Save food database to cache"""
        try:
            data = {
                'foods': self.foods,
                'categories': self.categories,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved database with {len(self.foods)} foods")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def populate_database(self, force_refresh: bool = False):
        """Populate database with real food data"""
        if self.foods and not force_refresh:
            logger.info(f"Database already populated with {len(self.foods)} foods")
            return
        
        categories = [
            'chicken', 'beef', 'fish', 'eggs', 'tofu',  # Proteins
            'rice', 'pasta', 'bread', 'oats', 'quinoa',  # Carbs
            'apple', 'banana', 'broccoli', 'spinach',  # Fruits/Vegetables
            'milk', 'cheese', 'yogurt',  # Dairy
            'olive oil', 'nuts', 'avocado'  # Fats
        ]
        
        logger.info("Fetching food data from Open Food Facts...")
        
        for category in categories:
            try:
                products = self.api.search_products(category, limit=50)
                
                for product in products:
                    try:
                        food_item = FoodItem.from_openfoodfacts(product)
                        if food_item.id and food_item.nutrition.calories > 0:
                            self.foods[food_item.id] = food_item
                            
                            # Categorize foods
                            if category not in self.categories:
                                self.categories[category] = []
                            self.categories[category].append(food_item.id)
                    
                    except Exception as e:
                        logger.debug(f"Error processing product: {e}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching category {category}: {e}")
        
        self._save_database()
        logger.info(f"Database populated with {len(self.foods)} total food items")
    
    def get_food_by_id(self, food_id: str) -> Optional[FoodItem]:
        """Get food item by ID"""
        return self.foods.get(food_id)
    
    def get_foods_by_category(self, category: str) -> List[FoodItem]:
        """Get all foods in a category"""
        food_ids = self.categories.get(category, [])
        return [self.foods[food_id] for food_id in food_ids if food_id in self.foods]
    
    def search_foods(self, query: str, max_results: int = 20) -> List[FoodItem]:
        """Search foods by name"""
        query_lower = query.lower()
        results = []
        
        for food in self.foods.values():
            if (query_lower in food.name.lower() or 
                query_lower in food.brand.lower() or
                any(query_lower in cat.lower() for cat in food.categories)):
                results.append(food)
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def get_random_foods(self, count: int = 10, seed: Optional[int] = None) -> List[FoodItem]:
        """Get random food items"""
        if seed is not None:
            np.random.seed(seed)
        
        food_ids = list(self.foods.keys())
        if count >= len(food_ids):
            return list(self.foods.values())
        
        selected_ids = np.random.choice(food_ids, size=count, replace=False)
        return [self.foods[food_id] for food_id in selected_ids]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.foods:
            return {}
        
        # Calculate nutrition grade distribution
        grade_dist = {}
        for food in self.foods.values():
            grade = food.nutrition_grade
            grade_dist[grade] = grade_dist.get(grade, 0) + 1
        
        # Calculate average nutrition values
        total_foods = len(self.foods)
        avg_nutrition = {
            'calories': np.mean([f.nutrition.calories for f in self.foods.values()]),
            'protein': np.mean([f.nutrition.protein for f in self.foods.values()]),
            'carbohydrates': np.mean([f.nutrition.carbohydrates for f in self.foods.values()]),
            'fat': np.mean([f.nutrition.fat for f in self.foods.values()])
        }
        
        return {
            'total_foods': total_foods,
            'categories': list(self.categories.keys()),
            'category_counts': {cat: len(foods) for cat, foods in self.categories.items()},
            'nutrition_grade_distribution': grade_dist,
            'average_nutrition': avg_nutrition
        } 