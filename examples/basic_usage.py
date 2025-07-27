"""
Basic usage example for the RL Meal Planning System.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import MealPlanningAgent
from src.data.user_profile import UserProfile, ActivityLevel, DietaryRestriction


def create_sample_user():
    """Create a sample user profile."""
    user = UserProfile(
        user_id="user_001",
        name="John Doe",
        age=30,
        gender="male",
        height=175.0,  # cm
        weight=70.0,   # kg
        activity_level=ActivityLevel.MODERATELY_ACTIVE
    )
    
    # Add dietary restrictions
    user.add_dietary_restriction(DietaryRestriction.VEGETARIAN)
    
    # Add allergies
    user.add_allergy("nuts")
    user.add_allergy("shellfish")
    
    # Add food preferences
    user.add_preferred_food("quinoa")
    user.add_preferred_food("broccoli")
    user.add_preferred_food("salmon")
    
    # Add disliked foods
    user.add_disliked_food("brussels sprouts")
    user.add_disliked_food("liver")
    
    return user


def main():
    """Main example function."""
    print("=== RL Meal Planning System - Basic Usage Example ===\n")
    
    # Initialize the meal planning agent
    print("1. Initializing meal planning agent...")
    agent = MealPlanningAgent("configs/default_config.yaml")
    print("✓ Agent initialized successfully\n")
    
    # Create a sample user profile
    print("2. Creating sample user profile...")
    user = create_sample_user()
    print(f"✓ User profile created for {user.name}")
    print(f"  - Age: {user.age}, Gender: {user.gender}")
    print(f"  - Height: {user.height}cm, Weight: {user.weight}kg")
    print(f"  - Activity Level: {user.activity_level.value}")
    print(f"  - Target Calories: {user.nutritional_goals.target_calories:.0f}")
    print(f"  - Dietary Restrictions: {[r.value for r in user.dietary_restrictions]}")
    print(f"  - Allergies: {list(user.allergies)}")
    print()
    
    # Generate a meal plan
    print("3. Generating personalized meal plan...")
    try:
        meal_plan = agent.generate_meal_plan(user)
        print("✓ Meal plan generated successfully")
        print(f"  - Total Calories: {meal_plan.total_calories:.0f}")
        print(f"  - Protein: {meal_plan.total_protein:.1f}g")
        print(f"  - Carbs: {meal_plan.total_carbs:.1f}g")
        print(f"  - Fats: {meal_plan.total_fats:.1f}g")
        print(f"  - Satisfaction Score: {meal_plan.satisfaction_score:.2f}")
        print(f"  - Variety Score: {meal_plan.variety_score:.2f}")
    except ValueError as e:
        print(f"⚠ {e}")
        print("  Training agent for demonstration...")
        agent.train(episodes=50)  # Quick training for demo
        meal_plan = agent.generate_meal_plan(user)
        print("✓ Meal plan generated successfully after training")
        print(f"  - Total Calories: {meal_plan.total_calories:.0f}")
        print(f"  - Protein: {meal_plan.total_protein:.1f}g")
        print(f"  - Carbs: {meal_plan.total_carbs:.1f}g")
        print(f"  - Fats: {meal_plan.total_fats:.1f}g")
        print(f"  - Satisfaction Score: {meal_plan.satisfaction_score:.2f}")
        print(f"  - Variety Score: {meal_plan.variety_score:.2f}")
    print()
    
    # Display the meals
    print("4. Generated Meals:")
    for i, meal in enumerate(meal_plan.meals, 1):
        print(f"  Meal {i}: {meal['name']}")
        print(f"    - Category: {meal['category']}")
        print(f"    - Calories: {meal['calories']:.0f}")
        print(f"    - Protein: {meal['protein']:.1f}g")
        print(f"    - Carbs: {meal['carbs']:.1f}g")
        print(f"    - Fats: {meal['fats']:.1f}g")
        print()
    
    # Simulate user feedback
    print("5. Simulating user feedback...")
    feedback = {
        'meal_plan_id': 'plan_001',
        'satisfaction_score': 0.8,
        'nutrition_score': 0.9,
        'variety_score': 0.7,
        'overall_rating': 4.0,
        'comments': 'Great meal plan! Loved the variety.',
        'specific_feedback': {
            'liked_foods': ['quinoa', 'salmon'],
            'disliked_foods': [],
            'nutritional_issues': [],
            'variety_issues': [],
            'preference_updates': {}
        }
    }
    
    agent.update_from_feedback(user.user_id, feedback)
    print("✓ Feedback processed and agent updated")
    print()
    
    # Generate another meal plan (should be improved based on feedback)
    print("6. Generating improved meal plan based on feedback...")
    improved_meal_plan = agent.generate_meal_plan(user)
    print("✓ Improved meal plan generated")
    print(f"  - Total Calories: {improved_meal_plan.total_calories:.0f}")
    print(f"  - Satisfaction Score: {improved_meal_plan.satisfaction_score:.2f}")
    print(f"  - Variety Score: {improved_meal_plan.variety_score:.2f}")
    print()
    
    # Train the agent (optional)
    print("7. Training the agent (optional)...")
    try:
        agent.train(episodes=100)  # Quick training for demonstration
        print("✓ Agent training completed")
    except Exception as e:
        print(f"⚠ Training skipped: {e}")
    print()
    
    print("=== Example completed successfully! ===")
    print("\nThe RL meal planning system can:")
    print("• Generate personalized meal plans based on user profiles")
    print("• Learn from user feedback to improve recommendations")
    print("• Support multiple RL algorithms (PPO, SAC, DDPG, A2C)")
    print("• Handle dietary restrictions, allergies, and preferences")
    print("• Provide nutritional analysis and variety scoring")


if __name__ == "__main__":
    main() 