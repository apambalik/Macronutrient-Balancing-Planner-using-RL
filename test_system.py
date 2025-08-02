#!/usr/bin/env python3
"""
Simple test script to verify the modular system components work together.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_config():
    """Test configuration management"""
    print("Testing configuration management...")
    from src.config import Config
    
    config = Config()
    print("✅ Configuration loaded successfully")
    print(f"   - Target calories: {config.env_target_calories}")
    print(f"   - Training episodes: {config.training_num_episodes}")
    return config

def test_data_components(config):
    """Test data loading components"""
    print("\nTesting data components...")
    from src.data_loader import OpenFoodFactsAPI, NutritionData, FoodItem
    
    # Test nutrition data
    nutrition = NutritionData(
        calories=250, protein=20, carbohydrates=30, fat=10
    )
    print("✅ NutritionData created successfully")
    print(f"   - Calories: {nutrition.calories}")
    
    # Test API (without actually making requests)
    api = OpenFoodFactsAPI(config)
    print("✅ OpenFoodFactsAPI initialized successfully")
    print(f"   - Base URL: {api.base_url}")
    
    return nutrition

def test_user_profile():
    """Test user profile management"""
    print("\nTesting user profile...")
    from src.user_profile import UserProfile, Gender, ActivityLevel, Goal
    
    profile = UserProfile(
        age=30,
        gender=Gender.MALE,
        weight=75.0,
        height=180.0,
        activity_level=ActivityLevel.MODERATELY_ACTIVE
    )
    
    print("✅ UserProfile created successfully")
    print(f"   - BMI: {profile.calculate_bmi():.1f}")
    print(f"   - BMR: {profile.calculate_bmr():.0f}")
    print(f"   - Daily calories: {profile.calculate_daily_calories()}")
    
    return profile

def test_evaluation():
    """Test evaluation components"""
    print("\nTesting evaluation components...")
    from src.evaluation import EvaluationMetrics, EvaluationResult
    from src.config import Config
    
    config = Config()
    evaluator = EvaluationMetrics(config)
    
    # Create mock evaluation result
    result = EvaluationResult(
        metrics={"avg_reward": 0.5, "success_rate": 0.7},
        episode_rewards=[0.1, 0.3, 0.5, 0.7, 0.9],
        episode_details=[],
        success_episodes=3,
        total_episodes=5
    )
    
    print("✅ EvaluationMetrics initialized successfully")
    print(f"   - Success rate: {result.get_success_rate():.2f}")
    print(f"   - Average reward: {result.get_average_reward():.2f}")
    
    return evaluator


def test_integration():
    """Test component integration"""
    print("\nTesting component integration...")
    from src.config import Config
    
    # Test that components can work together
    config = test_config()
    nutrition = test_data_components(config)
    profile = test_user_profile()
    evaluator = test_evaluation()
    
    print("\n✅ All components integrated successfully!")
    
    # Test configuration save/load
    config.save_yaml("test_config.yaml")
    loaded_config = Config.from_yaml("test_config.yaml")
    print("✅ Configuration save/load test passed")
    
    # Clean up
    Path("test_config.yaml").unlink(missing_ok=True)
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("MODULAR SYSTEM TEST")
    print("="*60)
    
    try:
        test_integration()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("The modular system is working correctly.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 