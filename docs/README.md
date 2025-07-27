# RL Meal Planning System Documentation

## Overview

The RL Meal Planning System is an adaptive reinforcement learning agent for personalized macronutrient meal planning. The system learns optimal meal choices through trial and error, incorporating user feedback and real-world nutritional data to provide dynamic, user-specific dietary recommendations.

## Key Features

### 1. Adaptive RL Agent
- **Multiple Algorithms**: Supports PPO, SAC, DDPG, and A2C algorithms
- **Personalized Learning**: Adapts to individual user preferences and dietary requirements
- **Continuous Improvement**: Learns from user feedback to improve recommendations over time

### 2. Reward Shaping Mechanisms
- **Multi-objective Rewards**: Balances nutrition, satisfaction, and variety
- **Nutritional Balance**: Penalizes macro imbalance and encourages balanced meals
- **User Satisfaction**: Incorporates user preferences and dietary restrictions
- **Variety Encouragement**: Promotes diverse food selection while avoiding repetition

### 3. Interactive Feedback Loop
- **User Feedback Processing**: Collects and processes user ratings and comments
- **Continuous Learning**: Updates agent parameters based on feedback
- **Feedback Analysis**: Provides insights into user preferences and system performance

### 4. Real-world Nutritional Data
- **Comprehensive Database**: Includes 12+ food categories with accurate nutritional information
- **Allergen Management**: Handles food allergies and dietary restrictions
- **Nutritional Accuracy**: Provides detailed macronutrient and micronutrient information

## System Architecture

### Core Components

1. **MealPlanningAgent** (`src/main.py`)
   - Main orchestrator for the RL meal planning system
   - Manages user profiles, generates meal plans, and processes feedback

2. **User Profile System** (`src/data/user_profile.py`)
   - Handles user demographics, dietary preferences, and nutritional goals
   - Calculates personalized nutritional targets using BMR and activity levels
   - Manages dietary restrictions, allergies, and food preferences

3. **Nutritional Database** (`src/data/nutritional_database.py`)
   - Manages food data with comprehensive nutritional information
   - Supports filtering by nutritional criteria and user preferences
   - Provides food suggestions based on nutritional targets

4. **RL Environment** (`src/environment/meal_planning_env.py`)
   - Gym-compatible environment for reinforcement learning
   - Defines state space, action space, and reward function
   - Handles meal planning episodes and nutritional tracking

5. **RL Agents** (`src/agents/`)
   - **PPO Agent**: Stable policy updates with clipping
   - **SAC Agent**: Maximum entropy RL for exploration
   - **DDPG Agent**: Continuous action spaces
   - **A2C Agent**: On-policy learning with advantage estimation

6. **Feedback Processor** (`src/feedback/feedback_processor.py`)
   - Processes user feedback and updates agent parameters
   - Analyzes feedback trends and provides insights
   - Maintains feedback history for continuous improvement

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Stable-Baselines3 2.1+

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Macronutrient-Balancing-Planner-using-RL

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
from src.main import MealPlanningAgent
from src.data.user_profile import UserProfile, ActivityLevel

# Initialize the agent
agent = MealPlanningAgent("configs/default_config.yaml")

# Create a user profile
user = UserProfile(
    user_id="user_001",
    name="John Doe",
    age=30,
    gender="male",
    height=175.0,
    weight=70.0,
    activity_level=ActivityLevel.MODERATELY_ACTIVE
)

# Generate a meal plan
meal_plan = agent.generate_meal_plan(user)
print(f"Generated meal plan with {meal_plan.total_calories:.0f} calories")
```

## Configuration

### Algorithm Selection
The system supports four RL algorithms, each with different characteristics:

| Algorithm | Type | Strengths | Best For |
|-----------|------|-----------|----------|
| PPO | On-policy | Stable, reliable | General meal planning |
| SAC | Off-policy | Exploration, continuous actions | Complex preferences |
| DDPG | Off-policy | Continuous actions | Fine-grained control |
| A2C | On-policy | Simple, fast | Quick iterations |

### Environment Configuration
```yaml
environment:
  max_steps: 50                    # Maximum steps per episode
  max_meals_per_day: 5             # Maximum meals to plan
  min_calories: 1200.0             # Minimum daily calories
  max_calories: 3000.0             # Maximum daily calories
  target_protein_ratio: 0.25       # Target protein ratio
  target_carbs_ratio: 0.45         # Target carbs ratio
  target_fats_ratio: 0.30          # Target fats ratio
  satisfaction_weight: 0.3          # Weight for satisfaction in reward
  nutrition_weight: 0.5            # Weight for nutrition in reward
  variety_weight: 0.2              # Weight for variety in reward
```

### Agent Configuration
```yaml
agent:
  learning_rate: 3e-4              # Learning rate for training
  batch_size: 64                   # Batch size for training
  gamma: 0.99                      # Discount factor
  # Algorithm-specific parameters...
```

## Usage Examples

### Basic Meal Planning
```python
# Create user profile with dietary restrictions
user = UserProfile(
    user_id="user_001",
    name="Jane Doe",
    age=28,
    gender="female",
    height=165.0,
    weight=60.0,
    activity_level=ActivityLevel.LIGHTLY_ACTIVE
)

# Add dietary preferences
user.add_dietary_restriction(DietaryRestriction.VEGETARIAN)
user.add_allergy("nuts")
user.add_preferred_food("quinoa")
user.add_disliked_food("brussels sprouts")

# Generate meal plan
agent = MealPlanningAgent()
meal_plan = agent.generate_meal_plan(user)

# Display results
print(f"Total Calories: {meal_plan.total_calories:.0f}")
print(f"Protein: {meal_plan.total_protein:.1f}g")
print(f"Carbs: {meal_plan.total_carbs:.1f}g")
print(f"Fats: {meal_plan.total_fats:.1f}g")
```

### Feedback Integration
```python
# Provide user feedback
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

# Update agent with feedback
agent.update_from_feedback(user.user_id, feedback)

# Generate improved meal plan
improved_plan = agent.generate_meal_plan(user)
```

### Training the Agent
```python
# Train the agent
agent.train(episodes=1000)

# Save the trained model
agent.save_model("models/trained_agent")

# Load a trained model
agent.load_model("models/trained_agent")
```

## Reward Function Design

The reward function is designed to balance multiple objectives:

### Nutrition Reward
- Encourages meeting macronutrient targets
- Penalizes over-consumption
- Rewards balanced nutritional profiles

### Satisfaction Reward
- Incorporates user preferences
- Respects dietary restrictions
- Considers food allergies

### Variety Reward
- Promotes diverse food selection
- Penalizes repetitive choices
- Encourages exploration of different food categories

### Combined Reward
```
Total Reward = w₁ × Nutrition + w₂ × Satisfaction + w₃ × Variety
```

Where w₁, w₂, w₃ are configurable weights (default: 0.5, 0.3, 0.2).

## Nutritional Database

### Food Categories
- **Protein**: Chicken, salmon, eggs, Greek yogurt
- **Grains**: Brown rice, quinoa
- **Vegetables**: Broccoli, spinach, sweet potato
- **Fruits**: Avocado, banana
- **Dairy**: Greek yogurt
- **Nuts**: Almonds

### Nutritional Information
Each food item includes:
- Calories per 100g
- Protein, carbs, fats (grams)
- Fiber, sugar, sodium
- Allergen information
- Dietary tags (vegetarian, vegan, gluten-free)

### Filtering Capabilities
- Nutritional range filtering
- Allergen exclusion
- Dietary restriction compliance
- Category-based selection

## Testing

Run the test suite to verify system functionality:

```bash
python -m pytest tests/
```

Or run specific test modules:

```bash
python tests/test_meal_planning.py
```

## Performance Metrics

### Training Metrics
- Episode rewards
- Nutritional balance scores
- User satisfaction scores
- Variety scores

### Evaluation Metrics
- Nutritional target achievement
- User preference satisfaction
- Meal plan variety
- Feedback response time

## Future Enhancements

### Planned Features
1. **Real-time API Integration**: Connect to USDA and Open Food Facts APIs
2. **Advanced Personalization**: Machine learning-based preference learning
3. **Mobile Interface**: Web and mobile applications
4. **Social Features**: Recipe sharing and community recommendations
5. **Advanced Analytics**: Detailed nutritional insights and trends

### Research Directions
1. **Multi-objective RL**: Advanced algorithms for complex reward functions
2. **Federated Learning**: Privacy-preserving collaborative learning
3. **Explainable AI**: Interpretable meal planning decisions
4. **Long-term Planning**: Weekly and monthly meal planning

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add type hints
- Include docstrings
- Write comprehensive tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples

## Acknowledgments

- Stable-Baselines3 for RL algorithms
- Gymnasium for environment framework
- USDA for nutritional data inspiration
- Open Food Facts for food database inspiration 