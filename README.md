# Macronutrient-Balancing-Planner-using-RL

An adaptive reinforcement learning agent for personalized macronutrient meal planning that learns optimal meal choices through trial and error.

## Features

- **Adaptive RL Agent**: Uses PPO, SAC, DDPG, and A2C algorithms for personalized meal planning
- **Reward Shaping**: Multi-objective reward function considering macro balance, variety, and satisfaction
- **Interactive Feedback Loop**: Continuous learning from user feedback to improve recommendations
- **Real-world Nutritional Data**: Integration with USDA/Open Food Facts databases
- **Personalized Profiles**: Considers dietary preferences, allergies, and calorie needs

## Project Structure

```
├── src/
│   ├── agents/           # RL agent implementations
│   ├── environment/      # Meal planning environment
│   ├── data/            # Data processing and nutritional databases
│   ├── models/          # Neural network models
│   ├── utils/           # Utility functions
│   └── feedback/        # User feedback processing
├── data/                # Nutritional databases and user data
├── configs/             # Configuration files
├── tests/               # Unit tests
└── examples/            # Usage examples
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.main import MealPlanningAgent

# Initialize the agent
agent = MealPlanningAgent()

# Generate personalized meal plan
meal_plan = agent.generate_meal_plan(user_profile)
```

## Algorithms

- **PPO (Proximal Policy Optimization)**: Stable policy updates with clipping
- **SAC (Soft Actor-Critic)**: Maximum entropy RL for exploration
- **DDPG (Deep Deterministic Policy Gradient)**: Continuous action spaces
- **A2C (Advantage Actor-Critic)**: On-policy learning with advantage estimation
