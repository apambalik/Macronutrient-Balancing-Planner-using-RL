# Macronutrient Balancing Planner using Reinforcement Learning

A comprehensive reinforcement learning system for optimal meal planning that balances macronutrients while considering user preferences and nutritional quality. (Currently, implemeneted PPO only)

## 🚀 Features

- **Real-world data integration** with Open Food Facts API (900+ food items)
- **Hybrid-action PPO agent** for food and portion size selection
- **Customizable `gymnasium` environment** simulating daily meal progression
- **Comprehensive evaluation metrics** and baseline agent comparisons (Random, Greedy, Heuristic)
- **Interactive analysis notebook** (`interactive_analysis.ipynb`) for visualizing results, generating meal plans, and experimenting with user profiles.
- **Flexible configuration management** with `config.yaml`
- **Modular and extensible architecture** for easy maintenance and new feature integration.

## 📁 Project Structure

```
├── config.yaml              # Main configuration file
├── main.py                  # Main training script
├── interactive_analysis.ipynb # Interactive notebook for analysis and demos
├── requirements.txt         # Project dependencies
├── src/                     # Source code modules
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── data_loader.py       # Food data and API integration
│   ├── user_profile.py      # User profile management
│   ├── environment.py       # Custom RL environment
│   ├── agent.py             # PPO agent implementation
│   └── evaluation.py        # Evaluation metrics and baselines
├── cache/                   # Cached food database
├── models/                  # Saved model checkpoints
├── training.log             # Log file for training sessions
└── README.md                # This README file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/apambalik/Macronutrient-Balancing-Planner-using-RL.git
   cd Macronutrient-Balancing-Planner-using-RL
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

The system uses `config.yaml` for flexible parameter management. You can adjust reward weights, agent hyperparameters, and training settings in this file.

### Key Configuration Sections:

- **`data`**: API settings, caching, and database management.
- **`environment`**: RL environment parameters, reward function weights, and success thresholds.
- **`agent`**: PPO agent hyperparameters (learning rate, gamma, clipping, etc.) and neural network architecture.
- **`training`**: Episode counts, update frequencies, and logging settings.
- **`evaluation`**: Number of episodes for evaluation and baseline agent toggles.

## 🚀 Usage

### 1. Training the Agent

Run the main script to start the training process. This will use the settings from `config.yaml` and save the trained model to the `models/` directory.

```bash
python main.py
```

You can override configuration settings with command-line arguments:

```bash
# Train for 500 episodes with verbose logging
python main.py --episodes 500 --verbose
```

### 2. Interactive Analysis and Demos

The `interactive_analysis.ipynb` notebook is the best way to explore the system's capabilities.

1. **Launch Jupyter Notebook or Jupyter Lab**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open `interactive_analysis.ipynb`** and run the cells sequentially.

The notebook allows you to:
- Visualize the agent's training progress.
- Compare the RL agent's performance against baseline models.
- Generate a sample meal plan with detailed nutritional analysis.
- Explore the food database with interactive charts.
- Experiment with different user profiles to see how nutritional needs change in real-time.

## 🧪 Extending the System

### Custom User Profiles

You can easily create custom user profiles for the agent to plan for. The `UserProfile` class is robust and handles default values, so you only need to specify what you want to change.

```python
from src.user_profile import UserProfile, Gender, ActivityLevel, Goal

# Example of a custom user profile
profile = UserProfile(
    age=25,
    gender=Gender.FEMALE,
    weight=60.0,
    height=165.0,
    activity_level=ActivityLevel.VERY_ACTIVE,
    goals={
        'weight_goal': Goal.LOSE_WEIGHT,
        'protein_ratio': 0.30,
        'carb_ratio': 0.40,
        'fat_ratio': 0.30
    }
)
```

### Adding New Features

- **Data Sources**: Extend `data_loader.py` to connect to new nutrition APIs.
- **Reward Functions**: Modify the `_calculate_reward` method in `src/environment.py` to change the agent's learning incentives.
- **Agents**: Add new RL algorithms by creating new agent classes in `agent.py` that conform to the expected interface.

## 🔧 Development and Debugging

- **Enable verbose logging** during training to see detailed output from the agent and environment:
  ```bash
  python main.py --verbose
  ```
- **Check `training.log`** for a persistent record of the training session.
