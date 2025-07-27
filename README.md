# Macronutrient Balancing Planner using Reinforcement Learning

A comprehensive reinforcement learning system for optimal meal planning that balances macronutrients while considering user preferences and nutritional quality.

## 🚀 Features

- **Real-world data integration** with Open Food Facts API (1000+ food items)
- **PPO reinforcement learning agent** for intelligent meal selection
- **Comprehensive evaluation metrics** and baseline comparisons
- **Interactive visualizations** and progress tracking
- **Flexible configuration management** with YAML
- **Modular architecture** for easy extension and maintenance

## 📁 Project Structure

```
├── config.yaml              # Main configuration file
├── main.py                   # Main training script
├── requirements.txt          # Dependencies
├── src/                      # Source code modules
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── data_loader.py       # Food data and API integration
│   ├── user_profile.py      # User profile management
│   ├── environment.py       # RL environment (to be created)
│   ├── agent.py             # RL agent implementation (to be created)
│   └── evaluation.py        # Evaluation metrics and baselines
├── cache/                   # Data cache directory
├── models/                  # Saved model directory
├── plots/                   # Generated visualizations
└── logs/                    # Training logs
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Macronutrient-Balancing-Planner-using-RL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

The system uses YAML configuration files for flexible parameter management. The main configuration file is `config.yaml`.

### Key Configuration Sections:

- **Data settings**: API configuration, caching, database management
- **Environment settings**: RL environment parameters, reward weights
- **Agent settings**: Neural network architecture, learning parameters
- **Training settings**: Episode counts, update frequency, checkpointing
- **Evaluation settings**: Metrics, baseline comparisons

### Example Configuration:

```yaml
# Environment settings
environment:
  max_steps_per_episode: 20
  target_calories: 2000
  macro_ratios:
    protein: 0.25
    carbohydrates: 0.45
    fat: 0.30

# Training settings
training:
  num_episodes: 1000
  update_frequency: 10
  batch_size: 64
```

## 🚀 Usage

### Basic Training

```bash
python main.py --config config.yaml --mode train
```

### Training with Custom Parameters

```bash
python main.py --config config.yaml --mode train --episodes 500 --verbose
```

### Command Line Options

- `--config, -c`: Path to configuration file (default: `config.yaml`)
- `--mode, -m`: Run mode - `train`, `evaluate`, or `demo` (default: `train`)
- `--episodes, -e`: Number of training episodes (overrides config)
- `--seed, -s`: Random seed for reproducibility (default: 42)
- `--verbose, -v`: Enable verbose logging
- `--no-baseline`: Skip baseline comparison
- `--refresh-data`: Force refresh of food database

### Training Output

The system generates:
- **Training logs** with progress updates
- **Interactive visualizations** showing training progress
- **Comparison charts** with baseline agents
- **Nutritional analysis** of meal plans
- **Performance metrics** and convergence analysis

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Performance Metrics
- **Average Reward**: Overall performance measure
- **Success Rate**: Episodes meeting success threshold
- **Calorie Accuracy**: How well calorie targets are met
- **Macro Balance Accuracy**: Macronutrient distribution accuracy
- **Meal Variety**: Diversity of food selections
- **Nutritional Quality**: Overall nutritional score

### Baseline Comparisons
- **Random Agent**: Random food selection baseline
- **Greedy Agent**: Macro-balance focused heuristic
- **Heuristic Agent**: Rule-based nutrition planning


## 🧪 Extending the System

### Adding New Agents

1. Create agent class inheriting from base agent interface
2. Implement required methods (`select_action`, `train`, etc.)
3. Add agent to configuration and main script

### Adding New Metrics

1. Extend `EvaluationMetrics` class in `src/evaluation.py`
2. Implement metric calculation methods
3. Update visualization components

### Custom User Profiles

```python
from src.user_profile import UserProfile, Gender, ActivityLevel, Goal

profile = UserProfile(
    age=25,
    gender=Gender.FEMALE,
    weight=60.0,
    height=165.0,
    activity_level=ActivityLevel.VERY_ACTIVE,
    goals={
        'weight_goal': Goal.LOSE_WEIGHT,
        'target_calories': 1800,
        'protein_ratio': 0.30,
        'carb_ratio': 0.40,
        'fat_ratio': 0.30
    }
)
```

## 🔧 Development

### Adding New Features

1. **Data Sources**: Extend `data_loader.py` for new nutrition APIs
2. **Environments**: Implement custom RL environments in `environment.py`
3. **Agents**: Add new RL algorithms in `agent.py`

### Testing

```bash
# Run with mock components for testing
python main.py --episodes 10 --verbose
```

### Debugging

Enable verbose logging and check log files:
```bash
python main.py --verbose
# Check logs in training.log
```
