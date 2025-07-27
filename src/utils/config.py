"""
Configuration management for the RL meal planning system.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    """Configuration for the meal planning environment."""
    max_steps: int = 50
    max_meals_per_day: int = 5
    min_calories: float = 1200.0
    max_calories: float = 3000.0
    target_protein_ratio: float = 0.25
    target_carbs_ratio: float = 0.45
    target_fats_ratio: float = 0.30
    variety_penalty: float = 0.1
    satisfaction_weight: float = 0.3
    nutrition_weight: float = 0.5
    variety_weight: float = 0.2


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 0.005
    target_update_freq: int = 1000
    policy_freq: int = 2
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class DatabaseConfig:
    """Configuration for nutritional database."""
    usda_api_key: Optional[str] = None
    openfoodfacts_url: str = "https://world.openfoodfacts.org/cgi/search.pl"
    cache_dir: str = "data/cache"
    update_frequency: int = 7  # days
    max_foods_per_category: int = 1000


@dataclass
class FeedbackConfig:
    """Configuration for feedback processing."""
    feedback_window: int = 30  # days
    min_feedback_threshold: int = 5
    feedback_decay_rate: float = 0.95
    satisfaction_threshold: float = 0.7


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or use defaults.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Initialize default values
        self.algorithm = "PPO"
        self.environment = EnvironmentConfig()
        self.agent = AgentConfig()
        self.database = DatabaseConfig()
        self.feedback = FeedbackConfig()
        
        # Training parameters
        self.training_episodes = 1000
        self.evaluation_episodes = 100
        self.save_frequency = 100
        
        # Logging
        self.log_level = "INFO"
        self.log_file = "logs/meal_planning.log"
        
        # Model paths
        self.model_dir = "models"
        self.checkpoint_dir = "checkpoints"
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update configuration with file data
        for key, value in config_data.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    # Handle nested configs
                    nested_config = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            # Convert string numbers to appropriate types
                            if isinstance(nested_value, str) and nested_value.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit():
                                try:
                                    if '.' in nested_value or 'e' in nested_value:
                                        nested_value = float(nested_value)
                                    else:
                                        nested_value = int(nested_value)
                                except ValueError:
                                    pass  # Keep as string if conversion fails
                            setattr(nested_config, nested_key, nested_value)
                else:
                    # Convert string numbers to appropriate types
                    if isinstance(value, str) and value.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit():
                        try:
                            if '.' in value or 'e' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass  # Keep as string if conversion fails
                    setattr(self, key, value)
    

    
    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file."""
        config_data = {
            'algorithm': self.algorithm,
            'environment': {
                'max_steps': self.environment.max_steps,
                'max_meals_per_day': self.environment.max_meals_per_day,
                'min_calories': self.environment.min_calories,
                'max_calories': self.environment.max_calories,
                'target_protein_ratio': self.environment.target_protein_ratio,
                'target_carbs_ratio': self.environment.target_carbs_ratio,
                'target_fats_ratio': self.environment.target_fats_ratio,
                'variety_penalty': self.environment.variety_penalty,
                'satisfaction_weight': self.environment.satisfaction_weight,
                'nutrition_weight': self.environment.nutrition_weight,
                'variety_weight': self.environment.variety_weight,
            },
            'agent': {
                'learning_rate': self.agent.learning_rate,
                'batch_size': self.agent.batch_size,
                'buffer_size': self.agent.buffer_size,
                'gamma': self.agent.gamma,
                'tau': self.agent.tau,
                'target_update_freq': self.agent.target_update_freq,
                'policy_freq': self.agent.policy_freq,
                'noise_clip': self.agent.noise_clip,
                'policy_noise': self.agent.policy_noise,
                'clip_range': self.agent.clip_range,
                'ent_coef': self.agent.ent_coef,
                'vf_coef': self.agent.vf_coef,
                'max_grad_norm': self.agent.max_grad_norm,
            },
            'database': {
                'usda_api_key': self.database.usda_api_key,
                'openfoodfacts_url': self.database.openfoodfacts_url,
                'cache_dir': self.database.cache_dir,
                'update_frequency': self.database.update_frequency,
                'max_foods_per_category': self.database.max_foods_per_category,
            },
            'feedback': {
                'feedback_window': self.feedback.feedback_window,
                'min_feedback_threshold': self.feedback.min_feedback_threshold,
                'feedback_decay_rate': self.feedback.feedback_decay_rate,
                'satisfaction_threshold': self.feedback.satisfaction_threshold,
            },
            'training_episodes': self.training_episodes,
            'evaluation_episodes': self.evaluation_episodes,
            'save_frequency': self.save_frequency,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'model_dir': self.model_dir,
            'checkpoint_dir': self.checkpoint_dir,
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """Get configuration specific to the selected algorithm."""
        base_config = {
            'learning_rate': self.agent.learning_rate,
            'batch_size': self.agent.batch_size,
            'gamma': self.agent.gamma,
        }
        
        if self.algorithm == "PPO":
            base_config.update({
                'clip_range': self.agent.clip_range,
                'ent_coef': self.agent.ent_coef,
                'vf_coef': self.agent.vf_coef,
                'max_grad_norm': self.agent.max_grad_norm,
            })
        elif self.algorithm == "SAC":
            base_config.update({
                'tau': self.agent.tau,
                'target_update_freq': self.agent.target_update_freq,
                'ent_coef': self.agent.ent_coef,
            })
        elif self.algorithm == "DDPG":
            base_config.update({
                'tau': self.agent.tau,
                'target_update_freq': self.agent.target_update_freq,
                'policy_freq': self.agent.policy_freq,
                'noise_clip': self.agent.noise_clip,
                'policy_noise': self.agent.policy_noise,
            })
        elif self.algorithm == "A2C":
            base_config.update({
                'ent_coef': self.agent.ent_coef,
                'vf_coef': self.agent.vf_coef,
                'max_grad_norm': self.agent.max_grad_norm,
            })
        
        return base_config 