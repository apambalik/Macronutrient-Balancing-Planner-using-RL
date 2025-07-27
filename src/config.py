"""
Configuration management for the Meal Planning RL system.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class that loads and manages all settings."""
    
    # Data settings
    data_cache_dir: str = "cache"
    data_food_cache_file: str = "food_cache.json"
    data_food_database_file: str = "food_database.pkl"
    data_api_base_url: str = "https://world.openfoodfacts.org"
    data_api_timeout: int = 30
    data_api_max_retries: int = 3
    data_api_batch_size: int = 50
    
    # Environment settings
    env_max_steps_per_episode: int = 20
    env_max_daily_meals: int = 5
    env_target_calories: int = 2000
    env_macro_ratios: Dict[str, float] = field(default_factory=lambda: {
        'protein': 0.25, 'carbohydrates': 0.45, 'fat': 0.30
    })
    env_reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'macro_balance': 1.0, 'calorie_target': 1.0, 'variety': 0.5, 'nutritional_quality': 0.3
    })
    env_penalties: Dict[str, float] = field(default_factory=lambda: {
        'calorie_excess': -2.0, 'calorie_deficit': -1.5, 'macro_imbalance': -1.0
    })
    
    # Agent settings
    agent_algorithm: str = "PPO"
    agent_hidden_size: int = 256
    agent_learning_rate: float = 0.0003
    agent_gamma: float = 0.99
    agent_tau: float = 0.95
    agent_eps_clip: float = 0.2
    agent_value_coef: float = 0.5
    agent_entropy_coef: float = 0.01
    agent_max_grad_norm: float = 0.5
    
    # Training settings
    training_num_episodes: int = 1000
    training_update_frequency: int = 10
    training_batch_size: int = 64
    training_epochs_per_update: int = 10
    training_save_frequency: int = 100
    training_log_frequency: int = 10
    training_success_threshold: float = 0.8
    
    # Evaluation settings
    eval_num_episodes: int = 20
    eval_frequency: int = 100
    eval_metrics: list = field(default_factory=lambda: [
        "average_reward", "success_rate", "macro_balance_accuracy", 
        "calorie_accuracy", "meal_variety", "nutritional_quality"
    ])
    
    # Baseline settings
    baseline_random_enabled: bool = True
    baseline_random_seed: int = 42
    baseline_greedy_enabled: bool = True
    baseline_greedy_macro_priority: list = field(default_factory=lambda: [0.25, 0.45, 0.30])
    baseline_heuristic_enabled: bool = True
    baseline_heuristic_rules: list = field(default_factory=lambda: [
        "prioritize_protein", "balance_macros", "minimize_calories"
    ])
    
    # Visualization settings
    viz_plot_style: str = "plotly"
    viz_save_path: str = "plots"
    viz_formats: list = field(default_factory=lambda: ["html", "png"])
    viz_interactive: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "training.log"
    log_console_output: bool = True
    
    # Model settings
    model_save_path: str = "models"
    model_checkpoint_frequency: int = 50
    model_keep_best_only: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Create config instance with default values
            config = cls()
            
            # Update with YAML values using proper mapping
            config._update_from_yaml(yaml_config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration.")
            return cls()
    
    def _update_from_yaml(self, yaml_config: Dict[str, Any]):
        """Update configuration from YAML structure."""
        # Data settings
        if 'data' in yaml_config:
            data_config = yaml_config['data']
            self.data_cache_dir = data_config.get('cache_dir', self.data_cache_dir)
            self.data_food_cache_file = data_config.get('food_cache_file', self.data_food_cache_file)
            self.data_food_database_file = data_config.get('food_database_file', self.data_food_database_file)
            
            if 'api' in data_config:
                api_config = data_config['api']
                self.data_api_base_url = api_config.get('base_url', self.data_api_base_url)
                self.data_api_timeout = api_config.get('timeout', self.data_api_timeout)
                self.data_api_max_retries = api_config.get('max_retries', self.data_api_max_retries)
                self.data_api_batch_size = api_config.get('batch_size', self.data_api_batch_size)
        
        # Environment settings
        if 'environment' in yaml_config:
            env_config = yaml_config['environment']
            self.env_max_steps_per_episode = env_config.get('max_steps_per_episode', self.env_max_steps_per_episode)
            self.env_max_daily_meals = env_config.get('max_daily_meals', self.env_max_daily_meals)
            self.env_target_calories = env_config.get('target_calories', self.env_target_calories)
            
            if 'macro_ratios' in env_config:
                macro_config = env_config['macro_ratios']
                self.env_macro_ratios.update(macro_config)
            
            if 'rewards' in env_config:
                reward_config = env_config['rewards']
                self.env_reward_weights.update(reward_config)
            
            if 'penalties' in env_config:
                penalty_config = env_config['penalties']
                self.env_penalties.update(penalty_config)
        
        # Agent settings
        if 'agent' in yaml_config:
            agent_config = yaml_config['agent']
            self.agent_algorithm = agent_config.get('algorithm', self.agent_algorithm)
            self.agent_hidden_size = agent_config.get('hidden_size', self.agent_hidden_size)
            self.agent_learning_rate = agent_config.get('learning_rate', self.agent_learning_rate)
            self.agent_gamma = agent_config.get('gamma', self.agent_gamma)
            self.agent_tau = agent_config.get('tau', self.agent_tau)
            self.agent_eps_clip = agent_config.get('eps_clip', self.agent_eps_clip)
            self.agent_value_coef = agent_config.get('value_coef', self.agent_value_coef)
            self.agent_entropy_coef = agent_config.get('entropy_coef', self.agent_entropy_coef)
            self.agent_max_grad_norm = agent_config.get('max_grad_norm', self.agent_max_grad_norm)
        
        # Training settings
        if 'training' in yaml_config:
            training_config = yaml_config['training']
            self.training_num_episodes = training_config.get('num_episodes', self.training_num_episodes)
            self.training_update_frequency = training_config.get('update_frequency', self.training_update_frequency)
            self.training_batch_size = training_config.get('batch_size', self.training_batch_size)
            self.training_epochs_per_update = training_config.get('epochs_per_update', self.training_epochs_per_update)
            self.training_save_frequency = training_config.get('save_frequency', self.training_save_frequency)
            self.training_log_frequency = training_config.get('log_frequency', self.training_log_frequency)
            self.training_success_threshold = training_config.get('success_threshold', self.training_success_threshold)
        
        # Evaluation settings
        if 'evaluation' in yaml_config:
            eval_config = yaml_config['evaluation']
            self.eval_num_episodes = eval_config.get('num_eval_episodes', self.eval_num_episodes)
            self.eval_frequency = eval_config.get('eval_frequency', self.eval_frequency)
            if 'metrics' in eval_config:
                self.eval_metrics = eval_config['metrics']
        
        # Baseline settings
        if 'baselines' in yaml_config:
            baseline_config = yaml_config['baselines']
            if 'random' in baseline_config:
                random_config = baseline_config['random']
                self.baseline_random_enabled = random_config.get('enabled', self.baseline_random_enabled)
                self.baseline_random_seed = random_config.get('seed', self.baseline_random_seed)
            
            if 'greedy' in baseline_config:
                greedy_config = baseline_config['greedy']
                self.baseline_greedy_enabled = greedy_config.get('enabled', self.baseline_greedy_enabled)
                if 'macro_priority' in greedy_config:
                    self.baseline_greedy_macro_priority = greedy_config['macro_priority']
            
            if 'heuristic' in baseline_config:
                heuristic_config = baseline_config['heuristic']
                self.baseline_heuristic_enabled = heuristic_config.get('enabled', self.baseline_heuristic_enabled)
                if 'rules' in heuristic_config:
                    self.baseline_heuristic_rules = heuristic_config['rules']
        
        # Visualization settings
        if 'visualization' in yaml_config:
            viz_config = yaml_config['visualization']
            self.viz_plot_style = viz_config.get('plot_style', self.viz_plot_style)
            self.viz_save_path = viz_config.get('save_path', self.viz_save_path)
            if 'formats' in viz_config:
                self.viz_formats = viz_config['formats']
            self.viz_interactive = viz_config.get('interactive', self.viz_interactive)
        
        # Logging settings
        if 'logging' in yaml_config:
            log_config = yaml_config['logging']
            self.log_level = log_config.get('level', self.log_level)
            self.log_file = log_config.get('log_file', self.log_file)
            self.log_console_output = log_config.get('console_output', self.log_console_output)
        
        # Model settings
        if 'model' in yaml_config:
            model_config = yaml_config['model']
            self.model_save_path = model_config.get('save_path', self.model_save_path)
            self.model_checkpoint_frequency = model_config.get('checkpoint_frequency', self.model_checkpoint_frequency)
            self.model_keep_best_only = model_config.get('keep_best_only', self.model_keep_best_only)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_yaml(self, config_path: str):
        """Save current configuration to YAML file."""
        config_dict = self._unflatten_config(self.to_dict())
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    @staticmethod
    def _unflatten_config(flattened: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flattened configuration back to nested structure."""
        unflattened = {}
        
        for key, value in flattened.items():
            parts = key.split('_')
            current = unflattened
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return unflattened
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if self.log_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self.data_cache_dir,
            self.viz_save_path,
            self.model_save_path,
            Path(self.log_file).parent if self.log_file else None
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created necessary directories")
    
    def validate(self) -> bool:
        """Validate configuration values."""
        is_valid = True
        
        # Validate macro ratios
        macro_sum = sum(self.env_macro_ratios.values())
        if abs(macro_sum - 1.0) > 0.01:
            logger.error(f"Macro ratios must sum to 1.0, got {macro_sum}")
            is_valid = False
        
        # Validate positive values
        positive_fields = [
            'agent_learning_rate', 'agent_gamma', 'agent_tau', 'training_success_threshold'
        ]
        for field in positive_fields:
            value = getattr(self, field)
            if value <= 0:
                logger.error(f"{field} must be positive, got {value}")
                is_valid = False
        
        # Validate ranges
        if not (0 < self.agent_gamma <= 1):
            logger.error(f"agent_gamma must be in (0, 1], got {self.agent_gamma}")
            is_valid = False
        
        if not (0 < self.training_success_threshold <= 1):
            logger.error(f"training_success_threshold must be in (0, 1], got {self.training_success_threshold}")
            is_valid = False
        
        return is_valid 