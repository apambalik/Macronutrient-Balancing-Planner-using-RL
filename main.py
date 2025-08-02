#!/usr/bin/env python3
"""
Main training script for the Macronutrient Balancing Planner using RL.

This script orchestrates the entire training and evaluation pipeline:
1. Load configuration
2. Initialize data components
3. Setup environment and agents
4. Run training
5. Evaluate performance
6. Generate visualizations and reports
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.data_loader import OpenFoodFactsAPI, RealFoodDatabase
from src.user_profile import UserProfile, Gender, ActivityLevel, Goal
from src.evaluation import EvaluationMetrics, BaselineComparison

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Macronutrient Balancing Planner with RL")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["train", "evaluate", "demo"],
        default="train",
        help="Run mode: train, evaluate, or demo"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        help="Number of episodes to run (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison"
    )
    
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force refresh of food database"
    )
    
    return parser.parse_args()


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_sample_user_profile() -> UserProfile:
    """Create a sample user profile for demonstration"""
    return UserProfile(
        age=30,
        gender=Gender.MALE,
        weight=75.0,  # kg
        height=180.0,  # cm
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        goals={
            'weight_goal': Goal.MAINTAIN_WEIGHT,
            'target_calories': 2200,
            'protein_ratio': 0.25,
            'carb_ratio': 0.45,
            'fat_ratio': 0.30
        },
        preferences={
            'liked_categories': ['chicken', 'rice', 'vegetables'],
            'disliked_foods': ['liver'],
            'dietary_restrictions': [],
            'allergens': []
        }
    )


def initialize_components(config: Config, refresh_data: bool = False) -> Dict[str, Any]:
    """Initialize all system components"""
    logger.info("Initializing system components...")
    
    # Initialize API and database
    logger.info("Setting up food database...")
    api = OpenFoodFactsAPI(config)
    database = RealFoodDatabase(api, config)
    
    # Populate database if needed
    database.populate_database(force_refresh=refresh_data)
    
    if len(database.foods) == 0:
        logger.error("No food data available. Please check your internet connection.")
        sys.exit(1)
    
    logger.info(f"Database ready with {len(database.foods)} food items")
    
    # Create user profile
    user_profile = create_sample_user_profile()
    logger.info(f"User profile created: {user_profile.get_summary()}")
    
    # Initialize evaluation and visualization
    evaluator = EvaluationMetrics(config)
    
    # Initialize baseline comparison
    baseline_comparison = BaselineComparison(config, database, user_profile)
    
    return {
        'api': api,
        'database': database,
        'user_profile': user_profile,
        'evaluator': evaluator,
        'baseline_comparison': baseline_comparison
    }


def create_environment(config: Config, database, user_profile):
    """Create the RL environment"""
    # Import here to avoid circular imports
    try:
        from src.environment import MealPlanningEnvironment
        
        env = MealPlanningEnvironment(
            food_database=database,
            user_profile=user_profile,
            config=config
        )
        
        logger.info(f"Environment created with observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        return env
        
    except ImportError as e:
        logger.error(f"Failed to import environment: {e}")
        logger.info("Creating mock environment for testing...")
        return create_mock_environment(config, database, user_profile)


def create_mock_environment(config, database, user_profile):
    """Create a mock environment for testing when the real environment is not available"""
    import gymnasium as gym
    from gymnasium import spaces
    
    class MockEnvironment(gym.Env):
        def __init__(self, food_database, user_profile, config):
            super().__init__()
            self.food_database = food_database
            self.user_profile = user_profile
            self.config = config
            self.max_steps_per_episode = config.env_max_steps_per_episode
            
            # Define observation and action spaces
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0, high=1, shape=(2,), dtype=np.float32
            )
            
            self.current_step = 0
            self.total_nutrition = {'calories': 0, 'protein': 0, 'carbohydrates': 0, 'fat': 0}
        
        def reset(self, seed=None):
            self.current_step = 0
            self.total_nutrition = {'calories': 0, 'protein': 0, 'carbohydrates': 0, 'fat': 0}
            return np.random.randn(20).astype(np.float32), {}
        
        def step(self, action):
            self.current_step += 1
            
            # Mock food selection
            food_ids = list(self.food_database.foods.keys())
            if food_ids:
                food_idx = int(action[0] * len(food_ids))
                food_idx = max(0, min(food_idx, len(food_ids) - 1))
                selected_food = self.food_database.foods[food_ids[food_idx]]
                
                portion_size = max(50, min(200, action[1] * 150 + 50))  # 50-200g
                
                # Update nutrition
                multiplier = portion_size / 100
                self.total_nutrition['calories'] += selected_food.nutrition.calories * multiplier
                self.total_nutrition['protein'] += selected_food.nutrition.protein * multiplier
                self.total_nutrition['carbohydrates'] += selected_food.nutrition.carbohydrates * multiplier
                self.total_nutrition['fat'] += selected_food.nutrition.fat * multiplier
                
                info = {'selected_food': selected_food, 'portion_size': portion_size}
            else:
                info = {}
            
            # Simple reward calculation
            target_calories = self.user_profile.goals['target_calories']
            calorie_diff = abs(self.total_nutrition['calories'] - target_calories)
            reward = max(-1, 1 - calorie_diff / target_calories)
            
            done = self.current_step >= self.max_steps_per_episode
            observation = np.random.randn(20).astype(np.float32)
            
            return observation, reward, done, False, info
    
    return MockEnvironment(database, user_profile, config)


def create_agent(config: Config, env):
    """Create the RL agent"""
    try:
        from src.agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=config
        )
        
        logger.info("PPO agent created successfully")
        return agent
        
    except ImportError as e:
        logger.error(f"Failed to import agent: {e}")
        logger.info("Creating mock agent for testing...")
        return create_mock_agent(config, env)


def create_mock_agent(config, env):
    """Create a mock agent for testing"""
    class MockAgent:
        def __init__(self, obs_dim, action_dim, config):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.config = config
            self.training_history = {
                'episode_rewards': [],
                'losses': [],
                'success_episodes': []
            }
        
        def select_action(self, observation, training=True):
            action = np.random.uniform(0, 1, self.action_dim)
            log_prob = -1.0  # Mock log probability
            value = 0.0      # Mock value
            return action, log_prob, value
        
        def train(self, num_episodes=1000, update_frequency=10, **kwargs):
            logger.info(f"Mock training for {num_episodes} episodes...")
            
            for episode in range(num_episodes):
                # Simulate episode reward
                reward = np.random.normal(-0.5, 1.0)  # Mock reward
                self.training_history['episode_rewards'].append(reward)
                
                # Simulate loss every update_frequency episodes
                if episode % update_frequency == 0:
                    loss = np.random.exponential(0.5)  # Mock loss
                    self.training_history['losses'].append(loss)
                
                # Simulate success
                success = reward > 0
                self.training_history['success_episodes'].append(success)
                
                if episode % 100 == 0:
                    logger.info(f"Episode {episode}: Reward={reward:.3f}")
            
            logger.info("Mock training completed")
            return self.training_history
        
        def save(self, filepath):
            logger.info(f"Mock save to {filepath}")
        
        def load(self, filepath):
            logger.info(f"Mock load from {filepath}")
    
    return MockAgent(env.observation_space.shape[0], env.action_space.shape[0], config)


def run_training(config: Config, components: Dict[str, Any], episodes: int = None) -> Dict[str, Any]:
    """Run the training pipeline"""
    logger.info("Starting training pipeline...")
    
    # Create environment and agent
    env = create_environment(config, components['database'], components['user_profile'])
    agent = create_agent(config, env)
    
    # Set training parameters
    num_episodes = episodes or config.training_num_episodes
    
    # Run training
    logger.info(f"Training agent for {num_episodes} episodes...")
    training_history = agent.train(
        environment=env,
        num_episodes=num_episodes,
        update_frequency=config.training_update_frequency,
        save_frequency=config.training_save_frequency
    )
    
    # Save trained model
    model_path = Path(config.model_save_path) / "trained_agent.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_path))
    
    logger.info("Training completed successfully")
    
    return {
        'agent': agent,
        'env': env,
        'training_history': training_history
    }


def run_evaluation(config: Config, components: Dict[str, Any], agent, env) -> Dict[str, Any]:
    """Run comprehensive evaluation"""
    logger.info("Starting evaluation pipeline...")
    
    evaluator = components['evaluator']
    baseline_comparison = components['baseline_comparison']
    
    # Evaluate trained agent
    logger.info("Evaluating trained agent...")
    agent_result = evaluator.evaluate_agent(agent, env, config.eval_num_episodes)
    
    # Run baseline comparison
    logger.info("Running baseline comparisons...")
    comparison_results = baseline_comparison.compare_with_rl_agent(
        agent, env, config.eval_num_episodes
    )
    
    logger.info("Evaluation completed successfully")
    
    return {
        'agent_result': agent_result,
        'comparison_results': comparison_results
    }


def generate_reports(config: Config, components: Dict[str, Any], 
                    training_results: Dict[str, Any], evaluation_results: Dict[str, Any]):
    """Generate simple text reports"""
    logger.info("Generating reports...")
    
    # Extract data
    training_history = training_results['training_history']
    comparison_results = evaluation_results['comparison_results']
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if training_history.get('episode_rewards'):
        final_reward = training_history['episode_rewards'][-1]
        avg_reward = np.mean(training_history['episode_rewards'][-100:])  # Last 100 episodes
        print(f"Final Episode Reward: {final_reward:.3f}")
        print(f"Average Reward (Last 100): {avg_reward:.3f}")
    
    print(f"Total Episodes: {len(training_history.get('episode_rewards', []))}")
    
    print("\n" + "="*60)
    print("AGENT COMPARISON")
    print("="*60)
    
    rankings = comparison_results['comparison']['agent_rankings']
    for rank, (agent_name, score) in enumerate(rankings, 1):
        print(f"{rank}. {agent_name}: {score:.3f}")
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Print evaluation metrics
    if 'agent_result' in evaluation_results:
        agent_result = evaluation_results['agent_result']
        success_rate = agent_result.get_success_rate()
        avg_reward = agent_result.get_average_reward()
        
        print(f"Success Rate: {success_rate:.3f}")
        print(f"Average Reward: {avg_reward:.3f}")
        
        # Print individual metrics if available
        metrics = agent_result.metrics
        for metric_name, value in metrics.items():
            if metric_name not in ['average_reward', 'success_rate']:
                print(f"{metric_name.replace('_', ' ').title()}: {value:.3f}")
    
    logger.info("Reports generated successfully")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.episodes:
        config.training_num_episodes = args.episodes
        config.eval_num_episodes = min(args.episodes // 10, 50)
    
    if args.verbose:
        config.log_level = "DEBUG"
    
    # Setup logging and directories
    config.setup_logging()
    config.create_directories()
    
    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    logger.info("Starting Macronutrient Balancing Planner with RL")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Initialize components
        components = initialize_components(config, args.refresh_data)
        
        if args.mode == "train":
            # Run training
            training_results = run_training(config, components, args.episodes)
            
            # Run evaluation
            if not args.no_baseline:
                evaluation_results = run_evaluation(
                    config, components, 
                    training_results['agent'], 
                    training_results['env']
                )
                
                # Generate reports
                generate_reports(config, components, training_results, evaluation_results)
            
            logger.info("Training pipeline completed successfully")
        
        elif args.mode == "evaluate":
            # Load existing model and evaluate
            logger.info("Evaluation mode not fully implemented yet")
            logger.info("Please use train mode with evaluation included")
        
        elif args.mode == "demo":
            # Run demonstration
            logger.info("Demo mode not implemented yet")
            logger.info("Please use train mode for full functionality")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 