"""
Evaluation and baseline comparison module for the Meal Planning RL system.

This module provides:
- Comprehensive evaluation metrics
- Baseline comparison methods (random, greedy, heuristic)
- Statistical analysis tools
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metrics: Dict[str, float]
    episode_rewards: List[float]
    episode_details: List[Dict[str, Any]]
    success_episodes: int
    total_episodes: int
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        return self.success_episodes / self.total_episodes if self.total_episodes > 0 else 0.0
    
    def get_average_reward(self) -> float:
        """Calculate average reward"""
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
    
    def get_std_reward(self) -> float:
        """Calculate standard deviation of rewards"""
        return np.std(self.episode_rewards) if self.episode_rewards else 0.0


class EvaluationMetrics:
    """Comprehensive evaluation metrics for meal planning"""
    
    def __init__(self, config):
        self.config = config
        self.success_threshold = config.training_success_threshold
    
    def evaluate_agent(self, agent, env, num_episodes: int = 20) -> EvaluationResult:
        """Evaluate an agent's performance"""
        episode_rewards = []
        episode_details = []
        success_episodes = 0
        
        for episode in range(num_episodes):
            observation, _ = env.reset()
            episode_reward = 0
            episode_data = {
                'meals': [],
                'total_nutrition': {'calories': 0, 'protein': 0, 'carbohydrates': 0, 'fat': 0},
                'macro_balance_accuracy': 0,
                'calorie_accuracy': 0,
                'variety_score': 0,
                'quality_score': 0
            }
            
            done = False
            step = 0
            
            while not done and step < env.max_steps_per_episode:
                if hasattr(agent, 'select_action'):
                    action, _, _ = agent.select_action(observation, training=False)
                else:
                    action = agent.get_action(observation, training=False)
                
                observation, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                # Track meal information
                if 'selected_food' in info:
                    food_item = info['selected_food']
                    portion_size = info.get('portion_size', 100)
                    
                    episode_data['meals'].append({
                        'food': food_item.name,
                        'portion': portion_size,
                        'nutrition': food_item.nutrition
                    })
                    
                    # Update total nutrition
                    multiplier = portion_size / 100
                    episode_data['total_nutrition']['calories'] += food_item.nutrition.calories * multiplier
                    episode_data['total_nutrition']['protein'] += food_item.nutrition.protein * multiplier
                    episode_data['total_nutrition']['carbohydrates'] += food_item.nutrition.carbohydrates * multiplier
                    episode_data['total_nutrition']['fat'] += food_item.nutrition.fat * multiplier
                
                step += 1
            
            # Calculate detailed metrics
            metrics = self._calculate_episode_metrics(episode_data, env.user_profile)
            episode_data.update(metrics)
            
            episode_rewards.append(episode_reward)
            episode_details.append(episode_data)
            
            # Check if episode was successful
            if episode_reward >= self.success_threshold:
                success_episodes += 1
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(episode_details)
        
        return EvaluationResult(
            metrics=overall_metrics,
            episode_rewards=episode_rewards,
            episode_details=episode_details,
            success_episodes=success_episodes,
            total_episodes=num_episodes
        )
    
    def _calculate_episode_metrics(self, episode_data: Dict, user_profile) -> Dict[str, float]:
        """Calculate detailed metrics for a single episode"""
        total_nutrition = episode_data['total_nutrition']
        target_calories = user_profile.goals['target_calories']
        target_protein, target_carbs, target_fat = user_profile.get_target_macro_grams()
        
        # Calorie accuracy
        calorie_diff = abs(total_nutrition['calories'] - target_calories)
        calorie_accuracy = max(0, 1 - (calorie_diff / target_calories))
        
        # Macro balance accuracy
        protein_diff = abs(total_nutrition['protein'] - target_protein)
        carbs_diff = abs(total_nutrition['carbohydrates'] - target_carbs)
        fat_diff = abs(total_nutrition['fat'] - target_fat)
        
        macro_error = (protein_diff / target_protein + 
                      carbs_diff / target_carbs + 
                      fat_diff / target_fat) / 3
        macro_balance_accuracy = max(0, 1 - macro_error)
        
        # Meal variety (unique foods / total meals)
        foods = [meal['food'] for meal in episode_data['meals']]
        variety_score = len(set(foods)) / len(foods) if foods else 0
        
        # Average nutritional quality
        if episode_data['meals']:
            quality_scores = []
            for meal in episode_data['meals']:
                # Simple quality score based on nutrition density
                nutrition = meal['nutrition']
                quality = (nutrition.protein + nutrition.fiber) / max(1, nutrition.calories / 100)
                quality_scores.append(quality)
            quality_score = np.mean(quality_scores)
        else:
            quality_score = 0
        
        return {
            'calorie_accuracy': calorie_accuracy,
            'macro_balance_accuracy': macro_balance_accuracy,
            'variety_score': variety_score,
            'quality_score': quality_score
        }
    
    def _calculate_overall_metrics(self, episode_details: List[Dict]) -> Dict[str, float]:
        """Calculate overall metrics across all episodes"""
        if not episode_details:
            return {}
        
        metrics = {}
        
        # Average each metric across episodes
        metric_keys = ['calorie_accuracy', 'macro_balance_accuracy', 'variety_score', 'quality_score']
        for key in metric_keys:
            values = [ep[key] for ep in episode_details if key in ep]
            metrics[f'avg_{key}'] = np.mean(values) if values else 0
            metrics[f'std_{key}'] = np.std(values) if values else 0
        
        # Meal-level statistics
        all_meals = []
        for ep in episode_details:
            all_meals.extend(ep.get('meals', []))
        
        if all_meals:
            # Average meals per episode
            metrics['avg_meals_per_episode'] = len(all_meals) / len(episode_details)
            
            # Most common foods
            food_counts = defaultdict(int)
            for meal in all_meals:
                food_counts[meal['food']] += 1
            
            # Diversity metrics
            total_unique_foods = len(food_counts)
            total_meals = len(all_meals)
            metrics['food_diversity'] = total_unique_foods / total_meals if total_meals > 0 else 0
            
        return metrics
    
    def compare_agents(self, agent_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Compare multiple agents' performance"""
        comparison = {
            'agent_rankings': [],
            'statistical_tests': {},
            'metric_comparisons': {}
        }
        
        # Rank agents by average reward
        agent_scores = {
            name: result.get_average_reward() 
            for name, result in agent_results.items()
        }
        ranked_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['agent_rankings'] = ranked_agents
        
        # Compare specific metrics
        for metric in ['calorie_accuracy', 'macro_balance_accuracy', 'variety_score']:
            metric_key = f'avg_{metric}'
            comparison['metric_comparisons'][metric] = {
                name: result.metrics.get(metric_key, 0)
                for name, result in agent_results.items()
            }
        
        return comparison


class BaselineAgent(ABC):
    """Abstract base class for baseline agents"""
    
    def __init__(self, config, food_database):
        self.config = config
        self.food_database = food_database
    
    @abstractmethod
    def get_action(self, observation: np.ndarray, training: bool = False) -> np.ndarray:
        """Get action from observation"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get agent name"""
        pass


class RandomAgent(BaselineAgent):
    """Random baseline agent"""
    
    def __init__(self, config, food_database):
        super().__init__(config, food_database)
        self.rng = random.Random(config.baseline_random_seed)
    
    def get_action(self, observation: np.ndarray, training: bool = False) -> np.ndarray:
        """Select random food and portion size"""
        # Random food selection (index 0-999 for 1000 foods)
        food_idx = self.rng.randint(0, min(999, len(self.food_database.foods) - 1))
        
        # Random portion size (50-200g)
        portion_size = self.rng.uniform(0.5, 2.0)  # Normalized (50-200g -> 0.5-2.0)
        
        return np.array([food_idx / 1000.0, portion_size])
    
    def get_name(self) -> str:
        return "Random"


class GreedyAgent(BaselineAgent):
    """Greedy baseline agent that prioritizes macro balance"""
    
    def __init__(self, config, food_database, user_profile):
        super().__init__(config, food_database)
        self.user_profile = user_profile
        self.macro_priority = config.baseline_greedy_macro_priority
        self.current_nutrition = {'protein': 0, 'carbohydrates': 0, 'fat': 0, 'calories': 0}
        self.reset()
    
    def reset(self):
        """Reset nutrition tracking for new episode"""
        self.current_nutrition = {'protein': 0, 'carbohydrates': 0, 'fat': 0, 'calories': 0}
    
    def get_action(self, observation: np.ndarray, training: bool = False) -> np.ndarray:
        """Select food that best balances current macro needs"""
        target_protein, target_carbs, target_fat = self.user_profile.get_target_macro_grams()
        target_calories = self.user_profile.goals['target_calories']
        
        # Calculate current deficits
        protein_deficit = max(0, target_protein - self.current_nutrition['protein'])
        carbs_deficit = max(0, target_carbs - self.current_nutrition['carbohydrates'])
        fat_deficit = max(0, target_fat - self.current_nutrition['fat'])
        calorie_deficit = max(0, target_calories - self.current_nutrition['calories'])
        
        best_food_idx = 0
        best_score = -float('inf')
        
        # Evaluate each food
        food_items = list(self.food_database.foods.values())
        for idx, food in enumerate(food_items[:1000]):  # Limit to first 1000 foods
            if not self.user_profile.is_food_compatible(food):
                continue
            
            # Calculate how well this food addresses current deficits
            score = 0
            
            # Protein contribution
            if protein_deficit > 0:
                score += min(food.nutrition.protein, protein_deficit) * self.macro_priority[0]
            
            # Carbs contribution
            if carbs_deficit > 0:
                score += min(food.nutrition.carbohydrates, carbs_deficit) * self.macro_priority[1]
            
            # Fat contribution
            if fat_deficit > 0:
                score += min(food.nutrition.fat, fat_deficit) * self.macro_priority[2]
            
            # Penalty for exceeding calorie needs
            if self.current_nutrition['calories'] + food.nutrition.calories > target_calories:
                score -= (self.current_nutrition['calories'] + food.nutrition.calories - target_calories) * 0.1
            
            if score > best_score:
                best_score = score
                best_food_idx = idx
        
        # Calculate appropriate portion size
        if best_food_idx < len(food_items):
            food = food_items[best_food_idx]
            remaining_calories = target_calories - self.current_nutrition['calories']
            
            if food.nutrition.calories > 0:
                optimal_portion = min(2.0, max(0.5, remaining_calories / food.nutrition.calories))
            else:
                optimal_portion = 1.0
            
            # Update current nutrition (for next action)
            multiplier = optimal_portion
            self.current_nutrition['protein'] += food.nutrition.protein * multiplier
            self.current_nutrition['carbohydrates'] += food.nutrition.carbohydrates * multiplier
            self.current_nutrition['fat'] += food.nutrition.fat * multiplier
            self.current_nutrition['calories'] += food.nutrition.calories * multiplier
        else:
            optimal_portion = 1.0
        
        return np.array([best_food_idx / 1000.0, optimal_portion])
    
    def get_name(self) -> str:
        return "Greedy"


class HeuristicAgent(BaselineAgent):
    """Heuristic baseline agent using nutritional rules"""
    
    def __init__(self, config, food_database, user_profile):
        super().__init__(config, food_database)
        self.user_profile = user_profile
        self.rules = config.baseline_heuristic_rules
        self.meal_count = 0
        self.current_nutrition = {'protein': 0, 'carbohydrates': 0, 'fat': 0, 'calories': 0}
    
    def reset(self):
        """Reset for new episode"""
        self.meal_count = 0
        self.current_nutrition = {'protein': 0, 'carbohydrates': 0, 'fat': 0, 'calories': 0}
    
    def get_action(self, observation: np.ndarray, training: bool = False) -> np.ndarray:
        """Select food based on heuristic rules"""
        food_items = list(self.food_database.foods.values())
        
        # Apply rules in order of priority
        candidates = food_items[:1000]  # Limit search space
        
        # Filter by compatibility
        candidates = [f for f in candidates if self.user_profile.is_food_compatible(f)]
        
        if not candidates:
            candidates = food_items[:100]  # Fallback
        
        # Apply heuristic rules
        for rule in self.rules:
            if rule == "prioritize_protein" and self.meal_count < 2:
                # Early meals should be protein-rich
                candidates = sorted(candidates, key=lambda f: f.nutrition.protein, reverse=True)
            
            elif rule == "balance_macros":
                # Select based on current macro balance
                target_protein, target_carbs, target_fat = self.user_profile.get_target_macro_grams()
                
                def balance_score(food):
                    protein_ratio = self.current_nutrition['protein'] / target_protein if target_protein > 0 else 0
                    carbs_ratio = self.current_nutrition['carbohydrates'] / target_carbs if target_carbs > 0 else 0
                    fat_ratio = self.current_nutrition['fat'] / target_fat if target_fat > 0 else 0
                    
                    # Choose macro that is most deficient
                    if protein_ratio <= carbs_ratio and protein_ratio <= fat_ratio:
                        return food.nutrition.protein
                    elif carbs_ratio <= fat_ratio:
                        return food.nutrition.carbohydrates
                    else:
                        return food.nutrition.fat
                
                candidates = sorted(candidates, key=balance_score, reverse=True)
            
            elif rule == "minimize_calories":
                # Prefer lower calorie options if approaching target
                target_calories = self.user_profile.goals['target_calories']
                remaining_calories = target_calories - self.current_nutrition['calories']
                
                if remaining_calories < target_calories * 0.3:  # Less than 30% remaining
                    candidates = sorted(candidates, key=lambda f: f.nutrition.calories)
        
        # Select best candidate
        if candidates:
            selected_food = candidates[0]
            food_idx = min(999, food_items.index(selected_food))
            
            # Calculate portion size
            target_calories = self.user_profile.goals['target_calories']
            remaining_calories = target_calories - self.current_nutrition['calories']
            
            if selected_food.nutrition.calories > 0:
                portion = min(2.0, max(0.5, remaining_calories / selected_food.nutrition.calories))
            else:
                portion = 1.0
            
            # Update tracking
            self.meal_count += 1
            multiplier = portion
            self.current_nutrition['protein'] += selected_food.nutrition.protein * multiplier
            self.current_nutrition['carbohydrates'] += selected_food.nutrition.carbohydrates * multiplier
            self.current_nutrition['fat'] += selected_food.nutrition.fat * multiplier
            self.current_nutrition['calories'] += selected_food.nutrition.calories * multiplier
            
            return np.array([food_idx / 1000.0, portion])
        
        # Fallback to random
        return np.array([0.5, 1.0])
    
    def get_name(self) -> str:
        return "Heuristic"


class BaselineComparison:
    """Baseline comparison system"""
    
    def __init__(self, config, food_database, user_profile):
        self.config = config
        self.food_database = food_database
        self.user_profile = user_profile
        self.evaluator = EvaluationMetrics(config)
        
        # Initialize baseline agents
        self.agents = {}
        
        if config.baseline_random_enabled:
            self.agents['Random'] = RandomAgent(config, food_database)
        
        if config.baseline_greedy_enabled:
            self.agents['Greedy'] = GreedyAgent(config, food_database, user_profile)
        
        if config.baseline_heuristic_enabled:
            self.agents['Heuristic'] = HeuristicAgent(config, food_database, user_profile)
    
    def evaluate_baselines(self, env, num_episodes: int = None) -> Dict[str, EvaluationResult]:
        """Evaluate all baseline agents"""
        if num_episodes is None:
            num_episodes = self.config.eval_num_episodes
        
        results = {}
        
        for name, agent in self.agents.items():
            logger.info(f"Evaluating {name} agent...")
            
            # Reset agent if it has state
            if hasattr(agent, 'reset'):
                agent.reset()
            
            result = self.evaluator.evaluate_agent(agent, env, num_episodes)
            results[name] = result
            
            logger.info(f"{name} - Avg Reward: {result.get_average_reward():.3f}, "
                       f"Success Rate: {result.get_success_rate():.3f}")
        
        return results
    
    def compare_with_rl_agent(self, rl_agent, env, num_episodes: int = None) -> Dict[str, Any]:
        """Compare RL agent with baseline agents"""
        if num_episodes is None:
            num_episodes = self.config.eval_num_episodes
        
        # Evaluate RL agent
        logger.info("Evaluating RL agent...")
        rl_result = self.evaluator.evaluate_agent(rl_agent, env, num_episodes)
        
        # Evaluate baseline agents
        baseline_results = self.evaluate_baselines(env, num_episodes)
        
        # Combine results
        all_results = {'RL_Agent': rl_result, **baseline_results}
        
        # Generate comparison report
        comparison = self.evaluator.compare_agents(all_results)
        
        # Add statistical significance tests if needed
        comparison['detailed_comparison'] = self._detailed_comparison(all_results)
        
        return {
            'results': all_results,
            'comparison': comparison
        }
    
    def _detailed_comparison(self, results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Generate detailed comparison statistics"""
        comparison = {}
        
        # Performance summary
        for name, result in results.items():
            comparison[name] = {
                'avg_reward': result.get_average_reward(),
                'std_reward': result.get_std_reward(),
                'success_rate': result.get_success_rate(),
                'avg_calorie_accuracy': result.metrics.get('avg_calorie_accuracy', 0),
                'avg_macro_balance': result.metrics.get('avg_macro_balance_accuracy', 0),
                'avg_variety': result.metrics.get('avg_variety_score', 0)
            }
        
        return comparison 