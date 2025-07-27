"""
Main entry point for the Macronutrient Balancing Planner using RL.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .agents.agent_factory import AgentFactory
from .environment.meal_planning_env import MealPlanningEnvironment
from .data.user_profile import UserProfile
from .data.nutritional_database import NutritionalDatabase
from .feedback.feedback_processor import FeedbackProcessor
from .utils.config import Config


@dataclass
class MealPlan:
    """Represents a generated meal plan with nutritional information."""
    meals: List[Dict[str, Any]]
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fats: float
    satisfaction_score: float
    variety_score: float


class MealPlanningAgent:
    """
    Main agent class that orchestrates the RL-based meal planning system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the meal planning agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.nutritional_db = NutritionalDatabase()
        self.feedback_processor = FeedbackProcessor()
        self.agent_factory = AgentFactory()
        
        # Initialize RL agent
        self.agent = self.agent_factory.create_agent(
            algorithm=self.config.algorithm,
            env_config=self.config.agent
        )
        
        # Initialize environment
        self.environment = MealPlanningEnvironment(
            nutritional_db=self.nutritional_db,
            config=self.config.environment
        )
        
        self.logger.info("MealPlanningAgent initialized successfully")
    
    def generate_meal_plan(self, user_profile: UserProfile) -> MealPlan:
        """
        Generate a personalized meal plan for the given user profile.
        
        Args:
            user_profile: User's dietary preferences and requirements
            
        Returns:
            MealPlan: Generated meal plan with nutritional information
        """
        self.logger.info(f"Generating meal plan for user: {user_profile.user_id}")
        
        # Set up environment with user profile
        self.environment.set_user_profile(user_profile)
        
        # Reset environment to initialize state
        self.environment.reset()
        
        # Generate meal plan using RL agent
        action = self.agent.predict(self.environment.get_state())
        self.environment.step(action)
        
        # Get the meal plan data
        meal_plan_data = self.environment.get_meal_plan()
        
        # Process feedback if available
        if hasattr(user_profile, 'previous_feedback') and user_profile.previous_feedback:
            # Only process feedback if it's a dictionary (not a list)
            if isinstance(user_profile.previous_feedback, dict):
                self.feedback_processor.process_feedback(
                    user_profile.user_id,
                    user_profile.previous_feedback
                )
        
        return MealPlan(**meal_plan_data)
    
    def update_from_feedback(self, user_id: str, feedback: Dict[str, Any]):
        """
        Update the agent based on user feedback.
        
        Args:
            user_id: User identifier
            feedback: User feedback on meal plan
        """
        self.logger.info(f"Processing feedback for user: {user_id}")
        
        # Process feedback
        processed_feedback = self.feedback_processor.process_feedback(user_id, feedback)
        
        # Update agent with feedback
        self.agent.update_from_feedback(processed_feedback)
        
        self.logger.info("Agent updated from feedback")
    
    def train(self, episodes: int = 1000):
        """
        Train the RL agent.
        
        Args:
            episodes: Number of training episodes
        """
        self.logger.info(f"Starting training for {episodes} episodes")
        
        self.agent.train(
            env=self.environment,
            total_timesteps=episodes * self.config.environment.max_steps
        )
        
        self.logger.info("Training completed")
    
    def save_model(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        self.agent.save(path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        self.agent.load(path)
        self.logger.info(f"Model loaded from {path}")


def main():
    """Main function for running the meal planning system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Meal Planning Agent")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = MealPlanningAgent(args.config)
    
    if args.train:
        agent.train(args.episodes)
        agent.save_model("models/meal_planning_agent")
    
    print("Meal Planning Agent initialized successfully!")


if __name__ == "__main__":
    main() 