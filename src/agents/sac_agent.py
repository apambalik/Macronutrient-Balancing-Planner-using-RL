"""
SAC (Soft Actor-Critic) agent for meal planning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class SACAgent:
    """
    SAC agent for meal planning using stable-baselines3.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.env = None
        
        # SAC-specific parameters
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.tau = config.tau
        self.target_update_freq = config.target_update_freq
        self.ent_coef = config.ent_coef
        
        self.logger.info("SAC Agent initialized")
    
    def setup_model(self, env):
        """Setup the SAC model with the environment."""
        self.env = DummyVecEnv([lambda: env])
        
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            target_update_freq=self.target_update_freq,
            ent_coef=self.ent_coef,
            verbose=1
        )
        
        self.logger.info("SAC model setup completed")
    
    def train(self, env, total_timesteps: int = 10000):
        """Train the SAC agent."""
        if self.model is None:
            self.setup_model(env)
        
        self.logger.info(f"Starting SAC training for {total_timesteps} timesteps")
        
        # Create callback for logging
        callback = SACCallback()
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        self.logger.info("SAC training completed")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict action given observation."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(path)
            self.logger.info(f"SAC model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model."""
        self.model = SAC.load(path)
        self.logger.info(f"SAC model loaded from {path}")
    
    def update_from_feedback(self, feedback: Dict[str, Any]):
        """Update the agent based on user feedback."""
        # SAC supports online updates through experience replay
        if feedback.get('satisfaction_score', 0) < 0.5:
            # If user is not satisfied, increase entropy for exploration
            self.ent_coef *= 1.2
            self.logger.info("Increased entropy coefficient for better exploration")
        
        if feedback.get('nutrition_score', 0) < 0.7:
            # If nutrition is poor, adjust learning rate
            self.learning_rate *= 0.95
            self.logger.info("Decreased learning rate for more stable learning")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.model is None:
            return {"status": "not_trained"}
        
        return {
            "algorithm": "SAC",
            "policy_type": "MlpPolicy",
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_update_freq": self.target_update_freq,
            "ent_coef": self.ent_coef,
            "status": "trained"
        }


class SACCallback(BaseCallback):
    """Callback for SAC training to log custom metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.entropy_values = []
    
    def _on_step(self) -> bool:
        """Called after each step during training."""
        # Log custom metrics
        if len(self.training_env.buf_rews) > 0:
            episode_reward = np.mean(self.training_env.buf_rews)
            self.episode_rewards.append(episode_reward)
            
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.logger.record("custom/avg_episode_reward", avg_reward)
        
        # Log entropy for SAC
        if hasattr(self.model, 'ent_coef'):
            entropy = self.model.ent_coef
            self.entropy_values.append(entropy)
            self.logger.record("custom/entropy_coefficient", entropy)
        
        return True
    
    def _on_training_end(self):
        """Called at the end of training."""
        if len(self.episode_rewards) > 0:
            final_avg_reward = np.mean(self.episode_rewards[-100:])
            self.logger.record("custom/final_avg_reward", final_avg_reward)
        
        if len(self.entropy_values) > 0:
            final_entropy = self.entropy_values[-1]
            self.logger.record("custom/final_entropy", final_entropy) 