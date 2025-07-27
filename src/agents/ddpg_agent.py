"""
DDPG (Deep Deterministic Policy Gradient) agent for meal planning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import logging
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class DDPGAgent:
    """
    DDPG agent for meal planning using stable-baselines3.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.env = None
        
        # DDPG-specific parameters
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.tau = config.tau
        self.target_update_freq = config.target_update_freq
        self.policy_freq = config.policy_freq
        self.noise_clip = config.noise_clip
        self.policy_noise = config.policy_noise
        
        self.logger.info("DDPG Agent initialized")
    
    def setup_model(self, env):
        """Setup the DDPG model with the environment."""
        self.env = DummyVecEnv([lambda: env])
        
        self.model = DDPG(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            target_update_freq=self.target_update_freq,
            policy_freq=self.policy_freq,
            noise_clip=self.noise_clip,
            policy_noise=self.policy_noise,
            verbose=1
        )
        
        self.logger.info("DDPG model setup completed")
    
    def train(self, env, total_timesteps: int = 10000):
        """Train the DDPG agent."""
        if self.model is None:
            self.setup_model(env)
        
        self.logger.info(f"Starting DDPG training for {total_timesteps} timesteps")
        
        # Create callback for logging
        callback = DDPGCallback()
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        self.logger.info("DDPG training completed")
    
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
            self.logger.info(f"DDPG model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model."""
        self.model = DDPG.load(path)
        self.logger.info(f"DDPG model loaded from {path}")
    
    def update_from_feedback(self, feedback: Dict[str, Any]):
        """Update the agent based on user feedback."""
        # DDPG supports online updates through experience replay
        if feedback.get('satisfaction_score', 0) < 0.5:
            # If user is not satisfied, increase noise for exploration
            self.policy_noise *= 1.1
            self.logger.info("Increased policy noise for better exploration")
        
        if feedback.get('nutrition_score', 0) < 0.7:
            # If nutrition is poor, adjust learning rate
            self.learning_rate *= 0.95
            self.logger.info("Decreased learning rate for more stable learning")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.model is None:
            return {"status": "not_trained"}
        
        return {
            "algorithm": "DDPG",
            "policy_type": "MlpPolicy",
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_update_freq": self.target_update_freq,
            "policy_freq": self.policy_freq,
            "noise_clip": self.noise_clip,
            "policy_noise": self.policy_noise,
            "status": "trained"
        }


class DDPGCallback(BaseCallback):
    """Callback for DDPG training to log custom metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_noise_values = []
    
    def _on_step(self) -> bool:
        """Called after each step during training."""
        # Log custom metrics
        if len(self.training_env.buf_rews) > 0:
            episode_reward = np.mean(self.training_env.buf_rews)
            self.episode_rewards.append(episode_reward)
            
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.logger.record("custom/avg_episode_reward", avg_reward)
        
        # Log action noise for DDPG
        if hasattr(self.model, 'policy_noise'):
            noise = self.model.policy_noise
            self.action_noise_values.append(noise)
            self.logger.record("custom/action_noise", noise)
        
        return True
    
    def _on_training_end(self):
        """Called at the end of training."""
        if len(self.episode_rewards) > 0:
            final_avg_reward = np.mean(self.episode_rewards[-100:])
            self.logger.record("custom/final_avg_reward", final_avg_reward)
        
        if len(self.action_noise_values) > 0:
            final_noise = self.action_noise_values[-1]
            self.logger.record("custom/final_action_noise", final_noise) 