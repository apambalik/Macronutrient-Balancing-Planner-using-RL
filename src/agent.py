"""
PPO (Proximal Policy Optimization) Agent for Meal Planning.

This module implements the RL agent using PyTorch and PPO algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random

logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """Actor network for continuous action space using Beta distribution"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Beta distribution parameters (alpha and beta)
        self.alpha_head = nn.Linear(hidden_size, action_dim)
        self.beta_head = nn.Linear(hidden_size, action_dim)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Beta distribution parameters"""
        features = self.network(obs)
        
        # Ensure alpha and beta are positive (>1 for unimodal distributions)
        alpha = F.softplus(self.alpha_head(features)) + 1.0
        beta = F.softplus(self.beta_head(features)) + 1.0
        
        return alpha, beta
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability"""
        alpha, beta = self.forward(obs)
        
        if deterministic:
            # Use mode of Beta distribution: (alpha-1)/(alpha+beta-2)
            action = (alpha - 1) / (alpha + beta - 2)
            action = torch.clamp(action, 0.01, 0.99)  # Avoid boundary issues
        else:
            # Sample from Beta distribution
            dist = Beta(alpha, beta)
            action = dist.sample()
        
        # Calculate log probability
        dist = Beta(alpha, beta)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic network for value function estimation"""
    
    def __init__(self, obs_dim: int, hidden_size: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value"""
        return self.network(obs)


class ReplayBuffer:
    """Experience replay buffer for PPO"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs: np.ndarray, action: np.ndarray, reward: float, 
             next_obs: np.ndarray, done: bool, log_prob: float, value: float):
        """Add experience to buffer"""
        experience = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class PPOAgent:
    """PPO Agent for continuous action spaces"""
    
    def __init__(self, obs_dim: int, action_dim: int, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.lr = config.agent_learning_rate
        self.gamma = config.agent_gamma
        self.tau = config.agent_tau  # GAE parameter
        self.eps_clip = config.agent_eps_clip
        self.value_coef = config.agent_value_coef
        self.entropy_coef = config.agent_entropy_coef
        self.max_grad_norm = config.agent_max_grad_norm
        
        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, config.agent_hidden_size)
        self.critic = CriticNetwork(obs_dim, config.agent_hidden_size)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Experience buffer
        self.buffer = ReplayBuffer()
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'losses': [],
            'success_episodes': []
        }
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        logger.info(f"PPO Agent initialized on device: {self.device}")
        logger.info(f"Network architecture: {obs_dim} -> {config.agent_hidden_size} -> {action_dim}")
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """Select action given observation"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get action from actor
            action, log_prob = self.actor.get_action(obs_tensor, deterministic=not training)
            
            # Get value from critic
            value = self.critic(obs_tensor)
        
        action_np = action.cpu().numpy()[0]
        log_prob_np = log_prob.cpu().item()
        value_np = value.cpu().item()
        
        return action_np, log_prob_np, value_np
    
    def store_experience(self, obs: np.ndarray, action: np.ndarray, reward: float,
                        next_obs: np.ndarray, done: bool, log_prob: float, value: float):
        """Store experience in replay buffer"""
        self.buffer.push(obs, action, reward, next_obs, done, log_prob, value)
    
    def update(self, epochs: int = 10, batch_size: int = 64):
        """Update actor and critic networks using PPO"""
        if len(self.buffer) < batch_size:
            return
        
        # Convert buffer to tensors
        experiences = list(self.buffer.buffer)
        
        observations = torch.FloatTensor([exp['obs'] for exp in experiences]).to(self.device)
        actions = torch.FloatTensor([exp['action'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        next_observations = torch.FloatTensor([exp['next_obs'] for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp['done'] for exp in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in experiences]).to(self.device)
        old_values = torch.FloatTensor([exp['value'] for exp in experiences]).to(self.device)
        
        # Calculate advantages using GAE
        with torch.no_grad():
            next_values = self.critic(next_observations).squeeze()
            advantages = self._calculate_gae(rewards, old_values, next_values, dones)
            returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        
        for epoch in range(epochs):
            # Create mini-batches
            indices = torch.randperm(len(experiences))
            
            for start_idx in range(0, len(experiences), batch_size):
                end_idx = min(start_idx + batch_size, len(experiences))
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Calculate new action probabilities
                alpha, beta = self.actor(batch_obs)
                dist = Beta(alpha, beta)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Calculate ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate actor loss
                clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                
                # Add entropy bonus
                entropy = dist.entropy().mean()
                actor_loss = actor_loss - self.entropy_coef * entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Calculate critic loss
                current_values = self.critic(batch_obs).squeeze()
                critic_loss = F.mse_loss(current_values, batch_returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
        
        # Store losses
        avg_actor_loss = total_actor_loss / (epochs * len(range(0, len(experiences), batch_size)))
        avg_critic_loss = total_critic_loss / (epochs * len(range(0, len(experiences), batch_size)))
        combined_loss = avg_actor_loss + avg_critic_loss
        
        self.training_history['losses'].append(combined_loss)
        
        # Clear buffer after update
        self.buffer.clear()
    
    def _calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                      next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] * (1 - dones[t])
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.tau * gae * (1 - dones[t])
            advantages[t] = gae
        
        return advantages
    
    def train(self, environment, num_episodes: int = 1000, update_frequency: int = 10,
              save_frequency: int = 100, **kwargs) -> Dict[str, List]:
        """Train the agent"""
        logger.info(f"Starting PPO training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            observation, _ = environment.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Select action
                action, log_prob, value = self.select_action(observation, training=True)
                
                # Take step in environment
                next_observation, reward, done, truncated, info = environment.step(action)
                
                # Store experience
                self.store_experience(
                    observation, action, reward, next_observation, 
                    done or truncated, log_prob, value
                )
                
                observation = next_observation
                episode_reward += reward
                episode_length += 1
            
            # Store episode results
            self.training_history['episode_rewards'].append(episode_reward)
            success = episode_reward > self.config.training_success_threshold
            self.training_history['success_episodes'].append(success)
            
            # Update networks
            if episode % update_frequency == 0 and len(self.buffer) > 0:
                self.update(
                    epochs=self.config.training_epochs_per_update,
                    batch_size=self.config.training_batch_size
                )
            
            # Logging
            if episode % self.config.training_log_frequency == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                success_rate = np.mean(self.training_history['success_episodes'][-100:])
                logger.info(f"Episode {episode}: Reward={episode_reward:.3f}, "
                           f"Avg Reward={avg_reward:.3f}, Success Rate={success_rate:.3f}")
            
            # Save model
            if episode % save_frequency == 0 and episode > 0:
                self.save(f"models/ppo_agent_episode_{episode}.pth")
        
        logger.info("Training completed")
        return self.training_history
    
    def evaluate(self, environment, num_eval_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent"""
        logger.info(f"Evaluating agent for {num_eval_episodes} episodes")
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_eval_episodes):
            observation, _ = environment.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Select action (deterministic)
                action, _, _ = self.select_action(observation, training=False)
                
                # Take step
                observation, reward, done, truncated, _ = environment.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths)
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save(self, filepath: str):
        """Save agent model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'lr': self.lr,
                'gamma': self.gamma,
                'tau': self.tau,
                'eps_clip': self.eps_clip
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_name(self) -> str:
        """Get agent name"""
        return "PPO" 