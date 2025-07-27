"""
Factory for creating different RL agents.
"""

from typing import Dict, Any, Optional
import logging

from .ppo_agent import PPOAgent
from .sac_agent import SACAgent
from .ddpg_agent import DDPGAgent
from .a2c_agent import A2CAgent


class AgentFactory:
    """Factory for creating RL agents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_agent(self, algorithm: str, env_config: Any) -> Any:
        """
        Create an RL agent based on the specified algorithm.
        
        Args:
            algorithm: Algorithm name (PPO, SAC, DDPG, A2C)
            env_config: Environment configuration
            
        Returns:
            RL agent instance
        """
        algorithm = algorithm.upper()
        
        if algorithm == "PPO":
            return PPOAgent(env_config)
        elif algorithm == "SAC":
            return SACAgent(env_config)
        elif algorithm == "DDPG":
            return DDPGAgent(env_config)
        elif algorithm == "A2C":
            return A2CAgent(env_config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def get_supported_algorithms(self) -> list:
        """Get list of supported algorithms."""
        return ["PPO", "SAC", "DDPG", "A2C"]
    
    def get_algorithm_description(self, algorithm: str) -> str:
        """Get description of the algorithm."""
        descriptions = {
            "PPO": "Proximal Policy Optimization - On-policy algorithm with stable policy updates",
            "SAC": "Soft Actor-Critic - Off-policy algorithm with maximum entropy for exploration",
            "DDPG": "Deep Deterministic Policy Gradient - Off-policy algorithm for continuous actions",
            "A2C": "Advantage Actor-Critic - On-policy algorithm with advantage estimation"
        }
        return descriptions.get(algorithm.upper(), "Unknown algorithm") 