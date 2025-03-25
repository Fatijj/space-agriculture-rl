"""
Reinforcement Learning Agents for Space Agriculture Optimization.
This module implements simplified versions of the RL agents.
"""

import os
import numpy as np
import random
import logging
from collections import deque

# Configure logging
logger = logging.getLogger('SpaceAgriRL.Agent')

class DQNAgent:
    """Simplified DQN Agent for continuous action space"""
    
    def __init__(self, state_size, action_size, batch_size=64, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Exploration parameters
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # discount factor
        
        # Track metrics
        self.episode_rewards = []
        self.avg_rewards = []
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        """Choose action based on current state"""
        if explore and np.random.rand() <= self.epsilon:
            # Random exploration
            return np.random.uniform(-1, 1, self.action_size)
        
        # Simplified "trained" behavior - just produce reasonable actions
        # based on the state without neural networks
        
        # Extract state variables (assuming a specific state format)
        # temperature, light, water, radiation, CO2, O2, humidity, N, P, K, height, health
        
        # Define some basic logic that attempts to keep things in reasonable ranges
        # For demonstration only - not a real trained agent!
        
        # Default: no change
        action = np.zeros(self.action_size)
        
        try:
            # Very simplified, imperfect heuristic for demonstration only
            # Adjust temperature if too hot or too cold
            if state[0] < 18:  # too cold
                action[0] = 0.5  # increase temperature
            elif state[0] > 30:  # too hot
                action[0] = -0.5  # decrease temperature
                
            # Adjust light
            if state[1] < 700:  # too dim
                action[1] = 0.5  # increase light
            elif state[1] > 1500:  # too bright
                action[1] = -0.5  # decrease light
                
            # Adjust water
            if state[2] < 60:  # too dry
                action[2] = 0.5  # increase water
            elif state[2] > 90:  # too wet
                action[2] = -0.5  # decrease water
                
            # Adjust radiation shield (inversely related to radiation level)
            if state[3] > 20:  # too much radiation
                action[3] = 0.5  # increase shield (reduce radiation)
                
            # Adjust nutrients based on NPK levels (average of indexes 7,8,9)
            avg_nutrient = (state[7] + state[8] + state[9]) / 3
            if avg_nutrient < 60:
                action[4] = 0.5  # increase nutrients
                
            # Add some noise
            action += np.random.normal(0, 0.1, self.action_size)
            
            # Clip to valid range
            action = np.clip(action, -1, 1)
        except:
            # Fallback if there's any error in the logic above
            action = np.random.uniform(-0.1, 0.1, self.action_size) 
        
        return action
    
    def replay(self, batch_size=None):
        """Simulate training the agent (simplified version)"""
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Return mock losses for monitoring
        return {
            'actor_loss': 0.1 * random.random(),
            'critic_loss': 0.1 * random.random(),
            'epsilon': self.epsilon
        }
    
    def save_model(self, actor_path='actor_model.h5', critic_path='critic_model.h5'):
        """Mock saving the models to disk"""
        logger.info(f"Mock models would be saved to {actor_path} and {critic_path}")
        return True
    
    def load_model(self, actor_path='actor_model.h5', critic_path='critic_model.h5'):
        """Mock loading the models from disk"""
        logger.info(f"Mock models would be loaded from {actor_path} and {critic_path}")
        return False
    
    def update_episode_rewards(self, episode_reward):
        """Track episode rewards"""
        self.episode_rewards.append(episode_reward)
        # Calculate rolling average over last 100 episodes
        window_size = min(100, len(self.episode_rewards))
        avg_reward = sum(self.episode_rewards[-window_size:]) / window_size
        self.avg_rewards.append(avg_reward)

class PPOAgent:
    """Simplified PPO Agent for continuous control tasks"""
    
    def __init__(self, state_size, action_size, batch_size=64, learning_rate=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Logging
        self.episode_rewards = []
        self.avg_rewards = []
    
    def get_action(self, state):
        """Sample action (simplified implementation without neural networks)"""
        # Default: no change
        action = np.zeros(self.action_size)
        
        try:
            # Very simplified, imperfect heuristic for demonstration only
            # Adjust temperature if too hot or too cold
            if state[0] < 18:  # too cold
                action[0] = 0.5  # increase temperature
            elif state[0] > 30:  # too hot
                action[0] = -0.5  # decrease temperature
                
            # Adjust light
            if state[1] < 700:  # too dim
                action[1] = 0.5  # increase light
            elif state[1] > 1500:  # too bright
                action[1] = -0.5  # decrease light
                
            # Adjust water
            if state[2] < 60:  # too dry
                action[2] = 0.5  # increase water
            elif state[2] > 90:  # too wet
                action[2] = -0.5  # decrease water
                
            # Adjust radiation shield (inversely related to radiation level)
            if state[3] > 20:  # too much radiation
                action[3] = 0.5  # increase shield (reduce radiation)
                
            # Adjust nutrients based on NPK levels (average of indexes 7,8,9)
            avg_nutrient = (state[7] + state[8] + state[9]) / 3
            if avg_nutrient < 60:
                action[4] = 0.5  # increase nutrients
                
            # Add some noise
            action += np.random.normal(0, 0.1, self.action_size)
            
            # Clip to valid range
            action = np.clip(action, -1, 1)
        except:
            # Fallback if there's any error in the logic above
            action = np.random.uniform(-0.1, 0.1, self.action_size) 
        
        # For PPO interface compatibility, return action and a mock log probability
        return action, 0.0
    
    def get_value(self, state):
        """Predict value (simplified implementation)"""
        # Simple heuristic - health score is a reasonable proxy for value
        try:
            return state[11]  # Health score is typically the last element
        except:
            return 0.5  # Default value
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store trajectory step"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def train(self):
        """Simulate training (simplified implementation)"""
        # Clear trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Return mock losses for monitoring
        return {
            'actor_loss': 0.1 * random.random(),
            'critic_loss': 0.1 * random.random()
        }
    
    def save_model(self, actor_path='ppo_actor_model.h5', critic_path='ppo_critic_model.h5'):
        """Mock saving models to disk"""
        logger.info(f"Mock PPO models would be saved to {actor_path} and {critic_path}")
        return True
    
    def load_model(self, actor_path='ppo_actor_model.h5', critic_path='ppo_critic_model.h5'):
        """Mock loading models from disk"""
        logger.info(f"Mock PPO models would be loaded from {actor_path} and {critic_path}")
        return False
    
    def update_episode_rewards(self, episode_reward):
        """Track episode rewards"""
        self.episode_rewards.append(episode_reward)
        # Calculate rolling average over last 100 episodes
        window_size = min(100, len(self.episode_rewards))
        avg_reward = sum(self.episode_rewards[-window_size:]) / window_size
        self.avg_rewards.append(avg_reward)
