"""
Space Agriculture Environment for Reinforcement Learning
This module implements a custom environment for space agriculture.
"""

import numpy as np
import pandas as pd
# Temporarily commented out due to dependency issues
# import gymnasium as gym
# from gymnasium import spaces
# import matplotlib.pyplot as plt
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='space_agri_rl.log')
logger = logging.getLogger('SpaceAgriRL')

class SpaceAgricultureEnv:
    """Custom Environment for Space Agriculture (simplified version)"""
    
    def __init__(self, plant_data, species='Dwarf Wheat'):
        # Filter data for specific species
        self.plant_data = plant_data[plant_data['species'] == species].copy() if isinstance(plant_data, pd.DataFrame) else pd.DataFrame()
        self.species = species
        
        # Define action space dimensions (adjust temperature, light, water, nutrients)
        # Each has a range from -1.0 (decrease) to 1.0 (increase)
        self.action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])  # temp, light, water, radiation_shield, nutrients
        self.action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Define observation dimensions
        # [temp, light, water, radiation, CO2, O2, humidity, N, P, K, height, health]
        self.observation_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.observation_high = np.array([50, 2000, 100, 100, 2000, 30, 100, 100, 100, 100, 100, 1.0])
        
        # Create action_space and observation_space objects with shape properties
        class SimpleSpace:
            def __init__(self, shape):
                self.shape = shape
                
        self.action_space = SimpleSpace((5,))
        self.observation_space = SimpleSpace((12,))
        
        # Add disease detection variables
        self.disease_reward_modifier = 1.0  # Default: no disease detected (multiplier for reward)
        
        # Define default optimal ranges
        default_ranges = {
            'temperature': (20, 25),
            'light_intensity': (800, 1200),
            'water_content': (60, 80),
            'radiation_level': (0, 10),
            'nutrient_mix': (70, 90)  # overall nutrient level
        }
        
        # Define optimal ranges for all species
        all_species_ranges = {
            'Dwarf Wheat': {
                'temperature': (20, 25),
                'light_intensity': (800, 1200),
                'water_content': (60, 80),
                'radiation_level': (0, 10),
                'nutrient_mix': (70, 90)
            },
            'Cherry Tomato': {
                'temperature': (22, 28),
                'light_intensity': (1000, 1600),
                'water_content': (65, 85),
                'radiation_level': (0, 8),
                'nutrient_mix': (75, 95)
            },
            'Lettuce': {
                'temperature': (18, 24),
                'light_intensity': (500, 900),
                'water_content': (70, 90),
                'radiation_level': (0, 5),
                'nutrient_mix': (65, 85)
            },
            'Space Potato': {
                'temperature': (15, 22),
                'light_intensity': (700, 1100),
                'water_content': (60, 80),
                'radiation_level': (0, 15),
                'nutrient_mix': (60, 80)
            },
            'Microgreens': {
                'temperature': (19, 23),
                'light_intensity': (600, 1000),
                'water_content': (70, 85),
                'radiation_level': (0, 7),
                'nutrient_mix': (65, 85)
            },
            'Space Basil': {
                'temperature': (21, 27),
                'light_intensity': (800, 1300),
                'water_content': (65, 80),
                'radiation_level': (0, 8),
                'nutrient_mix': (70, 90)
            },
            'Radish': {
                'temperature': (16, 20),
                'light_intensity': (650, 1100),
                'water_content': (60, 80),
                'radiation_level': (0, 9),
                'nutrient_mix': (60, 85)
            },
            'Spinach': {
                'temperature': (16, 22),
                'light_intensity': (600, 1000),
                'water_content': (65, 85),
                'radiation_level': (0, 6),
                'nutrient_mix': (65, 85)
            }
        }
        
        # Create optimal_ranges and ensure species is in the dictionary
        self.optimal_ranges = all_species_ranges.copy()
        if species not in self.optimal_ranges:
            self.optimal_ranges[species] = default_ranges.copy()
            logger.warning(f"Species '{species}' not found in predefined ranges, using default values")
        
        # State variables
        self.reset()
        
    def _get_observation(self):
        """Return the current observation state"""
        return np.array([
            self.state['temperature'],
            self.state['light_intensity'],
            self.state['water_content'],
            self.state['radiation_level'],
            self.state['co2_level'],
            self.state['o2_level'],
            self.state['humidity'],
            self.state['nitrogen_level'],
            self.state['phosphorus_level'],
            self.state['potassium_level'],
            self.state['height'],
            self.state['health_score']
        ], dtype=np.float32)
    
    def _calculate_reward(self):
        """Calculate reward based on how close parameters are to optimal and health score improvement"""
        reward = 0
        
        # Get optimal ranges for the current species, falling back to species directly if necessary
        species_ranges = self.optimal_ranges.get(self.species, self.optimal_ranges)
        
        # Reward for being in optimal ranges
        temp_range = species_ranges['temperature']
        if temp_range[0] <= self.state['temperature'] <= temp_range[1]:
            reward += 1
        else:
            dist = min(abs(self.state['temperature'] - temp_range[0]), 
                       abs(self.state['temperature'] - temp_range[1]))
            reward -= dist / 10  # Penalize based on distance from optimal range
            
        light_range = species_ranges['light_intensity']
        if light_range[0] <= self.state['light_intensity'] <= light_range[1]:
            reward += 1
        else:
            dist = min(abs(self.state['light_intensity'] - light_range[0]), 
                       abs(self.state['light_intensity'] - light_range[1]))
            reward -= dist / 200
            
        water_range = species_ranges['water_content']
        if water_range[0] <= self.state['water_content'] <= water_range[1]:
            reward += 1
        else:
            dist = min(abs(self.state['water_content'] - water_range[0]), 
                       abs(self.state['water_content'] - water_range[1]))
            reward -= dist / 10
            
        radiation_range = species_ranges['radiation_level']
        if radiation_range[0] <= self.state['radiation_level'] <= radiation_range[1]:
            reward += 1
        else:
            dist = min(abs(self.state['radiation_level'] - radiation_range[0]), 
                       abs(self.state['radiation_level'] - radiation_range[1]))
            reward -= dist / 5
            
        # Reward for health improvement
        health_diff = self.state['health_score'] - self.previous_health
        reward += health_diff * 10  # Significant reward for health improvement
        
        # Additional reward for height growth if in vegetative or flowering stage
        if self.state['growth_stage'] in ['vegetative', 'flowering']:
            height_diff = self.state['height'] - self.previous_height
            if height_diff > 0:
                reward += height_diff / 2
                
        # Penalize excessive adjustments (to promote stability)
        if np.abs(self.last_action).sum() > 2.0:
            reward -= 0.5
            
        # Big reward for successful fruit production in fruiting stage
        if self.state['growth_stage'] == 'fruiting' and self.state['fruit_count'] > self.previous_fruit_count:
            fruit_diff = self.state['fruit_count'] - self.previous_fruit_count
            reward += fruit_diff * 2
        
        # Apply disease detection modifier to reward
        # This penalizes the agent more for unhealthy plants or rewards proper disease management
        reward = reward * self.disease_reward_modifier
        
        # Prevent extreme negative rewards that could destabilize learning
        reward = max(reward, -10.0)  # Cap at -10.0 to avoid excessively negative rewards
            
        return reward
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.last_action = action
        self.steps_taken += 1
        
        # Store previous values to calculate changes
        self.previous_health = self.state['health_score']
        self.previous_height = self.state['height']
        self.previous_fruit_count = self.state['fruit_count']
        
        # Apply action adjustments (scaled)
        temp_adj = action[0] * 2.0  # +/- 2 degrees C
        light_adj = action[1] * 200.0  # +/- 200 μmol/m²/s
        water_adj = action[2] * 10.0  # +/- 10% water content
        radiation_shield_adj = action[3] * 5.0  # +/- 5 units of shielding
        nutrient_adj = action[4] * 5.0  # +/- 5% nutrient levels
        
        # Update state based on adjustments
        self.state['temperature'] = np.clip(self.state['temperature'] + temp_adj, 10, 40)
        self.state['light_intensity'] = np.clip(self.state['light_intensity'] + light_adj, 100, 2000)
        self.state['water_content'] = np.clip(self.state['water_content'] + water_adj, 10, 100)
        
        # Radiation shield reduces radiation
        radiation_reduction = radiation_shield_adj * 2 if radiation_shield_adj > 0 else 0
        self.state['radiation_level'] = np.clip(self.state['radiation_level'] - radiation_reduction, 0, 100)
        
        # Update nutrient levels
        self.state['nitrogen_level'] = np.clip(self.state['nitrogen_level'] + nutrient_adj, 20, 100)
        self.state['phosphorus_level'] = np.clip(self.state['phosphorus_level'] + nutrient_adj, 20, 100)
        self.state['potassium_level'] = np.clip(self.state['potassium_level'] + nutrient_adj, 20, 100)
        
        # Calculate average nutrient level
        avg_nutrient = (self.state['nitrogen_level'] + 
                        self.state['phosphorus_level'] + 
                        self.state['potassium_level']) / 3
        
        # Simulate random environmental fluctuations
        self.state['co2_level'] += np.random.normal(0, 20)
        self.state['co2_level'] = np.clip(self.state['co2_level'], 200, 2000)
        
        self.state['o2_level'] += np.random.normal(0, 0.5)
        self.state['o2_level'] = np.clip(self.state['o2_level'], 15, 25)
        
        self.state['humidity'] += np.random.normal(0, 2)
        self.state['humidity'] = np.clip(self.state['humidity'], 30, 90)
        
        # Update plant health based on environmental conditions
        # Get optimal ranges for the current species, falling back to species directly if necessary
        species_ranges = self.optimal_ranges.get(self.species, self.optimal_ranges)
        
        temp_opt = self._calculate_optimality(
            self.state['temperature'], 
            species_ranges['temperature'][0],
            species_ranges['temperature'][1]
        )
        
        light_opt = self._calculate_optimality(
            self.state['light_intensity'], 
            species_ranges['light_intensity'][0],
            species_ranges['light_intensity'][1]
        )
        
        water_opt = self._calculate_optimality(
            self.state['water_content'], 
            species_ranges['water_content'][0],
            species_ranges['water_content'][1]
        )
        
        radiation_opt = self._calculate_optimality(
            self.state['radiation_level'], 
            species_ranges['radiation_level'][0],
            species_ranges['radiation_level'][1],
            inverse=True  # Lower radiation is better
        )
        
        nutrient_opt = self._calculate_optimality(
            avg_nutrient, 
            species_ranges['nutrient_mix'][0],
            species_ranges['nutrient_mix'][1]
        )
        
        # Calculate new health score (weighted average of optimality scores)
        new_health = (
            0.25 * temp_opt + 
            0.25 * light_opt + 
            0.20 * water_opt + 
            0.15 * radiation_opt + 
            0.15 * nutrient_opt
        )
        
        # Add some randomness to health (e.g., pests, diseases)
        random_factor = np.random.normal(0, 0.05)
        new_health = np.clip(new_health + random_factor, 0.0, 1.0)
        
        # Smooth health changes (health changes gradually)
        self.state['health_score'] = 0.7 * self.state['health_score'] + 0.3 * new_health
        
        # Update growth metrics based on health
        if self.state['health_score'] > 0.7:
            # Good health, normal growth
            self.state['height'] += np.random.uniform(0.1, 0.5) * self.state['health_score']
            
            # Update growth stage based on height and time
            if self.steps_taken > 10 and self.state['height'] > 5:
                self.state['growth_stage'] = 'vegetative'
            if self.steps_taken > 20 and self.state['height'] > 10:
                self.state['growth_stage'] = 'flowering'
            if self.steps_taken > 30 and self.state['health_score'] > 0.6:
                self.state['growth_stage'] = 'fruiting'
                
            # Update fruit count in fruiting stage
            if self.state['growth_stage'] == 'fruiting':
                # Random chance to add fruits based on health
                if np.random.random() < self.state['health_score'] * 0.3:
                    self.state['fruit_count'] += 1
        else:
            # Poor health, minimal growth
            self.state['height'] += np.random.uniform(0, 0.1)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = (self.steps_taken >= self.max_steps or 
                self.state['health_score'] < 0.2 or  # Plant died
                (self.state['growth_stage'] == 'fruiting' and self.steps_taken > 40))  # Harvest time
        
        # Return step information
        return self._get_observation(), reward, done, False, {"state": self.state}
    
    def update_disease_modifier(self, diagnosis):
        """
        Update disease reward modifier based on plant disease detection results
        
        Args:
            diagnosis: Dictionary with disease detection results containing:
                - health_status: String indicating overall health status
                - confidence: Float from 0.0 to 1.0 indicating diagnosis confidence
                - disease_name: String with identified disease (if any)
                - disease_severity: Float from 0.0 to 1.0 indicating disease severity
                
        Returns:
            Updated disease_reward_modifier value
        """
        # Default modifier (no effect)
        default_modifier = 1.0
        
        # Extract diagnosis data
        health_status = diagnosis.get('health_status', 'Unknown')
        confidence = diagnosis.get('confidence', 0.5)
        disease_severity = diagnosis.get('disease_severity', 0.0)
        
        # Calculate modifier based on health status and severity
        if health_status == "Healthy":
            # Healthy plants get a slight reward bonus
            self.disease_reward_modifier = 1.1
        elif health_status == "Moderate Risk":
            # Moderate risk penalizes rewards proportional to severity
            penalty = 0.3 * disease_severity * confidence
            self.disease_reward_modifier = 1.0 - penalty
        elif health_status == "Severe Risk":
            # Severe risk strongly penalizes rewards
            penalty = 0.7 * disease_severity * confidence
            self.disease_reward_modifier = max(0.3, 1.0 - penalty)  # Cap at 0.3 to avoid excessive penalty
        else:
            # Unknown status, reset to default
            self.disease_reward_modifier = default_modifier
            
        # Log the update
        logger.info(f"Updated disease_reward_modifier to {self.disease_reward_modifier:.2f} based on health status: {health_status}")
        
        return self.disease_reward_modifier
    
    def _calculate_optimality(self, value, min_optimal, max_optimal, inverse=False):
        """Calculate how optimal a value is compared to its ideal range
        Returns a value between 0 (far from optimal) and 1.0 (perfectly optimal)
        
        Args:
            value: The current value
            min_optimal: Lower bound of optimal range
            max_optimal: Upper bound of optimal range
            inverse: If True, lower values are better (e.g., for radiation)
        """
        if inverse:
            # For metrics where lower is better (like radiation)
            if value <= min_optimal:
                return 1.0
            elif value >= max_optimal * 2:  # Far above max is very bad
                return 0.0
            else:
                # Linear decrease from optimal min to 2x max
                range_size = (max_optimal * 2) - min_optimal
                distance = value - min_optimal
                return 1.0 - (distance / range_size)
        else:
            # Normal case where being in range is optimal
            if min_optimal <= value <= max_optimal:
                return 1.0  # In optimal range
            
            # Calculate how far we are from the optimal range
            if value < min_optimal:
                distance = min_optimal - value
                reference = min_optimal / 2  # We consider half of min as very bad
            else:  # value > max_optimal
                distance = value - max_optimal
                reference = max_optimal * 0.5  # We consider 50% above max as very bad
            
            # Calculate score based on distance (1.0 = optimal, 0.0 = very bad)
            score = 1.0 - (distance / reference)
            return max(0.0, min(score, 1.0))  # Clamp between 0 and 1

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize state with default values
        self.state = {
            'temperature': 22.0,  # degrees C
            'light_intensity': 1000.0,  # μmol/m²/s (PAR)
            'water_content': 70.0,  # % of optimal
            'radiation_level': 20.0,  # arbitrary units
            'co2_level': 800.0,  # ppm
            'o2_level': 21.0,  # %
            'humidity': 60.0,  # %
            'nitrogen_level': 80.0,  # % of optimal
            'phosphorus_level': 80.0,  # % of optimal
            'potassium_level': 80.0,  # % of optimal
            'height': 1.0,  # cm
            'growth_stage': 'germination',  # germination, vegetative, flowering, fruiting
            'health_score': 0.9,  # 0.0 to 1.0
            'fruit_count': 0  # number of fruits/grains produced
        }
        
        # Reset tracking variables
        self.steps_taken = 0
        self.max_steps = 50  # maximum steps per episode
        self.previous_health = self.state['health_score']
        self.previous_height = self.state['height']
        self.previous_fruit_count = 0
        self.last_action = np.zeros(5)
        self.disease_reward_modifier = 1.0  # Reset disease detection modifier
        
        # Add some initial randomness
        self.state['temperature'] += np.random.uniform(-2, 2)
        self.state['light_intensity'] += np.random.uniform(-100, 100)
        self.state['water_content'] += np.random.uniform(-5, 5)
        self.state['radiation_level'] += np.random.uniform(0, 10)
        
        # Add some random variation to nutrients
        nutrient_var = np.random.uniform(-10, 10)
        self.state['nitrogen_level'] += nutrient_var
        self.state['phosphorus_level'] += nutrient_var
        self.state['potassium_level'] += nutrient_var
        
        # Clip values to valid ranges
        self.state['temperature'] = np.clip(self.state['temperature'], 10, 40)
        self.state['light_intensity'] = np.clip(self.state['light_intensity'], 100, 2000)
        self.state['water_content'] = np.clip(self.state['water_content'], 10, 100)
        self.state['radiation_level'] = np.clip(self.state['radiation_level'], 0, 100)
        self.state['nitrogen_level'] = np.clip(self.state['nitrogen_level'], 20, 100)
        self.state['phosphorus_level'] = np.clip(self.state['phosphorus_level'], 20, 100)
        self.state['potassium_level'] = np.clip(self.state['potassium_level'], 20, 100)
        
        return self._get_observation(), {"state": self.state}
    
    def render(self, mode="human"):
        """Render the current state of the environment"""
        if mode == "human":
            print(f"Step: {self.steps_taken}/{self.max_steps}")
            print(f"Species: {self.species}")
            print(f"Growth Stage: {self.state['growth_stage']}")
            print(f"Health: {self.state['health_score']:.2f}")
            print(f"Height: {self.state['height']:.1f} cm")
            if self.state['growth_stage'] == 'fruiting':
                print(f"Fruits: {self.state['fruit_count']}")
            print(f"Temperature: {self.state['temperature']:.1f}°C")
            print(f"Light: {self.state['light_intensity']:.0f} μmol/m²/s")
            print(f"Water: {self.state['water_content']:.1f}%")
            print(f"Radiation: {self.state['radiation_level']:.1f} units")
            print(f"CO2: {self.state['co2_level']:.0f} ppm")
            print(f"Nutrients (N-P-K): {self.state['nitrogen_level']:.0f}-"
                  f"{self.state['phosphorus_level']:.0f}-{self.state['potassium_level']:.0f}")
            print("-" * 40)
        
        return None

    def close(self):
        """Clean up resources"""
        pass
        
    def update_disease_modifier(self, diagnosis_results):
        """
        Update disease reward modifier based on plant disease detection results
        
        Args:
            diagnosis_results: Dictionary with disease detection results containing:
                - has_disease: Boolean indicating disease presence
                - severity: Float from 0.0 to 1.0 indicating disease severity
                - confidence: Float from 0.0 to 1.0 indicating diagnosis confidence
                
        Returns:
            Updated disease_reward_modifier value
        """
        # Default no change if diagnosis is not confident
        if 'confidence' in diagnosis_results and diagnosis_results['confidence'] < 0.5:
            return self.disease_reward_modifier
            
        # Set the modifier based on disease presence and severity
        if 'has_disease' in diagnosis_results and diagnosis_results['has_disease']:
            # Diseases reduce rewards based on severity
            severity = diagnosis_results.get('severity', 0.5)  # Default to medium severity
            # Calculate modifier: 1.0 (no disease) to 0.2 (severe disease)
            self.disease_reward_modifier = max(0.2, 1.0 - (severity * 0.8))
            logger.info(f"Disease detected with severity {severity:.2f}. Reward modifier set to {self.disease_reward_modifier:.2f}")
        else:
            # No disease, normal rewards
            self.disease_reward_modifier = 1.0
            
        return self.disease_reward_modifier
