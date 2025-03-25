"""
Plant data generator for space agriculture simulation.
This module generates synthetic data for different plant species in space environments.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logger = logging.getLogger('SpaceAgriRL.DataGenerator')

def generate_plant_data(num_samples=1000, seed=None):
    """
    Generate synthetic plant growth data for space agriculture
    
    Args:
        num_samples: Number of data points to generate per species
        seed: Random seed for reproducibility
        
    Returns:
        pandas DataFrame with plant growth data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define plant species and their optimal growth parameters
    species_list = ['Dwarf Wheat', 'Cherry Tomato', 'Lettuce', 'Space Potato']
    
    optimal_params = {
        'Dwarf Wheat': {
            'temperature': (20, 25),
            'light_intensity': (800, 1200),
            'water_content': (60, 80),
            'radiation_tolerance': 10,
            'growth_cycle_days': 60,
            'nitrogen_need': 'high',
            'phosphorus_need': 'medium',
            'potassium_need': 'medium'
        },
        'Cherry Tomato': {
            'temperature': (22, 28),
            'light_intensity': (1000, 1600),
            'water_content': (65, 85),
            'radiation_tolerance': 8,
            'growth_cycle_days': 90,
            'nitrogen_need': 'medium',
            'phosphorus_need': 'high',
            'potassium_need': 'high'
        },
        'Lettuce': {
            'temperature': (18, 24),
            'light_intensity': (500, 900),
            'water_content': (70, 90),
            'radiation_tolerance': 5,
            'growth_cycle_days': 45,
            'nitrogen_need': 'high',
            'phosphorus_need': 'medium',
            'potassium_need': 'low'
        },
        'Space Potato': {
            'temperature': (15, 22),
            'light_intensity': (700, 1100),
            'water_content': (60, 80),
            'radiation_tolerance': 15,
            'growth_cycle_days': 120,
            'nitrogen_need': 'medium',
            'phosphorus_need': 'high',
            'potassium_need': 'high'
        }
    }
    
    # Function to generate growth stage based on day in growth cycle
    def get_growth_stage(day, total_days):
        if day < total_days * 0.2:
            return 'germination'
        elif day < total_days * 0.5:
            return 'vegetative'
        elif day < total_days * 0.8:
            return 'flowering'
        else:
            return 'fruiting'
    
    # Function to calculate health score based on environmental conditions
    def calculate_health(temp, light, water, radiation, temp_opt, light_opt, water_opt, rad_tol):
        # Optimal temperature score
        if temp_opt[0] <= temp <= temp_opt[1]:
            temp_score = 1.0
        else:
            dist = min(abs(temp - temp_opt[0]), abs(temp - temp_opt[1]))
            temp_score = max(0, 1.0 - dist / 10)
            
        # Optimal light score
        if light_opt[0] <= light <= light_opt[1]:
            light_score = 1.0
        else:
            dist = min(abs(light - light_opt[0]), abs(light - light_opt[1]))
            light_score = max(0, 1.0 - dist / 500)
            
        # Optimal water score
        if water_opt[0] <= water <= water_opt[1]:
            water_score = 1.0
        else:
            dist = min(abs(water - water_opt[0]), abs(water - water_opt[1]))
            water_score = max(0, 1.0 - dist / 20)
            
        # Radiation score (lower is better)
        rad_score = max(0, 1.0 - radiation / (rad_tol * 2))
        
        # Weighted average
        health = 0.3 * temp_score + 0.3 * light_score + 0.25 * water_score + 0.15 * rad_score
        
        # Add some noise
        health = min(1.0, max(0, health + np.random.normal(0, 0.05)))
        
        return health
    
    # Generate data for each species
    data_list = []
    
    for species in species_list:
        species_params = optimal_params[species]
        growth_cycle = species_params['growth_cycle_days']
        
        for _ in range(num_samples):
            # Random day in growth cycle
            day = np.random.randint(1, growth_cycle + 1)
            growth_stage = get_growth_stage(day, growth_cycle)
            
            # Environmental parameters with variation
            temp_range = species_params['temperature']
            light_range = species_params['light_intensity']
            water_range = species_params['water_content']
            
            # Generate values with more variance outside optimal range
            if np.random.random() < 0.7:  # 70% chance of being near optimal
                temperature = np.random.uniform(temp_range[0], temp_range[1])
                light_intensity = np.random.uniform(light_range[0], light_range[1])
                water_content = np.random.uniform(water_range[0], water_range[1])
            else:  # 30% chance of being outside optimal
                temperature = np.random.uniform(5, 40)
                light_intensity = np.random.uniform(100, 2000)
                water_content = np.random.uniform(20, 100)
            
            # Other environmental factors
            radiation_level = np.random.exponential(scale=species_params['radiation_tolerance'])
            co2_level = np.random.normal(800, 200)  # ppm
            o2_level = np.random.normal(21, 2)  # percent
            humidity = np.random.normal(70, 10)  # percent
            
            # Nutrient levels
            nutrient_mapping = {'low': 60, 'medium': 75, 'high': 90}
            base_n = nutrient_mapping[species_params['nitrogen_need']]
            base_p = nutrient_mapping[species_params['phosphorus_need']]
            base_k = nutrient_mapping[species_params['potassium_need']]
            
            nitrogen_level = np.random.normal(base_n, 15)
            phosphorus_level = np.random.normal(base_p, 15)
            potassium_level = np.random.normal(base_k, 15)
            
            # Calculate health based on conditions
            health_score = calculate_health(
                temperature, light_intensity, water_content, radiation_level,
                temp_range, light_range, water_range, species_params['radiation_tolerance']
            )
            
            # Growth metrics
            progress_ratio = day / growth_cycle
            
            # Height calculation (sigmoid curve)
            max_height = {'Dwarf Wheat': 40, 'Cherry Tomato': 60, 'Lettuce': 25, 'Space Potato': 30}[species]
            base_height = max_height / (1 + np.exp(-10 * (progress_ratio - 0.5)))
            
            # Adjust height based on health
            height = base_height * (0.5 + 0.5 * health_score) * (1 + np.random.normal(0, 0.1))
            
            # Fruit/yield calculation (only in fruiting stage)
            fruit_count = 0
            if growth_stage == 'fruiting':
                max_fruits = {'Dwarf Wheat': 50, 'Cherry Tomato': 20, 'Lettuce': 1, 'Space Potato': 8}[species]
                late_progress = (day - 0.8 * growth_cycle) / (0.2 * growth_cycle)  # Progress within fruiting stage
                fruit_count = int(max_fruits * late_progress * health_score * (1 + np.random.normal(0, 0.2)))
                fruit_count = max(0, fruit_count)
            
            # Create data point
            data_point = {
                'species': species,
                'day': day,
                'growth_stage': growth_stage,
                'temperature': temperature,
                'light_intensity': light_intensity,
                'water_content': water_content,
                'radiation_level': radiation_level,
                'co2_level': co2_level,
                'o2_level': o2_level,
                'humidity': humidity,
                'nitrogen_level': nitrogen_level,
                'phosphorus_level': phosphorus_level,
                'potassium_level': potassium_level,
                'height': height,
                'health_score': health_score,
                'fruit_count': fruit_count
            }
            data_list.append(data_point)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Ensure all numerical columns have appropriate data types
    numeric_cols = ['day', 'temperature', 'light_intensity', 'water_content', 
                    'radiation_level', 'co2_level', 'o2_level', 'humidity',
                    'nitrogen_level', 'phosphorus_level', 'potassium_level',
                    'height', 'health_score', 'fruit_count']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    logger.info(f"Generated plant data with {len(df)} samples")
    return df

def save_plant_data(df, filename='plant_data.csv'):
    """Save plant data to CSV file"""
    df.to_csv(filename, index=False)
    logger.info(f"Plant data saved to {filename}")
    return filename

def load_plant_data(filename='plant_data.csv'):
    """Load plant data from CSV file"""
    if not os.path.exists(filename):
        logger.warning(f"Plant data file {filename} not found. Generating new data.")
        df = generate_plant_data()
        save_plant_data(df, filename)
        return df
    
    try:
        df = pd.read_csv(filename)
        logger.info(f"Loaded plant data from {filename} with {len(df)} samples")
        return df
    except Exception as e:
        logger.error(f"Error loading plant data: {e}")
        logger.info("Generating new plant data instead")
        df = generate_plant_data()
        save_plant_data(df, filename)
        return df

if __name__ == "__main__":
    # When run directly, generate and save plant data
    logging.basicConfig(level=logging.INFO)
    
    df = generate_plant_data(num_samples=2000, seed=42)
    save_plant_data(df, 'plant_data.csv')
    
    # Output some statistics
    print(f"Generated {len(df)} plant data samples")
    print(f"Species distribution:\n{df['species'].value_counts()}")
    print(f"Growth stage distribution:\n{df['growth_stage'].value_counts()}")
