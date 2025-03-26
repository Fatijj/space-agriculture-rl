"""
Utility functions for the Space Agriculture RL system.
Simplified version without matplotlib dependencies.
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt - temporarily commented out
# from matplotlib.colors import LinearSegmentedColormap - temporarily commented out
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger('SpaceAgriRL.Utils')

def visualize_growth_progress(history, species, save_path=None):
    """
    Create textual representation of plant growth progress over time.
    Simplified version that doesn't use matplotlib.
    
    Args:
        history: List of state dictionaries from each step
        species: Plant species name
        save_path: Path to save the figure (if None, display only)
    
    Returns:
        A dictionary with data that can be displayed in streamlit
    """
    # Extract data from history
    if not history:
        return {"error": "No history data available"}
    
    steps = list(range(len(history)))
    heights = [state['height'] for state in history]
    health = [state['health_score'] for state in history]
    growth_stages = [state.get('growth_stage', 'unknown') for state in history]
    
    # Find growth stage changes
    stage_changes = []
    for i in range(1, len(growth_stages)):
        if growth_stages[i] != growth_stages[i-1]:
            stage_changes.append((i, growth_stages[i]))
    
    # Extract fruit count if in fruiting stage
    fruits = [state.get('fruit_count', 0) for state in history if 'fruit_count' in state]
    
    # Create a dictionary with the data
    result = {
        "species": species,
        "steps": steps,
        "heights": heights,
        "health": health,
        "growth_stages": growth_stages,
        "stage_changes": stage_changes,
        "fruits": fruits,
        "initial_height": heights[0] if heights else 0,
        "final_height": heights[-1] if heights else 0,
        "initial_health": health[0] if health else 0,
        "final_health": health[-1] if health else 0,
        "final_stage": growth_stages[-1] if growth_stages else "unknown",
        "total_steps": len(steps)
    }
    
    # Log that we would have saved a plot in the original version
    if save_path:
        logger.info(f"Growth progress data for {save_path} (not saved as image due to matplotlib being unavailable)")
    
    return result

def visualize_environment_parameters(history, save_path=None):
    """
    Simplified function to extract environmental parameters over time.
    
    Args:
        history: List of state dictionaries from each step
        save_path: Path to save the figure (if None, display only)
    
    Returns:
        Dictionary of environment parameter data
    """
    # Check for empty history
    if not history:
        return {"error": "No history data available"}
        
    # Extract data
    steps = list(range(len(history)))
    temp = [state['temperature'] for state in history]
    light = [state['light_intensity'] for state in history]
    water = [state['water_content'] for state in history]
    radiation = [state['radiation_level'] for state in history]
    co2 = [state['co2_level'] for state in history]
    
    # Nutrients
    n_level = [state['nitrogen_level'] for state in history]
    p_level = [state['phosphorus_level'] for state in history]
    k_level = [state['potassium_level'] for state in history]
    
    # Create a dictionary with the data
    result = {
        "steps": steps,
        "temperature": {
            "values": temp,
            "min": min(temp) if temp else 0,
            "max": max(temp) if temp else 0,
            "avg": sum(temp) / len(temp) if temp else 0
        },
        "light_intensity": {
            "values": light,
            "min": min(light) if light else 0,
            "max": max(light) if light else 0,
            "avg": sum(light) / len(light) if light else 0
        },
        "water_content": {
            "values": water,
            "min": min(water) if water else 0,
            "max": max(water) if water else 0,
            "avg": sum(water) / len(water) if water else 0
        },
        "radiation_level": {
            "values": radiation,
            "min": min(radiation) if radiation else 0,
            "max": max(radiation) if radiation else 0,
            "avg": sum(radiation) / len(radiation) if radiation else 0
        },
        "co2_level": {
            "values": co2,
            "min": min(co2) if co2 else 0,
            "max": max(co2) if co2 else 0,
            "avg": sum(co2) / len(co2) if co2 else 0
        },
        "nutrients": {
            "nitrogen": {
                "values": n_level,
                "min": min(n_level) if n_level else 0,
                "max": max(n_level) if n_level else 0,
                "avg": sum(n_level) / len(n_level) if n_level else 0
            },
            "phosphorus": {
                "values": p_level,
                "min": min(p_level) if p_level else 0,
                "max": max(p_level) if p_level else 0,
                "avg": sum(p_level) / len(p_level) if p_level else 0
            },
            "potassium": {
                "values": k_level,
                "min": min(k_level) if k_level else 0,
                "max": max(k_level) if k_level else 0,
                "avg": sum(k_level) / len(k_level) if k_level else 0
            }
        }
    }
    
    # Log that we would have saved a plot in the original version
    if save_path:
        logger.info(f"Environment parameters data for {save_path} (not saved as image due to matplotlib being unavailable)")
    
    return result

def visualize_agent_learning(agent, save_path=None):
    """
    Extract data about the agent's learning progress.
    
    Args:
        agent: RL agent with episode_rewards and avg_rewards attributes
        save_path: Path to save the figure (if None, display only)
    
    Returns:
        Dictionary with learning progress data
    """
    if not hasattr(agent, 'episode_rewards') or len(agent.episode_rewards) == 0:
        logger.warning("Agent has no recorded rewards to visualize")
        return {"error": "No reward data available"}
    
    # Extract data
    episodes = list(range(1, len(agent.episode_rewards) + 1))
    episode_rewards = agent.episode_rewards
    
    # Get average rewards if available
    avg_rewards = agent.avg_rewards if hasattr(agent, 'avg_rewards') and len(agent.avg_rewards) > 0 else []
    
    # Calculate statistics
    if episode_rewards:
        max_reward = max(episode_rewards)
        min_reward = min(episode_rewards)
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        last_reward = episode_rewards[-1]
    else:
        max_reward = min_reward = avg_reward = last_reward = 0
    
    # Create a dictionary with the data
    result = {
        "episodes": episodes,
        "episode_rewards": episode_rewards,
        "moving_avg_rewards": avg_rewards,
        "stats": {
            "max_reward": max_reward,
            "min_reward": min_reward,
            "avg_reward": avg_reward,
            "last_reward": last_reward,
            "total_episodes": len(episodes)
        }
    }
    
    # Log that we would have saved a plot in the original version
    if save_path:
        logger.info(f"Agent learning data for {save_path} (not saved as image due to matplotlib being unavailable)")
    
    return result

def save_experiment_results(experiment_name, agent, history, config, metrics):
    """
    Save experiment results to disk
    
    Args:
        experiment_name: Name/identifier for the experiment
        agent: RL agent with trained model
        history: List of state dictionaries from environment steps
        config: Dictionary containing experiment configuration
        metrics: Dictionary containing performance metrics
    """
    # Create directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/{experiment_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Save agent model
    agent.save_model(f"{result_dir}/actor_model.h5", f"{result_dir}/critic_model.h5")
    
    # Save history as JSON
    with open(f"{result_dir}/history.json", 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        clean_history = []
        for state in history:
            clean_state = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    clean_state[key] = value.tolist()
                elif isinstance(value, np.floating):
                    clean_state[key] = float(value)
                elif isinstance(value, np.integer):
                    clean_state[key] = int(value)
                else:
                    clean_state[key] = value
            clean_history.append(clean_state)
        
        json.dump(clean_history, f, indent=2)
    
    # Save configuration
    with open(f"{result_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    with open(f"{result_dir}/metrics.json", 'w') as f:
        # Clean metrics of numpy types
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                clean_metrics[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                clean_metrics[key] = float(value)
            else:
                clean_metrics[key] = value
        
        json.dump(clean_metrics, f, indent=2)
    
    # Generate and save visualizations
    visualize_growth_progress(history, config.get('species', 'Unknown'), 
                             save_path=f"{result_dir}/growth_progress.png")
    
    visualize_environment_parameters(history, 
                                   save_path=f"{result_dir}/environment_parameters.png")
    
    visualize_agent_learning(agent, 
                           save_path=f"{result_dir}/agent_learning.png")
    
    logger.info(f"Experiment results saved to {result_dir}")
    return result_dir

def calculate_performance_metrics(history, optimal_ranges):
    """
    Calculate performance metrics for the experiment
    
    Args:
        history: List of state dictionaries from environment steps
        optimal_ranges: Dictionary of optimal parameter ranges
        
    Returns:
        Dictionary of performance metrics
    """
    if not history:
        return {}
    
    # Final state metrics
    final_state = history[-1]
    
    # Calculate time spent in optimal ranges
    temp_range = optimal_ranges['temperature']
    light_range = optimal_ranges['light_intensity']
    water_range = optimal_ranges['water_content']
    radiation_range = optimal_ranges['radiation_level']
    
    temp_optimal_time = sum(1 for state in history 
                           if temp_range[0] <= state['temperature'] <= temp_range[1])
    
    light_optimal_time = sum(1 for state in history 
                            if light_range[0] <= state['light_intensity'] <= light_range[1])
    
    water_optimal_time = sum(1 for state in history 
                            if water_range[0] <= state['water_content'] <= water_range[1])
    
    radiation_optimal_time = sum(1 for state in history 
                               if radiation_range[0] <= state['radiation_level'] <= radiation_range[1])
    
    # Calculate overall optimality
    total_steps = len(history)
    temp_optimal_pct = (temp_optimal_time / total_steps) * 100
    light_optimal_pct = (light_optimal_time / total_steps) * 100
    water_optimal_pct = (water_optimal_time / total_steps) * 100
    radiation_optimal_pct = (radiation_optimal_time / total_steps) * 100
    
    # Calculate growth metrics
    initial_height = history[0]['height']
    final_height = final_state['height']
    growth_rate = (final_height - initial_height) / total_steps
    
    # Fruit production if applicable
    fruit_count = final_state.get('fruit_count', 0)
    
    # Calculate average health
    avg_health = sum(state['health_score'] for state in history) / total_steps
    
    # Calculate resource efficiency (change per reward unit)
    total_temp_change = sum(abs(history[i+1]['temperature'] - history[i]['temperature']) 
                           for i in range(total_steps-1))
    
    total_light_change = sum(abs(history[i+1]['light_intensity'] - history[i]['light_intensity']) 
                            for i in range(total_steps-1))
    
    total_water_change = sum(abs(history[i+1]['water_content'] - history[i]['water_content']) 
                            for i in range(total_steps-1))
    
    return {
        'final_height': final_height,
        'growth_rate': growth_rate,
        'fruit_count': fruit_count,
        'final_health': final_state['health_score'],
        'avg_health': avg_health,
        'growth_stage': final_state.get('growth_stage', 'unknown'),
        'temperature_optimal_time_pct': temp_optimal_pct,
        'light_optimal_time_pct': light_optimal_pct,
        'water_optimal_time_pct': water_optimal_pct,
        'radiation_optimal_time_pct': radiation_optimal_pct,
        'total_steps': total_steps,
        'total_temp_adjustments': total_temp_change,
        'total_light_adjustments': total_light_change,
        'total_water_adjustments': total_water_change,
    }

def generate_growth_heatmap(agent, env, param1, param2, save_path=None):
    """
    Generate data that can be used to create a heatmap showing expected plant growth 
    for different combinations of parameters
    
    Args:
        agent: Trained RL agent
        env: Space agriculture environment
        param1: Dictionary with details for first parameter {'name': 'temperature', 'range': (15, 30), 'steps': 15}
        param2: Dictionary with details for second parameter {'name': 'light_intensity', 'range': (500, 1500), 'steps': 10}
        save_path: Path to save the figure (if None, display only)
        
    Returns:
        Dictionary with heatmap data
    """
    # Create parameter grids
    p1_vals = np.linspace(param1['range'][0], param1['range'][1], param1['steps'])
    p2_vals = np.linspace(param2['range'][0], param2['range'][1], param2['steps'])
    
    # Initialize result lists (for easier JSON serialization)
    health_data = []
    growth_data = []
    
    # Reset environment
    observation, _ = env.reset()
    
    # For each parameter combination
    for j, p2_val in enumerate(p2_vals):
        health_row = []
        growth_row = []
        
        for i, p1_val in enumerate(p1_vals):
            # Create a custom state for evaluation
            test_state = env.state.copy()
            test_state[param1['name']] = p1_val
            test_state[param2['name']] = p2_val
            
            # Create observation from state
            test_obs = np.array([
                test_state['temperature'],
                test_state['light_intensity'],
                test_state['water_content'],
                test_state['radiation_level'],
                test_state['co2_level'],
                test_state['o2_level'],
                test_state['humidity'],
                test_state['nitrogen_level'],
                test_state['phosphorus_level'],
                test_state['potassium_level'],
                test_state['height'],
                test_state['health_score']
            ], dtype=np.float32)
            
            # Get agent's action for this state - handle different agent types
            if hasattr(agent, 'act'):
                action = agent.act(test_obs, explore=False)
            elif hasattr(agent, 'get_action'):
                action = agent.get_action(test_obs)
            
            # Use environment's internal models to predict outcomes
            # Get the right optimal ranges for the species
            species_ranges = env.optimal_ranges.get(env.species, env.optimal_ranges[list(env.optimal_ranges.keys())[0]])
            
            temp_opt = env._calculate_optimality(
                test_state['temperature'], 
                species_ranges['temperature'][0],
                species_ranges['temperature'][1]
            )
            
            light_opt = env._calculate_optimality(
                test_state['light_intensity'], 
                species_ranges['light_intensity'][0],
                species_ranges['light_intensity'][1]
            )
            
            water_opt = env._calculate_optimality(
                test_state['water_content'], 
                species_ranges['water_content'][0],
                species_ranges['water_content'][1]
            )
            
            # Compute expected health score
            expected_health = (
                0.3 * temp_opt + 
                0.3 * light_opt + 
                0.2 * water_opt + 
                0.2  # Base value for other factors
            )
            
            # Higher health = higher growth
            expected_growth = expected_health * 0.5  # cm per step
            
            # Store results
            health_row.append(float(expected_health))
            growth_row.append(float(expected_growth))
        
        health_data.append(health_row)
        growth_data.append(growth_row)
    
    # Extract optimal ranges if available
    optimal_ranges = {}
    species_ranges = env.optimal_ranges.get(env.species, env.optimal_ranges[list(env.optimal_ranges.keys())[0]])
    
    if param1['name'] in species_ranges:
        optimal_ranges[param1['name']] = species_ranges[param1['name']]
    if param2['name'] in species_ranges:
        optimal_ranges[param2['name']] = species_ranges[param2['name']]
    
    # Create a dictionary with the data
    result = {
        "param1": {
            "name": param1['name'],
            "values": p1_vals.tolist(),
            "min": float(np.min(p1_vals)),
            "max": float(np.max(p1_vals))
        },
        "param2": {
            "name": param2['name'],
            "values": p2_vals.tolist(),
            "min": float(np.min(p2_vals)),
            "max": float(np.max(p2_vals))
        },
        "health_data": health_data,
        "growth_data": growth_data,
        "optimal_ranges": optimal_ranges
    }
    
    # Log that we would have saved a plot in the original version
    if save_path:
        logger.info(f"Growth heatmap data for {save_path} (not saved as image due to matplotlib being unavailable)")
    
    return result
