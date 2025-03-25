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
    Visualize environmental parameters over time.
    
    Args:
        history: List of state dictionaries from each step
        save_path: Path to save the figure (if None, display only)
    
    Returns:
        matplotlib figure
    """
    # Extract data
    steps = list(range(len(history)))
    temp = [state['temperature'] for state in history]
    light = [state['light_intensity'] for state in history]
    water = [state['water_content'] for state in history]
    radiation = [state['radiation_level'] for state in history]
    co2 = [state['co2_level'] for state in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Temperature
    axes[0, 0].plot(steps, temp, 'r-')
    axes[0, 0].set_title('Temperature (°C)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Light
    axes[0, 1].plot(steps, light, 'y-')
    axes[0, 1].set_title('Light Intensity (μmol/m²/s)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Water
    axes[1, 0].plot(steps, water, 'b-')
    axes[1, 0].set_title('Water Content (%)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Radiation
    axes[1, 1].plot(steps, radiation, 'm-')
    axes[1, 1].set_title('Radiation Level')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # CO2
    axes[2, 0].plot(steps, co2, 'g-')
    axes[2, 0].set_title('CO2 Level (ppm)')
    axes[2, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Nutrients
    n_level = [state['nitrogen_level'] for state in history]
    p_level = [state['phosphorus_level'] for state in history]
    k_level = [state['potassium_level'] for state in history]
    
    axes[2, 1].plot(steps, n_level, 'g-', label='N')
    axes[2, 1].plot(steps, p_level, 'b-', label='P')
    axes[2, 1].plot(steps, k_level, 'r-', label='K')
    axes[2, 1].set_title('Nutrient Levels')
    axes[2, 1].legend()
    axes[2, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Set common x-axis label
    for ax in axes.flat:
        ax.set_xlabel('Time Steps')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Environment parameters visualization saved to {save_path}")
    
    return fig

def visualize_agent_learning(agent, save_path=None):
    """
    Visualize the agent's learning progress.
    
    Args:
        agent: RL agent with episode_rewards and avg_rewards attributes
        save_path: Path to save the figure (if None, display only)
    
    Returns:
        matplotlib figure
    """
    if not hasattr(agent, 'episode_rewards') or len(agent.episode_rewards) == 0:
        logger.warning("Agent has no recorded rewards to visualize")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot episode rewards
    episodes = range(1, len(agent.episode_rewards) + 1)
    ax.plot(episodes, agent.episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
    
    # Plot average rewards
    if hasattr(agent, 'avg_rewards') and len(agent.avg_rewards) > 0:
        ax.plot(episodes, agent.avg_rewards, 'r-', linewidth=2, label='Moving Average')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Agent Learning Progress')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Agent learning visualization saved to {save_path}")
    
    return fig

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
    Generate a heatmap showing expected plant growth for different combinations of parameters
    
    Args:
        agent: Trained RL agent
        env: Space agriculture environment
        param1: Dictionary with details for first parameter {'name': 'temperature', 'range': (15, 30), 'steps': 15}
        param2: Dictionary with details for second parameter {'name': 'light_intensity', 'range': (500, 1500), 'steps': 10}
        save_path: Path to save the figure (if None, display only)
        
    Returns:
        matplotlib figure
    """
    # Create parameter grids
    p1_vals = np.linspace(param1['range'][0], param1['range'][1], param1['steps'])
    p2_vals = np.linspace(param2['range'][0], param2['range'][1], param2['steps'])
    p1_grid, p2_grid = np.meshgrid(p1_vals, p2_vals)
    
    # Initialize result grid
    health_grid = np.zeros_like(p1_grid)
    growth_grid = np.zeros_like(p1_grid)
    
    # Reset environment
    observation, _ = env.reset()
    
    # For each parameter combination
    for i in range(len(p1_vals)):
        for j in range(len(p2_vals)):
            # Create a custom state for evaluation
            test_state = env.state.copy()
            test_state[param1['name']] = p1_grid[j, i]
            test_state[param2['name']] = p2_grid[j, i]
            
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
            
            # Get agent's action for this state
            action = agent.act(test_obs, explore=False)
            
            # Use environment's internal models to predict outcomes
            temp_opt = env._calculate_optimality(
                test_state['temperature'], 
                env.optimal_ranges['temperature'][0],
                env.optimal_ranges['temperature'][1]
            )
            
            light_opt = env._calculate_optimality(
                test_state['light_intensity'], 
                env.optimal_ranges['light_intensity'][0],
                env.optimal_ranges['light_intensity'][1]
            )
            
            water_opt = env._calculate_optimality(
                test_state['water_content'], 
                env.optimal_ranges['water_content'][0],
                env.optimal_ranges['water_content'][1]
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
            health_grid[j, i] = expected_health
            growth_grid[j, i] = expected_growth
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Health heatmap
    cmap1 = LinearSegmentedColormap.from_list('health_cmap', ['red', 'yellow', 'green'])
    c1 = ax1.pcolormesh(p1_grid, p2_grid, health_grid, cmap=cmap1, vmin=0, vmax=1)
    ax1.set_xlabel(param1['name'].replace('_', ' ').title())
    ax1.set_ylabel(param2['name'].replace('_', ' ').title())
    ax1.set_title('Expected Plant Health')
    plt.colorbar(c1, ax=ax1)
    
    # Growth heatmap
    cmap2 = 'viridis'
    c2 = ax2.pcolormesh(p1_grid, p2_grid, growth_grid, cmap=cmap2)
    ax2.set_xlabel(param1['name'].replace('_', ' ').title())
    ax2.set_ylabel(param2['name'].replace('_', ' ').title())
    ax2.set_title('Expected Growth Rate (cm/step)')
    plt.colorbar(c2, ax=ax2)
    
    # Add optimal range indicators
    if param1['name'] in env.optimal_ranges and param2['name'] in env.optimal_ranges:
        p1_opt = env.optimal_ranges[param1['name']]
        p2_opt = env.optimal_ranges[param2['name']]
        
        # Draw rectangles for optimal ranges
        for ax in [ax1, ax2]:
            rect = plt.Rectangle(
                (p1_opt[0], p2_opt[0]), 
                p1_opt[1] - p1_opt[0], 
                p2_opt[1] - p2_opt[0],
                linewidth=2, 
                edgecolor='white', 
                facecolor='none', 
                linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(
                p1_opt[0] + (p1_opt[1] - p1_opt[0])/2, 
                p2_opt[0] + (p2_opt[1] - p2_opt[0])/2,
                'Optimal',
                color='white',
                ha='center',
                va='center'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Growth heatmap saved to {save_path}")
    
    return fig
