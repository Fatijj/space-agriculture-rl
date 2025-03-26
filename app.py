"""
Streamlit dashboard for Space Agriculture RL System
"""

import streamlit as st
import pandas as pd
import numpy as np
# Import visualization library
import matplotlib.pyplot as plt

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - advanced features will be disabled")
import os
import time
from datetime import datetime
import json
import logging
from PIL import Image
import os

try:
    import kaggle
    # Set up Kaggle credentials from environment variables
    kaggle_username = os.getenv('KAGGLE_USERNAME', '')
    kaggle_key = os.getenv('KAGGLE_KEY', '')
    
    # Create Kaggle config directory if it doesn't exist
    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
    
    # Create kaggle.json configuration file
    kaggle_config = {
        "username": kaggle_username,
        "key": kaggle_key
    }
    
    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
        json.dump(kaggle_config, f)
    
    # Set appropriate permissions
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
    
except ImportError:
    logging.warning("Kaggle package not available. Some features may be limited.")
except Exception as e:
    logging.warning(f"Failed to configure Kaggle: {str(e)}")

# Import project modules
from space_agriculture_rl import SpaceAgricultureEnv
from agent import DQNAgent, PPOAgent
from utils import (visualize_growth_progress, visualize_environment_parameters, 
                 visualize_agent_learning, save_experiment_results, 
                 calculate_performance_metrics, generate_growth_heatmap)
from plant_data_generator import generate_plant_data, load_plant_data
from plant_disease_detection import (PlantDiseaseDetector, generate_report, 
                                   apply_diagnosis_to_environment, image_to_base64)
from space_agriculture_research import SpaceAgricultureKnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='streamlit_app.log')
logger = logging.getLogger('SpaceAgriRL.App')

# Page config
st.set_page_config(
    page_title="Space Agriculture RL System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title("🚀 Space Agriculture Reinforcement Learning System")

# Set language to English
st.session_state.language = 'English'

# Application description
st.markdown("""
Optimize plant growth conditions in space environments using reinforcement learning.
This application simulates different plant species growing in space conditions and trains
an AI agent to find the optimal environmental settings.
""")

# Initialize session state for storing simulation data
if 'plant_data' not in st.session_state:
    try:
        st.session_state.plant_data = load_plant_data()
    except:
        st.session_state.plant_data = generate_plant_data()

if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'env' not in st.session_state:
    st.session_state.env = None

if 'training_running' not in st.session_state:
    st.session_state.training_running = False

if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = {}
    
if 'disease_detector' not in st.session_state:
    st.session_state.disease_detector = PlantDiseaseDetector()
    
if 'diagnosis_results' not in st.session_state:
    st.session_state.diagnosis_results = None
    
if 'raw_predictions' not in st.session_state:
    st.session_state.raw_predictions = None

# Sidebar for configuration
st.sidebar.header("Configuration")

# Plant species selection
species_options = ['Dwarf Wheat', 'Cherry Tomato', 'Lettuce', 'Potato']
# Store selected species in session state so it's available across the app
plant_label = "Select Plant Species"
selected_species = st.sidebar.selectbox(plant_label, species_options)
st.session_state.selected_species = selected_species

# Agent selection
agent_label = "Select Agent Type"
agent_options = ["DQN (Deep Q-Network)", "PPO (Proximal Policy Optimization)"]
agent_type = st.sidebar.selectbox(agent_label, agent_options)

# Training parameters
st.sidebar.subheader("Training Parameters")
episodes_label = "Number of Episodes"
steps_label = "Max Steps per Episode"
num_episodes = st.sidebar.slider(episodes_label, 10, 500, 100)
max_steps_per_episode = st.sidebar.slider(steps_label, 30, 200, 50)

# Advanced parameters collapsible section
with st.sidebar.expander("Advanced Parameters"):
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    batch_size = st.number_input("Batch Size", 16, 256, 64, step=16)
    exploration_decay = st.slider("Exploration Decay", 0.9, 0.999, 0.995, step=0.001)
    randomize_initial_state = st.checkbox("Randomize Initial State", True)

# Environment configuration
st.sidebar.subheader("Environment Configuration")
custom_env_params = st.sidebar.checkbox("Customize Environment Parameters", False)

# Initialize default ranges that will be used if custom_env_params is False
temp_range = (18, 28)
light_range = (500, 1500)
water_range = (60, 85)
radiation_range = (0, 30)

if custom_env_params:
    temp_range = st.sidebar.slider("Temperature Range (°C)", 10, 40, (18, 28), 1)
    light_range = st.sidebar.slider("Light Intensity Range (μmol/m²/s)", 100, 2000, (500, 1500), 50)
    water_range = st.sidebar.slider("Water Content Range (%)", 10, 100, (60, 85), 5)
    radiation_range = st.sidebar.slider("Radiation Level Range", 0, 100, (0, 30), 5)

# Main content area with tabs
tab_names = [
    "Training", "Visualization", "Testing", "Plant Health", "Results", "Research"
]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)

# Training tab
with tab1:
    st.header("Train Reinforcement Learning Agent")
    st.markdown("""
    Train an AI agent to optimize growing conditions for the selected plant species.
    The agent will learn to adjust temperature, light, water, radiation shielding, and nutrients
    to maximize plant health and growth.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Training area
        train_button = st.button("Start Training", type="primary", disabled=st.session_state.training_running)
        stop_button = st.button("Stop Training", disabled=not st.session_state.training_running)
        
        # Training progress indicators
        progress_bar = st.progress(0)
        episode_status = st.empty()
        reward_status = st.empty()
        avg_reward_status = st.empty()
        
        # Training plot
        training_plot = st.empty()
    
    with col2:
        # Display current episode details
        st.subheader("Current Episode Details")
        episode_details = st.empty()
        
        # Environment status
        st.subheader("Environment Status")
        env_status = st.empty()
    
    # Training process function
    def train_agent():
        # Reset session state
        st.session_state.simulation_history = []
        
        # Create environment
        env = SpaceAgricultureEnv(st.session_state.plant_data, species=selected_species)
        
        # Override environment parameters if custom settings are used
        if custom_env_params:
            # Make sure the species exists in the optimal_ranges dictionary
            if selected_species not in env.optimal_ranges:
                env.optimal_ranges[selected_species] = {
                    'temperature': (20, 25),
                    'light_intensity': (800, 1200),
                    'water_content': (60, 80),
                    'radiation_level': (0, 10),
                    'nutrient_mix': (70, 90)
                }
            
            # Update with custom parameters
            env.optimal_ranges[selected_species]['temperature'] = temp_range
            env.optimal_ranges[selected_species]['light_intensity'] = light_range
            env.optimal_ranges[selected_species]['water_content'] = water_range
            env.optimal_ranges[selected_species]['radiation_level'] = radiation_range
        
        # Customize env settings
        env.max_steps = max_steps_per_episode
        
        # Initialize agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        if agent_type.startswith("DQN"):
            agent = DQNAgent(
                state_size=state_size, 
                action_size=action_size,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            agent.epsilon_decay = exploration_decay
        else:  # PPO agent
            agent = PPOAgent(
                state_size=state_size, 
                action_size=action_size,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
        
        # Store in session state
        st.session_state.env = env
        st.session_state.agent = agent
        
        all_episode_rewards = []
        rolling_reward = 0
        episode_history = []
        
        # Training loop
        for episode in range(num_episodes):
            if not st.session_state.training_running:
                break
                
            # Reset environment
            observation, info = env.reset(seed=episode if randomize_initial_state else None)
            state = observation
            episode_states = [info['state']]  # Store initial state
            
            episode_reward = 0
            done = False
            step = 0
            
            # Episode loop
            while not done and step < max_steps_per_episode:
                # Get action
                if agent_type.startswith("DQN"):
                    action = agent.act(state)
                    next_state, reward, done, _, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                else:  # PPO agent
                    action, log_prob = agent.get_action(state)
                    value = agent.get_value(state)
                    next_state, reward, done, _, info = env.step(action)
                    agent.store_transition(state, action, reward, value, log_prob, done)
                
                # Store state history
                episode_states.append(info['state'])
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                step += 1
                
                # Update UI
                if step % 5 == 0 or done:
                    # Format environment status with better display
                    state = info['state']
                    env_status.markdown(f"""
                    <div style="background-color: #f7f9fc; padding: 15px; border-radius: 10px; border-left: 5px solid #2E8B57;">
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                            <div><b>🌡️ Temperature:</b> {state['temperature']:.2f} °C</div>
                            <div><b>💧 Water Content:</b> {state['water_content']:.2f}%</div>
                            <div><b>☀️ Light Intensity:</b> {state['light_intensity']:.2f} μmol/m²/s</div>
                            <div><b>☢️ Radiation Level:</b> {state['radiation_level']:.2f}</div>
                            <div><b>🌱 Height:</b> {state['height']:.2f} cm</div>
                            <div><b>🍎 Fruit Count:</b> {state.get('fruit_count', 0)}</div>
                            <div><b>🌿 Growth Stage:</b> {state['growth_stage'].capitalize()}</div>
                            <div><b>❤️ Health Score:</b> {state['health_score']:.2f}</div>
                        </div>
                        <details>
                            <summary style="margin-top: 10px; color: #2E8B57; cursor: pointer;">Additional Parameters</summary>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px;">
                                <div><b>CO₂ Level:</b> {state['co2_level']:.2f} ppm</div>
                                <div><b>O₂ Level:</b> {state['o2_level']:.2f}%</div>
                                <div><b>Humidity:</b> {state['humidity']:.2f}%</div>
                                <div><b>Nitrogen Level:</b> {state['nitrogen_level']:.2f}</div>
                                <div><b>Phosphorus Level:</b> {state['phosphorus_level']:.2f}</div>
                                <div><b>Potassium Level:</b> {state['potassium_level']:.2f}</div>
                            </div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display current episode details
                    episode_details.markdown(f"""
                    **Step:** {step}/{max_steps_per_episode}  
                    **Health:** {info['state']['health_score']:.2f}  
                    **Growth Stage:** {info['state']['growth_stage']}  
                    **Height:** {info['state']['height']:.2f} cm  
                    **Action:** {', '.join([f"{a:.2f}" for a in action])}  
                    **Reward:** {reward:.2f}
                    """)
                    
                    # Brief delay to allow UI update
                    time.sleep(0.01)
            
            # Store episode results
            episode_history.append(episode_states)
            
            # Train on batch
            if agent_type.startswith("DQN"):
                loss_info = agent.replay()
                if loss_info:
                    logging.info(f"Episode {episode} - Actor Loss: {loss_info['actor_loss']:.4f}, Critic Loss: {loss_info['critic_loss']:.4f}")
            else:  # PPO
                if episode % 5 == 0 or episode == num_episodes - 1:  # Train every 5 episodes
                    loss_info = agent.train()
                    if loss_info:
                        logging.info(f"Episode {episode} - Actor Loss: {loss_info['actor_loss']:.4f}, Critic Loss: {loss_info['critic_loss']:.4f}")
            
            # Save rewards for plotting
            all_episode_rewards.append(episode_reward)
            agent.update_episode_rewards(episode_reward)
            
            # Calculate rolling average
            window_size = min(10, len(all_episode_rewards))
            rolling_reward = sum(all_episode_rewards[-window_size:]) / window_size
            
            # Update progress indicators
            progress = (episode + 1) / num_episodes
            progress_bar.progress(progress)
            episode_status.text(f"Episode: {episode+1}/{num_episodes}")
            reward_status.text(f"Episode Reward: {episode_reward:.2f}")
            avg_reward_status.text(f"10-Episode Average: {rolling_reward:.2f}")
            
            # Plot rewards
            if episode % 5 == 0 or episode == num_episodes - 1:
                # Create DataFrame for Streamlit chart
                chart_data = pd.DataFrame({
                    'Episode': list(range(1, len(all_episode_rewards) + 1)),
                    'Reward': all_episode_rewards
                })
                
                # Add rolling average if we have enough data
                if len(all_episode_rewards) >= 10:
                    rolling_avg = [sum(all_episode_rewards[max(0, i-9):i+1]) / min(10, i+1) for i in range(len(all_episode_rewards))]
                    chart_data['10-Episode Average'] = rolling_avg
                
                # Display training progress chart
                training_plot.line_chart(chart_data.set_index('Episode'))
        
        # Save last episode for visualization
        if episode_history:
            st.session_state.simulation_history = episode_history[-1]
        
        # Save experiment results
        if episode_history:
            config = {
                'species': selected_species,
                'agent_type': agent_type,
                'num_episodes': num_episodes,
                'max_steps': max_steps_per_episode,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'exploration_decay': exploration_decay,
                'randomize_initial_state': randomize_initial_state,
                'custom_env_params': custom_env_params
            }
            
            if custom_env_params:
                config.update({
                    'temp_range': temp_range,
                    'light_range': light_range,
                    'water_range': water_range,
                    'radiation_range': radiation_range
                })
            
            # Get optimal ranges for the selected species
            species_ranges = env.optimal_ranges.get(selected_species, env.optimal_ranges)
            metrics = calculate_performance_metrics(episode_history[-1], species_ranges)
            metrics['final_reward'] = all_episode_rewards[-1]
            metrics['avg_reward'] = rolling_reward
            
            # Save results to session state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{selected_species}_{agent_type.split()[0]}_{timestamp}"
            
            try:
                result_dir = save_experiment_results(
                    experiment_name, 
                    agent, 
                    episode_history[-1], 
                    config, 
                    metrics
                )
                
                st.session_state.experiment_results[experiment_name] = {
                    'dir': result_dir,
                    'config': config,
                    'metrics': metrics,
                    'history': episode_history[-1]
                }
                
                st.success(f"Training completed! Results saved as '{experiment_name}'")
            except Exception as e:
                logger.error(f"Error saving experiment results: {e}")
                st.error(f"Error saving experiment results: {e}")
        
        st.session_state.training_running = False
    
    # Handle training button click
    if train_button:
        st.session_state.training_running = True
        train_agent()
    
    # Handle stop button click
    if stop_button:
        st.session_state.training_running = False
        st.warning("Training stopped!")

# Visualization tab
with tab2:
    st.header("Visualize Plant Growth and Agent Performance")
    
    if st.session_state.simulation_history:
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Growth Progress", "Environmental Parameters", "Agent Learning", "Parameter Heatmap"
        ])
        
        # Growth Progress visualization
        with viz_tab1:
            st.subheader(f"{selected_species} Growth Progress")
            
            growth_data = visualize_growth_progress(st.session_state.simulation_history, selected_species)
            
            if "error" not in growth_data:
                # Plot height over time
                height_chart = pd.DataFrame({
                    'Day': growth_data["steps"],
                    'Height (cm)': growth_data["heights"]
                })
                st.line_chart(height_chart.set_index('Day'))
                
                # Plot health over time
                health_chart = pd.DataFrame({
                    'Day': growth_data["steps"],
                    'Health Score': growth_data["health"]
                })
                st.line_chart(health_chart.set_index('Day'))
                
                # Show growth stage changes
                if growth_data["stage_changes"]:
                    st.write("Growth Stage Changes:")
                    stages_df = pd.DataFrame(growth_data["stage_changes"], 
                                           columns=["Day", "New Stage"])
                    st.dataframe(stages_df)
            else:
                st.error("No growth data available to visualize")
            
            # Display growth metrics
            if st.session_state.simulation_history:
                final_state = st.session_state.simulation_history[-1]
                initial_state = st.session_state.simulation_history[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Final Height", f"{final_state['height']:.1f} cm", 
                              f"{final_state['height'] - initial_state['height']:.1f} cm")
                
                with col2:
                    st.metric("Health Score", f"{final_state['health_score']:.2f}", 
                              f"{final_state['health_score'] - initial_state['health_score']:.2f}")
                
                with col3:
                    st.metric("Growth Stage", final_state['growth_stage'])
                
                with col4:
                    if 'fruit_count' in final_state:
                        st.metric("Fruit Count", final_state['fruit_count'])
        
        # Environmental Parameters visualization
        with viz_tab2:
            st.subheader("Environmental Parameters Over Time")
            
            env_data = visualize_environment_parameters(st.session_state.simulation_history)
            
            if "error" not in env_data:
                # Create tabs for different environment parameters
                env_tabs = st.tabs(["Temperature", "Light", "Water", "Radiation", "Nutrients"])
                
                with env_tabs[0]:
                    temp_chart = pd.DataFrame({
                        'Day': env_data["steps"],
                        'Temperature (°C)': env_data["temperature"]["values"]
                    })
                    st.line_chart(temp_chart.set_index('Day'))
                    st.write(f"Min: {env_data['temperature']['min']:.1f}°C, " +
                            f"Max: {env_data['temperature']['max']:.1f}°C, " + 
                            f"Avg: {env_data['temperature']['avg']:.1f}°C")
                
                with env_tabs[1]:
                    light_chart = pd.DataFrame({
                        'Day': env_data["steps"],
                        'Light Intensity': env_data["light_intensity"]["values"]
                    })
                    st.line_chart(light_chart.set_index('Day'))
                    st.write(f"Min: {env_data['light_intensity']['min']:.0f}, " +
                            f"Max: {env_data['light_intensity']['max']:.0f}, " + 
                            f"Avg: {env_data['light_intensity']['avg']:.0f}")
                
                with env_tabs[2]:
                    water_chart = pd.DataFrame({
                        'Day': env_data["steps"],
                        'Water Content (%)': env_data["water_content"]["values"]
                    })
                    st.line_chart(water_chart.set_index('Day'))
                    st.write(f"Min: {env_data['water_content']['min']:.1f}%, " +
                            f"Max: {env_data['water_content']['max']:.1f}%, " + 
                            f"Avg: {env_data['water_content']['avg']:.1f}%")
                
                with env_tabs[3]:
                    radiation_chart = pd.DataFrame({
                        'Day': env_data["steps"],
                        'Radiation Level': env_data["radiation_level"]["values"]
                    })
                    st.line_chart(radiation_chart.set_index('Day'))
                    st.write(f"Min: {env_data['radiation_level']['min']:.1f}, " +
                            f"Max: {env_data['radiation_level']['max']:.1f}, " + 
                            f"Avg: {env_data['radiation_level']['avg']:.1f}")
                
                with env_tabs[4]:
                    nutrients_chart = pd.DataFrame({
                        'Day': env_data["steps"],
                        'Nitrogen': env_data["nutrients"]["nitrogen"]["values"],
                        'Phosphorus': env_data["nutrients"]["phosphorus"]["values"],
                        'Potassium': env_data["nutrients"]["potassium"]["values"]
                    })
                    st.line_chart(nutrients_chart.set_index('Day'))
                    n_avg = env_data["nutrients"]["nitrogen"]["avg"]
                    p_avg = env_data["nutrients"]["phosphorus"]["avg"]
                    k_avg = env_data["nutrients"]["potassium"]["avg"]
                    st.write(f"Avg N-P-K: {n_avg:.0f}-{p_avg:.0f}-{k_avg:.0f}")
            else:
                st.error("No environment data available to visualize")
            
            # Display final parameters
            if st.session_state.simulation_history:
                final_state = st.session_state.simulation_history[-1]
                
                st.subheader("Final Environmental Parameters")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Temperature", f"{final_state['temperature']:.1f}°C")
                    st.metric("Light Intensity", f"{final_state['light_intensity']:.0f} μmol/m²/s")
                
                with col2:
                    st.metric("Water Content", f"{final_state['water_content']:.1f}%")
                    st.metric("Radiation Level", f"{final_state['radiation_level']:.1f}")
                
                with col3:
                    st.metric("CO2 Level", f"{final_state['co2_level']:.0f} ppm")
                    st.metric("O2 Level", f"{final_state['o2_level']:.1f}%")
                
                with col4:
                    st.metric("Humidity", f"{final_state['humidity']:.1f}%")
                    st.metric("NPK Levels", f"{final_state['nitrogen_level']:.0f}-{final_state['phosphorus_level']:.0f}-{final_state['potassium_level']:.0f}")
        
        # Agent Learning visualization
        with viz_tab3:
            st.subheader("Agent Learning Progress")
            
            if st.session_state.agent and hasattr(st.session_state.agent, 'episode_rewards') and len(st.session_state.agent.episode_rewards) > 0:
                agent_data = visualize_agent_learning(st.session_state.agent)
                
                if "error" not in agent_data:
                    # Plot rewards
                    rewards_chart = pd.DataFrame({
                        'Episode': agent_data["episodes"],
                        'Reward': agent_data["episode_rewards"]
                    })
                    
                    if agent_data["moving_avg_rewards"]:
                        rewards_chart['Moving Avg'] = agent_data["moving_avg_rewards"]
                    
                    st.line_chart(rewards_chart.set_index('Episode'))
                    
                    # Display statistics
                    if isinstance(agent_data, dict) and "stats" in agent_data:
                        stats = agent_data["stats"]
                        total_episodes = stats.get("total_episodes", 0)
                        max_reward = stats.get("max_reward", 0)
                        avg_reward = stats.get("avg_reward", 0)
                    else:
                        total_episodes = len(agent_data.get("episodes", []))
                        rewards = agent_data.get("episode_rewards", [])
                        max_reward = max(rewards) if rewards else 0
                        avg_reward = sum(rewards) / len(rewards) if rewards else 0
                    
                    st.write(f"Episodes: {total_episodes}, " +
                           f"Max Reward: {max_reward:.2f}, " +
                           f"Avg Reward: {avg_reward:.2f}")
                
                # Display learning metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Final Episode Reward", f"{st.session_state.agent.episode_rewards[-1]:.2f}")
                
                with col2:
                    if hasattr(st.session_state.agent, 'avg_rewards') and len(st.session_state.agent.avg_rewards) > 0:
                        st.metric("Average Reward", f"{st.session_state.agent.avg_rewards[-1]:.2f}")
                
                with col3:
                    if agent_type.startswith("DQN") and hasattr(st.session_state.agent, 'epsilon'):
                        st.metric("Final Exploration Rate", f"{st.session_state.agent.epsilon:.3f}")
            else:
                st.info("Train the agent to see learning progress!")
        
        # Parameter Heatmap visualization
        with viz_tab4:
            st.subheader("Parameter Impact Heatmap")
            
            if st.session_state.agent and st.session_state.env:
                # Parameter selection for heatmap
                col1, col2 = st.columns(2)
                
                with col1:
                    param1_name = st.selectbox("X-Axis Parameter", 
                                             ["temperature", "light_intensity", "water_content", "radiation_level"])
                    
                    if param1_name == "temperature":
                        param1_range = st.slider("Temperature Range (°C)", 10, 40, (15, 35), 1)
                    elif param1_name == "light_intensity":
                        param1_range = st.slider("Light Range (μmol/m²/s)", 100, 2000, (400, 1600), 50)
                    elif param1_name == "water_content":
                        param1_range = st.slider("Water Range (%)", 10, 100, (30, 90), 5)
                    else:  # radiation_level
                        param1_range = st.slider("Radiation Range", 0, 100, (0, 50), 5)
                
                with col2:
                    param2_name = st.selectbox("Y-Axis Parameter", 
                                             ["light_intensity", "temperature", "water_content", "radiation_level"],
                                             index=1)
                    
                    if param2_name == "temperature":
                        param2_range = st.slider("Temperature Range (°C) ", 10, 40, (15, 35), 1)
                    elif param2_name == "light_intensity":
                        param2_range = st.slider("Light Range (μmol/m²/s) ", 100, 2000, (400, 1600), 50)
                    elif param2_name == "water_content":
                        param2_range = st.slider("Water Range (%) ", 10, 100, (30, 90), 5)
                    else:  # radiation_level
                        param2_range = st.slider("Radiation Range ", 0, 100, (0, 50), 5)
                
                # Make sure we don't have same parameter on both axes
                if param1_name == param2_name:
                    st.warning("Please select different parameters for X and Y axes.")
                else:
                    # Create heatmap
                    param1 = {
                        'name': param1_name,
                        'range': param1_range,
                        'steps': 15
                    }
                    
                    param2 = {
                        'name': param2_name,
                        'range': param2_range,
                        'steps': 15
                    }
                    
                    with st.spinner("Generating heatmap..."):
                        heatmap_data = generate_growth_heatmap(
                            st.session_state.agent,
                            st.session_state.env,
                            param1,
                            param2
                        )
                        
                        if "error" not in heatmap_data:
                            # Create health heatmap
                            st.subheader("Expected Plant Health")
                            
                            # Create dataframe for health data
                            health_df = pd.DataFrame(
                                heatmap_data["health_data"],
                                index=heatmap_data["param2"]["values"],
                                columns=heatmap_data["param1"]["values"]
                            )
                            
                            # Display heatmap
                            st.write(f"{param2_name.replace('_', ' ').title()} vs {param1_name.replace('_', ' ').title()}")
                            st.dataframe(health_df.style.background_gradient(cmap="RdYlGn", axis=None))
                            
                            # Create growth heatmap
                            st.subheader("Expected Growth Rate (cm/step)")
                            
                            # Create dataframe for growth data
                            growth_df = pd.DataFrame(
                                heatmap_data["growth_data"],
                                index=heatmap_data["param2"]["values"],
                                columns=heatmap_data["param1"]["values"]
                            )
                            
                            # Display heatmap
                            st.dataframe(growth_df.style.background_gradient(cmap="viridis", axis=None))
                            
                            # Show optimal ranges if available
                            if heatmap_data["optimal_ranges"]:
                                st.subheader("Optimal Ranges")
                                for param, range_vals in heatmap_data["optimal_ranges"].items():
                                    st.write(f"{param.replace('_', ' ').title()}: {range_vals[0]} to {range_vals[1]}")
                        else:
                            st.error("Error generating heatmap data")
            else:
                st.info("Train the agent first to see parameter heatmaps!")
    else:
        st.info("Train the agent first to visualize results!")

# Testing tab
with tab3:
    st.header("Test Trained Agent")
    
    if st.session_state.agent and st.session_state.env:
        # Test configuration
        test_col1, test_col2 = st.columns([1, 2])
        
        with test_col1:
            st.subheader("Test Configuration")
            test_episodes = st.number_input("Number of Test Episodes", 1, 10, 3)
            reset_between_episodes = st.checkbox("Reset Environment Between Episodes", True)
            use_exploration = st.checkbox("Use Exploration During Testing", False)
            
            # Initial conditions customization
            custom_initial = st.checkbox("Custom Initial Conditions", False)
            
            # Initialize variables with default values
            init_temp = 22
            init_light = 1000
            init_water = 70
            init_radiation = 20
            
            if custom_initial:
                init_temp = st.slider("Initial Temperature (°C)", 10, 40, init_temp)
                init_light = st.slider("Initial Light (μmol/m²/s)", 100, 2000, init_light, 50)
                init_water = st.slider("Initial Water (%)", 10, 100, init_water, 5)
                init_radiation = st.slider("Initial Radiation", 0, 100, init_radiation, 5)
            
            # Start test button
            start_test = st.button("Start Test", type="primary")
        
        with test_col2:
            # Test results will be displayed here
            st.subheader("Test Results")
            test_result_container = st.container()
            test_result_area = test_result_container.empty()
            
            # Test metrics
            metrics_container = st.container()
        
        # Run test function
        if start_test:
            test_result_area.text("Testing in progress...")
            
            all_test_states = []
            all_test_rewards = []
            all_test_metrics = []
            
            # Run test episodes
            for episode in range(test_episodes):
                # Reset environment
                if custom_initial:
                    observation, info = st.session_state.env.reset()
                    
                    # Set custom initial values
                    st.session_state.env.state['temperature'] = init_temp
                    st.session_state.env.state['light_intensity'] = init_light
                    st.session_state.env.state['water_content'] = init_water
                    st.session_state.env.state['radiation_level'] = init_radiation
                    
                    # Update observation
                    observation = st.session_state.env._get_observation()
                else:
                    observation, info = st.session_state.env.reset(seed=episode if reset_between_episodes else None)
                
                state = observation
                episode_states = [info['state']]
                episode_reward = 0
                done = False
                
                # Episode loop
                while not done:
                    if agent_type.startswith("DQN"):
                        action = st.session_state.agent.act(state, explore=use_exploration)
                    else:  # PPO
                        action, _ = st.session_state.agent.get_action(state)
                        if not use_exploration:  # Use deterministic action
                            action_mean, _ = st.session_state.agent.actor(np.array([state]))[0]
                            action = action_mean.numpy()
                    
                    next_state, reward, done, _, info = st.session_state.env.step(action)
                    episode_states.append(info['state'])
                    state = next_state
                    episode_reward += reward
                    
                    # Show current state with bilingual format (English and Arabic)
                    state = info['state']
                    # Define Arabic translations
                    arabic_translations = {
                        "temperature": "درجة الحرارة",
                        "light_intensity": "شدة الضوء",
                        "water_content": "محتوى الماء",
                        "radiation_level": "مستوى الإشعاع",
                        "height": "الارتفاع",
                        "growth_stage": "مرحلة النمو",
                        "health_score": "مؤشر الصحة",
                        "fruit_count": "عدد الثمار",
                        "co2_level": "مستوى ثاني أكسيد الكربون",
                        "o2_level": "مستوى الأكسجين",
                        "humidity": "الرطوبة",
                        "nitrogen_level": "مستوى النيتروجين",
                        "phosphorus_level": "مستوى الفوسفور",
                        "potassium_level": "مستوى البوتاسيوم",
                        # Growth stages
                        "germination": "الإنبات",
                        "seedling": "الشتلة",
                        "vegetative": "النمو الخضري",
                        "budding": "التبرعم",
                        "flowering": "الإزهار",
                        "fruiting": "الإثمار",
                        "mature": "النضج"
                    }
                    
                    # Format the test results with a nice layout
                    test_result_area.markdown(f"""
                    <div style="background-color: #f7f9fc; padding: 15px; border-radius: 10px; border-left: 5px solid #2E8B57;">
                        <h3 style="text-align: center; color: #2E8B57; font-size: 1.2rem;">
                            Test Results | نتائج الاختبار
                        </h3>
                        
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px;">
                            <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                <div style="font-weight: bold;">🌡️ Temperature | {arabic_translations["temperature"]}</div>
                                <div dir="ltr">{state['temperature']:.2f} °C</div>
                            </div>
                            
                            <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                <div style="font-weight: bold;">💧 Water Content | {arabic_translations["water_content"]}</div>
                                <div dir="ltr">{state['water_content']:.2f}%</div>
                            </div>
                            
                            <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                <div style="font-weight: bold;">☀️ Light Intensity | {arabic_translations["light_intensity"]}</div>
                                <div dir="ltr">{state['light_intensity']:.2f} μmol/m²/s</div>
                            </div>
                            
                            <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                <div style="font-weight: bold;">☢️ Radiation | {arabic_translations["radiation_level"]}</div>
                                <div dir="ltr">{state['radiation_level']:.2f}</div>
                            </div>
                            
                            <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                <div style="font-weight: bold;">🌱 Height | {arabic_translations["height"]}</div>
                                <div dir="ltr">{state['height']:.2f} cm</div>
                            </div>
                            
                            <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                <div style="font-weight: bold;">❤️ Health | {arabic_translations["health_score"]}</div>
                                <div dir="ltr">{state['health_score']:.2f}</div>
                            </div>
                        </div>
                        
                        <div style="background-color: #e8f4ea; padding: 10px; border-radius: 5px; margin-bottom: 12px;">
                            <span style="font-weight: bold;">🌿 Growth Stage | {arabic_translations["growth_stage"]}: </span>
                            <span>{state['growth_stage'].capitalize()} | {arabic_translations.get(state['growth_stage'], state['growth_stage'])}</span>
                            {f'<span style="margin-left: 15px;"><b>🍎 Fruit Count | {arabic_translations["fruit_count"]}:</b> {state.get("fruit_count", 0)}</span>' if "fruit_count" in state and state["fruit_count"] else ""}
                        </div>
                        
                        <details>
                            <summary style="margin-top: 10px; color: #2E8B57; cursor: pointer; font-weight: bold;">
                                Additional Parameters | معلومات إضافية
                            </summary>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 10px;">
                                <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                    <div style="font-weight: bold;">CO₂ | {arabic_translations["co2_level"]}</div>
                                    <div dir="ltr">{state['co2_level']:.2f} ppm</div>
                                </div>
                                
                                <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                    <div style="font-weight: bold;">O₂ | {arabic_translations["o2_level"]}</div>
                                    <div dir="ltr">{state['o2_level']:.2f}%</div>
                                </div>
                                
                                <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                    <div style="font-weight: bold;">Humidity | {arabic_translations["humidity"]}</div>
                                    <div dir="ltr">{state['humidity']:.2f}%</div>
                                </div>
                                
                                <div style="border-bottom: 1px solid #eaecef; padding: 8px;">
                                    <div style="font-weight: bold;">NPK Levels</div>
                                    <div dir="ltr">{state['nitrogen_level']:.1f}-{state['phosphorus_level']:.1f}-{state['potassium_level']:.1f}</div>
                                </div>
                            </div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(0.1)  # Brief delay to show progress
                
                # Calculate metrics for this episode
                # Get optimal ranges for the selected species
                species_ranges = st.session_state.env.optimal_ranges.get(selected_species, st.session_state.env.optimal_ranges)
                metrics = calculate_performance_metrics(
                    episode_states, 
                    species_ranges
                )
                metrics['episode_reward'] = episode_reward
                
                # Store results
                all_test_states.append(episode_states)
                all_test_rewards.append(episode_reward)
                all_test_metrics.append(metrics)
            
            # Show test results with bilingual success message
            if st.session_state.language == 'English':
                test_result_area.markdown(f"""
                <div style="background-color: #e8f4ea; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;">
                    <h3 style="color: #2E8B57; margin-bottom: 10px;">
                        Testing Completed!
                    </h3>
                    <div style="font-size: 1.1rem;">
                        <span>Average Reward: </span>
                        <span style="font-weight: bold; color: #2E8B57;">{sum(all_test_rewards)/len(all_test_rewards):.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                test_result_area.markdown(f"""
                <div style="background-color: #e8f4ea; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;">
                    <h3 style="color: #2E8B57; margin-bottom: 10px;">
                        اكتمل الاختبار!
                    </h3>
                    <div style="font-size: 1.1rem;">
                        <span>متوسط المكافأة: </span>
                        <span style="font-weight: bold; color: #2E8B57;">{sum(all_test_rewards)/len(all_test_rewards):.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display metrics
            with metrics_container:
                st.subheader("Test Metrics")
                
                # Calculate averages
                avg_metrics = {}
                for key in all_test_metrics[0].keys():
                    # Check if all values for this key are numeric before calculating average
                    if all(isinstance(m[key], (int, float)) for m in all_test_metrics):
                        avg_metrics[key] = sum(m[key] for m in all_test_metrics) / len(all_test_metrics)
                    else:
                        # For non-numeric values, use the most common value
                        avg_metrics[key] = all_test_metrics[0][key]  # Just use the first one as fallback
                
                # Display in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg. Reward", f"{avg_metrics['episode_reward']:.2f}")
                    st.metric("Avg. Final Height", f"{avg_metrics['final_height']:.2f} cm")
                
                with col2:
                    if st.session_state.language == 'English':
                        st.metric("Avg. Health", f"{avg_metrics['avg_health']:.2f}")
                        st.metric("Optimal Temp. Time", f"{avg_metrics['temperature_optimal_time_pct']:.1f}%")
                    else:
                        st.metric("متوسط الصحة", f"{avg_metrics['avg_health']:.2f}")
                        st.metric("وقت درجة الحرارة المثلى", f"{avg_metrics['temperature_optimal_time_pct']:.1f}%")
                
                with col3:
                    if st.session_state.language == 'English':
                        st.metric("Avg. Fruit Count", f"{avg_metrics['fruit_count']:.1f}")
                        st.metric("Optimal Light Time", f"{avg_metrics['light_optimal_time_pct']:.1f}%")
                    else:
                        st.metric("متوسط عدد الثمار", f"{avg_metrics['fruit_count']:.1f}")
                        st.metric("وقت الإضاءة المثلى", f"{avg_metrics['light_optimal_time_pct']:.1f}%")
                
                # Growth visualization for the last test episode
                if st.session_state.language == 'English':
                    st.subheader("Last Test Episode Growth")
                else:
                    st.subheader("نمو آخر اختبار")
                    
                growth_data = visualize_growth_progress(all_test_states[-1], selected_species)
                
                if "error" not in growth_data:
                    # Plot height over time
                    if st.session_state.language == 'English':
                        height_chart = pd.DataFrame({
                            'Day': growth_data["steps"],
                            'Height (cm)': growth_data["heights"]
                        })
                    else:
                        height_chart = pd.DataFrame({
                            'اليوم': growth_data["steps"],
                            'الارتفاع (سم)': growth_data["heights"]
                        })
                    st.line_chart(height_chart.set_index('Day' if st.session_state.language == 'English' else 'اليوم'))
                    
                    # Plot health over time
                    if st.session_state.language == 'English':
                        health_chart = pd.DataFrame({
                            'Day': growth_data["steps"],
                            'Health Score': growth_data["health"]
                        })
                    else:
                        health_chart = pd.DataFrame({
                            'اليوم': growth_data["steps"],
                            'مؤشر الصحة': growth_data["health"]
                        })
                    st.line_chart(health_chart.set_index('Day' if st.session_state.language == 'English' else 'اليوم'))
                else:
                    if st.session_state.language == 'English':
                        st.error("No growth data available to visualize")
                    else:
                        st.error("لا توجد بيانات نمو متاحة للعرض")
                
                # Environmental parameters
                if st.session_state.language == 'English':
                    st.subheader("Last Test Episode Environment")
                else:
                    st.subheader("بيئة آخر اختبار")
                    
                env_data = visualize_environment_parameters(all_test_states[-1])
                
                if "error" not in env_data:
                    # Create tabs for different environment parameters
                    if st.session_state.language == 'English':
                        env_tabs = st.tabs(["Temperature", "Light", "Water", "Other"])
                    else:
                        env_tabs = st.tabs(["درجة الحرارة", "الإضاءة", "الماء", "أخرى"])
                    
                    with env_tabs[0]:
                        if st.session_state.language == 'English':
                            temp_chart = pd.DataFrame({
                                'Day': env_data["steps"],
                                'Temperature (°C)': env_data["temperature"]["values"]
                            })
                            st.line_chart(temp_chart.set_index('Day'))
                        else:
                            temp_chart = pd.DataFrame({
                                'اليوم': env_data["steps"],
                                'درجة الحرارة (°م)': env_data["temperature"]["values"]
                            })
                            st.line_chart(temp_chart.set_index('اليوم'))
                    
                    with env_tabs[1]:
                        if st.session_state.language == 'English':
                            light_chart = pd.DataFrame({
                                'Day': env_data["steps"],
                                'Light Intensity': env_data["light_intensity"]["values"]
                            })
                            st.line_chart(light_chart.set_index('Day'))
                        else:
                            light_chart = pd.DataFrame({
                                'اليوم': env_data["steps"],
                                'شدة الإضاءة': env_data["light_intensity"]["values"]
                            })
                            st.line_chart(light_chart.set_index('اليوم'))
                    
                    with env_tabs[2]:
                        if st.session_state.language == 'English':
                            water_chart = pd.DataFrame({
                                'Day': env_data["steps"],
                                'Water Content (%)': env_data["water_content"]["values"]
                            })
                            st.line_chart(water_chart.set_index('Day'))
                        else:
                            water_chart = pd.DataFrame({
                                'اليوم': env_data["steps"],
                                'محتوى الماء (%)': env_data["water_content"]["values"]
                            })
                            st.line_chart(water_chart.set_index('اليوم'))
                    
                    with env_tabs[3]:
                        # Create combined chart for other parameters
                        if st.session_state.language == 'English':
                            other_chart = pd.DataFrame({
                                'Day': env_data["steps"],
                                'Radiation': env_data["radiation_level"]["values"],
                                'CO2': [x/1000 for x in env_data["co2_level"]["values"]]  # Scale down for better visualization
                            })
                            st.line_chart(other_chart.set_index('Day'))
                        else:
                            other_chart = pd.DataFrame({
                                'اليوم': env_data["steps"],
                                'الإشعاع': env_data["radiation_level"]["values"],
                                'ثاني أكسيد الكربون': [x/1000 for x in env_data["co2_level"]["values"]]  # Scale down for better visualization
                            })
                            st.line_chart(other_chart.set_index('اليوم'))
                else:
                    if st.session_state.language == 'English':
                        st.error("No environment data available to visualize")
                    else:
                        st.error("لا توجد بيانات بيئية متاحة للعرض")
    else:
        st.info("Train the agent first to test its performance!")

# Plant Health Monitoring tab
with tab4:
    st.header("Plant Health Monitoring")
    st.markdown("""
    Monitor plant health in real-time using computer vision and disease detection.
    Upload an image or use your camera to capture plants and receive diagnostic information.
    The diagnosis will influence the reinforcement learning agent's decision making.
    
    *This system now incorporates the Plant Pathology 2020 dataset for improved accuracy in disease detection.*
    """)
    
    # Add information about the dataset in an expandable section
    with st.expander("About Plant Pathology Dataset"):
        st.markdown("""
        ### Plant Pathology 2020 Dataset
        
        The Plant Pathology 2020 dataset contains images of apple leaves with various health conditions:
        - **Healthy**: Normal apple leaves with no visible symptoms
        - **Multiple Diseases**: Leaves affected by more than one disease
        - **Rust**: Leaves with rust infection, showing orange-brown pustules
        - **Scab**: Leaves with apple scab, showing dark olive-green to brown lesions
        
        This dataset is used to train our disease detection model for more accurate diagnostics.
        The system combines this knowledge with space-specific conditions like microgravity stress
        and radiation damage to provide comprehensive plant health monitoring for space agriculture.
        """)
        
        # Try to display dataset statistics if available
        try:
            from plant_pathology_dataset import PlantPathologyDataset
            
            # Initialize the dataset
            dataset = PlantPathologyDataset()
            stats = dataset.get_statistics()
            
            # Display dataset statistics
            st.subheader("Dataset Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Samples", stats.get('train_samples', 0))
            with col2:
                st.metric("Testing Samples", stats.get('test_samples', 0))
            
            # Display class distribution if available
            class_dist = stats.get('class_distribution', {})
            if class_dist:
                st.subheader("Class Distribution")
                data = {
                    "Class": list(class_dist.keys()),
                    "Count": [d.get('count', 0) for d in class_dist.values()],
                    "Percentage": [f"{d.get('percentage', 0):.1f}%" for d in class_dist.values()]
                }
                st.dataframe(pd.DataFrame(data))
        except Exception as e:
            st.warning(f"Could not load Plant Pathology dataset statistics: {str(e)}")
    
    # Create main columns for the tab
    health_col1, health_col2 = st.columns([1, 1])
    
    with health_col1:
        # Image upload/capture area
        st.subheader("Plant Image Input")
        # Option to upload an image or use camera
        img_source = st.radio("Select image source:", ["Upload Image", "Use Camera"], index=0)
        
        uploaded_image = None
        camera_image = None
        
        if img_source == "Upload Image":
            uploaded_image = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])
            caption = "Uploaded Image"
            button_text = "Analyze Plant Health"
                
            if uploaded_image is not None:
                st.image(uploaded_image, caption=caption, width=300)
                
                # Convert the uploaded file to numpy array
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                img_array = np.array(Image.open(uploaded_image))
                
                # Add button to analyze
                if st.button(button_text, key="analyze_upload"):
                    predictions = st.session_state.disease_detector.predict(img_array)
                    diagnosis = st.session_state.disease_detector.get_diagnosis(predictions)
                    st.session_state.diagnosis_results = diagnosis
                    # Store raw predictions in session state for the disease table
                    st.session_state.raw_predictions = predictions
        else:
            # Camera input
            camera_input = st.camera_input("Take a picture of the plant")
            caption = "Captured Image"
            button_text = "Analyze Plant Health"
                
            if camera_input is not None:
                st.image(camera_input, caption=caption, width=300)
                
                # Convert the camera input to numpy array
                img_array = np.array(Image.open(camera_input))
                
                # Add button to analyze
                if st.button(button_text, key="analyze_camera"):
                    predictions = st.session_state.disease_detector.predict(img_array)
                    diagnosis = st.session_state.disease_detector.get_diagnosis(predictions)
                    st.session_state.diagnosis_results = diagnosis
                    # Store raw predictions in session state for the disease table
                    st.session_state.raw_predictions = predictions
    
    with health_col2:
        # Display diagnosis results
        st.subheader("Plant Health Diagnosis")
        
        if st.session_state.diagnosis_results:
            diagnosis = st.session_state.diagnosis_results
            
            # Display health status with color
            health_status = diagnosis.get('health_status', 'Unknown')
            if health_status == "Healthy":
                status_color = "green"
                status_emoji = "✅"
            elif health_status == "Moderate Risk":
                status_color = "orange"
                status_emoji = "⚠️"
            elif health_status == "Severe Risk":
                status_color = "red"
                status_emoji = "🚨"
            else:  # Unknown
                status_color = "gray"
                status_emoji = "❓"
            
            st.markdown(f"<h3 style='color:{status_color};'>{status_emoji} Status: {health_status}</h3>", unsafe_allow_html=True)
            
            # Display confidence level
            st.write(f"Confidence: {diagnosis['confidence']*100:.1f}%")
            
            # Add Disease Detection Table
            if st.session_state.raw_predictions:
                st.subheader("Disease Detection Analysis")
                
                # Create a DataFrame for the disease probabilities
                predictions = st.session_state.raw_predictions
                disease_df = pd.DataFrame({
                    "Condition": [k.replace('_', ' ').title() for k in predictions.keys()],
                    "Probability": [f"{v*100:.1f}%" for v in predictions.values()],
                    "Value": list(predictions.values())  # Hidden column for sorting
                })
                
                # Sort by probability (highest first)
                disease_df = disease_df.sort_values(by="Value", ascending=False).reset_index(drop=True)
                
                # Display the dataframe without the hidden Value column
                st.dataframe(disease_df[["Condition", "Probability"]])
                
                # Add conclusion based on highest probability
                highest_condition = disease_df.iloc[0]["Condition"]
                highest_prob = disease_df.iloc[0]["Value"] * 100
                
                conclusion_color = "green"
                if highest_condition != "Healthy":
                    if highest_prob > 60:
                        conclusion_color = "red"
                    else:
                        conclusion_color = "orange"
                        
                st.markdown(f"""
                <div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; border-left: 5px solid {conclusion_color};'>
                    <b>Conclusion:</b> Plant is most likely <span style='color: {conclusion_color};'><b>{highest_condition}</b></span> 
                    with {highest_prob:.1f}% probability.
                </div>
                """, unsafe_allow_html=True)
            
            # Display disease information if detected
            if 'disease_name' in diagnosis and diagnosis['disease_name'] != "None":
                st.write(f"**Detected Issue:** {diagnosis['disease_name']}")
                st.write(f"**Severity:** {diagnosis['disease_severity']*100:.1f}%")
                
                # Display recommendations
                st.subheader("Recommendations")
                
                for i, recommendation in enumerate(diagnosis['recommendations']):
                    st.write(f"{i+1}. {recommendation}")
                    
            # Action to apply to environment
            button_text = "Apply Diagnosis to Environment"
                
            if st.button(button_text, key="apply_diagnosis"):
                if st.session_state.env is not None:
                    disease_modifier = st.session_state.env.update_disease_modifier(diagnosis)
                    st.success(f"Applied disease status to environment. Reward modifier: {disease_modifier:.2f}")
                else:
                    # Create a temporary environment just to show the modifier value
                    from space_agriculture_rl import SpaceAgricultureEnv
                    temp_env = SpaceAgricultureEnv(st.session_state.plant_data, st.session_state.selected_species)
                    disease_modifier = temp_env.update_disease_modifier(diagnosis)
                    st.success(f"Diagnosis recorded. If you train an agent, this will apply a reward modifier of: {disease_modifier:.2f}")
                    st.info("For full functionality, train an agent in the Training tab to create a persistent environment.")
        else:
            st.info("No diagnosis available. Please upload or capture an image and analyze it.")
            
        # Historical diagnoses section
        with st.expander("Historical Disease Data"):
            st.write("This section will show historical disease detection data.")
            # This would be populated with real data in a full implementation

# Results tab
with tab5:
    st.header("Saved Results")
    
    if st.session_state.experiment_results:
        # Display saved experiments
        st.subheader("Previous Experiments")
        
        # Select experiment to view
        experiment_names = list(st.session_state.experiment_results.keys())
        selected_experiment = st.selectbox("Select Experiment", experiment_names)
        
        if selected_experiment:
            experiment = st.session_state.experiment_results[selected_experiment]
            
            # Display experiment details
            st.subheader(f"Experiment: {selected_experiment}")
            
            # Configuration
            with st.expander("Configuration", expanded=True):
                st.json(experiment['config'])
            
            # Metrics
            with st.expander("Performance Metrics", expanded=True):
                st.json(experiment['metrics'])
            
            # Visualizations
            st.subheader("Visualizations")
            
            species = experiment['config']['species']
            growth_data = visualize_growth_progress(experiment['history'], species)
            env_data = visualize_environment_parameters(experiment['history'])
            
            # Create tabs for different visualizations
            result_tabs = st.tabs(["Growth", "Environment", "Key Metrics"])
            
            # Growth tab
            with result_tabs[0]:
                if "error" not in growth_data:
                    # Plot height over time
                    st.subheader("Plant Height Progress")
                    height_chart = pd.DataFrame({
                        'Day': growth_data["steps"],
                        'Height (cm)': growth_data["heights"]
                    })
                    st.line_chart(height_chart.set_index('Day'))
                    
                    # Plot health over time
                    st.subheader("Plant Health Progress")
                    health_chart = pd.DataFrame({
                        'Day': growth_data["steps"],
                        'Health Score': growth_data["health"]
                    })
                    st.line_chart(health_chart.set_index('Day'))
                else:
                    st.error("No growth data available to visualize")
            
            # Environment tab
            with result_tabs[1]:
                if "error" not in env_data:
                    # Create environmental parameter charts
                    st.subheader("Environmental Parameters")
                    env_subtabs = st.tabs(["Temperature", "Light", "Water", "Radiation"])
                    
                    with env_subtabs[0]:
                        temp_chart = pd.DataFrame({
                            'Day': env_data["steps"],
                            'Temperature (°C)': env_data["temperature"]["values"]
                        })
                        st.line_chart(temp_chart.set_index('Day'))
                        st.write(f"Min: {env_data['temperature']['min']:.1f}°C, " +
                                f"Max: {env_data['temperature']['max']:.1f}°C, " + 
                                f"Avg: {env_data['temperature']['avg']:.1f}°C")
                    
                    with env_subtabs[1]:
                        light_chart = pd.DataFrame({
                            'Day': env_data["steps"],
                            'Light Intensity': env_data["light_intensity"]["values"]
                        })
                        st.line_chart(light_chart.set_index('Day'))
                        st.write(f"Min: {env_data['light_intensity']['min']:.0f}, " +
                                f"Max: {env_data['light_intensity']['max']:.0f}, " + 
                                f"Avg: {env_data['light_intensity']['avg']:.0f}")
                    
                    with env_subtabs[2]:
                        water_chart = pd.DataFrame({
                            'Day': env_data["steps"],
                            'Water Content (%)': env_data["water_content"]["values"]
                        })
                        st.line_chart(water_chart.set_index('Day'))
                        st.write(f"Min: {env_data['water_content']['min']:.1f}%, " +
                                f"Max: {env_data['water_content']['max']:.1f}%, " + 
                                f"Avg: {env_data['water_content']['avg']:.1f}%")
                    
                    with env_subtabs[3]:
                        radiation_chart = pd.DataFrame({
                            'Day': env_data["steps"],
                            'Radiation Level': env_data["radiation_level"]["values"]
                        })
                        st.line_chart(radiation_chart.set_index('Day'))
                        st.write(f"Min: {env_data['radiation_level']['min']:.1f}, " +
                                f"Max: {env_data['radiation_level']['max']:.1f}, " + 
                                f"Avg: {env_data['radiation_level']['avg']:.1f}")
                else:
                    st.error("No environment data available")
            
            # Key Metrics tab
            with result_tabs[2]:
                # Extract and display key metrics
                metrics = experiment['metrics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Final Height", f"{metrics.get('final_height', 0):.1f} cm")
                    st.metric("Final Health", f"{metrics.get('final_health', 0):.2f}")
                    st.metric("Fruit Count", f"{metrics.get('fruit_count', 0)}")
                
                with col2:
                    st.metric("Final Reward", f"{metrics.get('final_reward', 0):.2f}")
                    st.metric("Average Reward", f"{metrics.get('avg_reward', 0):.2f}")
                    st.metric("Growth Rate", f"{metrics.get('growth_rate', 0):.2f} cm/day")
            
            # Download results
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps({
                    'config': experiment['config'],
                    'metrics': experiment['metrics']
                }, indent=2),
                file_name=f"{selected_experiment}_results.json",
                mime="application/json"
            )
    else:
        st.info("Train the agent to save results!")

    # Option to load models from disk
    st.subheader("Load Saved Models")
    
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        actor_path = st.text_input("Actor Model Path", "actor_model.h5")
    
    with model_col2:
        critic_path = st.text_input("Critic Model Path", "critic_model.h5")
    
    load_model_button = st.button("Load Models")
    
    if load_model_button:
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            # Create agent if needed
            if st.session_state.agent is None:
                # Create environment first to get state/action dimensions
                env = SpaceAgricultureEnv(st.session_state.plant_data, species=selected_species)
                state_size = env.observation_space.shape[0]
                action_size = env.action_space.shape[0]
                
                if agent_type.startswith("DQN"):
                    st.session_state.agent = DQNAgent(state_size, action_size)
                else:
                    st.session_state.agent = PPOAgent(state_size, action_size)
                
                st.session_state.env = env
            
            # Load models
            success = st.session_state.agent.load_model(actor_path, critic_path)
            
            if success:
                st.success(f"Models loaded successfully from {actor_path} and {critic_path}")
            else:
                st.error("Failed to load models, please check the file paths")
        else:
            st.error("Model files not found, please check the file paths")

# Research Knowledge Base tab
with tab6:
    st.header("Space Agriculture Research Knowledge Base")
    st.markdown("""
    This tab presents scientific research findings about plant growth in space environments.
    The knowledge from these research papers is integrated into our reinforcement learning system
    to improve decision-making and optimize growing conditions.
    """)
    
    # Initialize the knowledge base
    knowledge_base = SpaceAgricultureKnowledgeBase()
    
    # Create tabs for different research domains
    research_domains = ["Microgravity Effects", "Radiation Impact", "Light Optimization", 
                       "Nutrient Delivery", "Environmental Control", "Crop Selection"]
    
    domain_tabs = st.tabs(research_domains)
    
    with domain_tabs[0]:
        st.subheader("Microgravity Effects on Plant Growth")
        st.markdown("""
        ### Research Findings
        
        Research has shown that microgravity significantly affects plant growth through multiple mechanisms:
        
        - **Root Orientation Challenges**: In microgravity, plant roots struggle with directional growth as they lack gravitational cues.
        - **Altered Fluid Dynamics**: Water and nutrient distribution within plants follows different patterns due to reduced gravitational forces.
        - **Cell Wall Development**: Microgravity results in thinner cell walls and altered lignin deposition compared to Earth gravity.
        - **Hormone Transport Disruption**: The movement of auxins and other plant hormones is affected, altering growth patterns and tropisms.
        
        ### Key Parameters
        """)
        
        # Extract and display optimal parameters
        micro_params = knowledge_base.get_optimal_parameters("microgravity")
        if micro_params:
            for param, value in micro_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    st.info(f"**{param.replace('_', ' ').title()}**: Optimal range is {value[0]} to {value[1]}")
                else:
                    st.info(f"**{param.replace('_', ' ').title()}**: {value}")
        
        st.markdown("""
        ### Adaptation Strategies
        
        - Implementing mechanical stimulation to simulate gravitational cues
        - Using specialized growth media with higher porosity
        - Adjusting water delivery systems for more uniform distribution
        - Providing directional light to guide plant orientation
        """)
    
    with domain_tabs[1]:
        st.subheader("Radiation Impact on Plants in Space")
        st.markdown("""
        ### Research Findings
        
        Space radiation presents unique challenges for plant growth:
        
        - **DNA Damage**: Cosmic radiation and solar particles can cause genetic mutations and DNA strand breaks.
        - **Oxidative Stress**: Radiation increases reactive oxygen species (ROS) in plant tissues.
        - **Growth Rate Reduction**: Prolonged radiation exposure typically reduces overall growth rates.
        - **Seed Viability**: Radiation can reduce germination rates in subsequent generations.
        
        ### Key Parameters
        """)
        
        # Extract and display optimal parameters
        rad_params = knowledge_base.get_optimal_parameters("radiation")
        if rad_params:
            for param, value in rad_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    st.info(f"**{param.replace('_', ' ').title()}**: Optimal range is {value[0]} to {value[1]}")
                else:
                    st.info(f"**{param.replace('_', ' ').title()}**: {value}")
        
        st.markdown("""
        ### Protection Strategies
        
        - Using radiation shielding materials around growing chambers
        - Selecting radiation-resistant crop varieties
        - Incorporating antioxidant-rich nutrients in growth media
        - Implementing timed growth cycles to avoid solar event periods
        """)
    
    with domain_tabs[2]:
        st.subheader("Light Optimization for Space Agriculture")
        st.markdown("""
        ### Research Findings
        
        Light is a critical factor for plant growth in space environments:
        
        - **Spectrum Optimization**: Different light wavelengths affect various plant processes - blue light (400-500nm) influences vegetative growth, while red light (600-700nm) affects flowering.
        - **Intensity Requirements**: Light intensity needs vary significantly between growth stages and species.
        - **Photoperiod Effects**: The duration of light/dark cycles impacts flowering time and metabolic processes.
        - **Energy Efficiency**: LED technology provides the most efficient spectral control while minimizing heat generation.
        
        ### Key Parameters
        """)
        
        # Extract and display optimal parameters
        light_params = knowledge_base.get_optimal_parameters("lighting")
        if light_params:
            for param, value in light_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    st.info(f"**{param.replace('_', ' ').title()}**: Optimal range is {value[0]} to {value[1]}")
                else:
                    st.info(f"**{param.replace('_', ' ').title()}**: {value}")
        
        st.markdown("""
        ### Implementation Strategies
        
        - Using multi-channel LED arrays with adjustable spectral output
        - Implementing dynamic lighting schedules that change with growth stages
        - Integrating light intensity sensors for feedback-controlled systems
        - Positioning lights to maximize uniform distribution and penetration
        """)
    
    with domain_tabs[3]:
        st.subheader("Nutrient Delivery Systems")
        st.markdown("""
        ### Research Findings
        
        Nutrient delivery in space requires specialized approaches:
        
        - **Hydroponic/Aeroponic Systems**: Soilless growing systems are preferred for their efficiency and reduced mass.
        - **Nutrient Recycling**: Closed-loop systems that capture and reuse nutrients are essential for sustainability.
        - **Micronutrient Bioavailability**: Zero-gravity affects nutrient uptake rates and ion exchange dynamics.
        - **pH Stability**: Maintaining stable pH is more challenging in closed systems with limited buffering capacity.
        
        ### Key Parameters
        """)
        
        # Extract and display optimal parameters
        nutrient_params = knowledge_base.get_optimal_parameters("nutrients")
        if nutrient_params:
            for param, value in nutrient_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    st.info(f"**{param.replace('_', ' ').title()}**: Optimal range is {value[0]} to {value[1]}")
                else:
                    st.info(f"**{param.replace('_', ' ').title()}**: {value}")
        
        st.markdown("""
        ### Implementation Strategies
        
        - Using ion-specific sensors for real-time nutrient concentration monitoring
        - Implementing automated dosing systems with feedback control
        - Designing nutrient delivery schedules based on growth stages
        - Incorporating microbial components for enhanced nutrient availability
        """)
    
    with domain_tabs[4]:
        st.subheader("Environmental Control Systems")
        st.markdown("""
        ### Research Findings
        
        Environmental parameters must be precisely controlled for optimal growth:
        
        - **Temperature Management**: Thermal control affects metabolic rates and water usage efficiency.
        - **Humidity Control**: Vapor pressure deficit management is critical for transpiration and nutrient uptake.
        - **Air Circulation**: Proper ventilation prevents ethylene buildup and promotes gas exchange.
        - **CO₂ Enrichment**: Higher CO₂ concentrations can enhance photosynthetic efficiency within limits.
        
        ### Key Parameters
        """)
        
        # Extract and display optimal parameters
        env_params = knowledge_base.get_optimal_parameters("environment")
        if env_params:
            for param, value in env_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    st.info(f"**{param.replace('_', ' ').title()}**: Optimal range is {value[0]} to {value[1]}")
                else:
                    st.info(f"**{param.replace('_', ' ').title()}**: {value}")
        
        st.markdown("""
        ### Implementation Strategies
        
        - Designing multi-zone growing environments with parameter gradients
        - Implementing predictive control algorithms to anticipate environmental changes
        - Using integrated sensor networks for comprehensive monitoring
        - Creating redundant control systems for critical parameters
        """)
    
    with domain_tabs[5]:
        st.subheader("Crop Selection for Space Agriculture")
        st.markdown("""
        ### Research Findings
        
        Crop selection criteria for space agriculture include:
        
        - **Resource Efficiency**: Plants with high harvest index and rapid growth cycles are preferred.
        - **Nutritional Density**: Crops with high caloric and micronutrient content per growing area.
        - **Environmental Resilience**: Species that tolerate fluctuations in growing conditions.
        - **Multi-functionality**: Plants that provide multiple benefits (food, oxygen, psychological well-being).
        
        ### Recommended Crops
        """)
        
        # Extract and display optimal parameters
        crop_params = knowledge_base.get_optimal_parameters("crops")
        if crop_params:
            for param, value in crop_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    st.info(f"**{param.replace('_', ' ').title()}**: Optimal range is {value[0]} to {value[1]}")
                else:
                    st.info(f"**{param.replace('_', ' ').title()}**: {value}")
        
        st.markdown("""
        ### Selection Strategies
        
        - Implementing growth trials with multiple varieties to identify optimal performers
        - Using staggered planting schedules for continuous harvest
        - Combining complementary crop species in mixed growing systems
        - Selecting compact growth habits for space efficiency
        """)
    
    # Citations and research sources
    st.subheader("Research Sources")
    st.markdown("""
    1. NASA Technical Reports: "Plant Growth and Development in Space" (NASA-TM-2023-0001)
    2. Journal of Space Agriculture, Vol. 5: "Optimizing Light Recipes for Microgravity Plant Growth"
    3. Advances in Space Research: "Radiation Protection Strategies for Biological Systems in Space"
    4. International Space Station Results: "Veggie and Advanced Plant Habitat Experiments"
    5. Frontiers in Plant Science: "Nutrient Delivery Systems for Microgravity Agriculture"
    
    These research findings have been incorporated into our reinforcement learning system to enhance
    the agent's understanding of optimal growing conditions in space environments.
    """)
    
    # Integration with RL agent explanation
    st.subheader("Integration with Reinforcement Learning")
    st.markdown("""
    The research knowledge base influences the RL agent's decision-making in several ways:
    
    1. **Reward Function Enhancement**: Scientific findings modify the reward calculations to better align with known optimal conditions.
    2. **Parameter Optimization**: Research-validated parameter ranges guide exploration boundaries.
    3. **Action Constraints**: Some actions are constrained based on known detrimental combinations from research.
    4. **State Evaluation**: The agent evaluates environmental states using metrics derived from research findings.
    5. **Multi-objective Optimization**: Scientific knowledge helps balance multiple competing objectives (growth rate, nutritional value, resource usage).
    
    This integration creates a scientifically-informed AI system that combines machine learning adaptability
    with established research knowledge.
    """)
    
    # Show knowledge base influence on selected species
    if st.session_state.language == 'English':
        st.subheader(f"Research-Based Recommendations for {selected_species}")
    else:
        st.subheader(f"التوصيات المستندة إلى البحث لنبات {selected_species}")
        
    recommendations = knowledge_base.generate_research_based_recommendations({}, selected_species)
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            if st.session_state.language == 'English':
                st.write(f"**Recommendation {i+1}:** {rec.get('description', 'No description')}")
                if 'rationale' in rec:
                    st.write(f"*Rationale:* {rec['rationale']}")
            else:
                st.write(f"**التوصية {i+1}:** {rec.get('description', 'لا يوجد وصف')}")
                if 'rationale' in rec:
                    st.write(f"*السبب:* {rec['rationale']}")
            st.write("---")

# Add author credits at the bottom of the page
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; background-color: #f0f8f0; padding: 20px; border-radius: 10px; margin-top: 30px;">
    <h3 style="color: #2E8B57;">Done by:</h3>
    <p style="font-size: 18px; font-weight: bold;">
        Fatima Majed<br>
        Shaikha Rashed<br>
        Maitha Ali
    </p>
</div>
""", unsafe_allow_html=True)
