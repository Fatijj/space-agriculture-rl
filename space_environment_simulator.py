"""
Space Environment Simulator for Agriculture
Implementation of advanced space environment effects on plant growth
"""

import numpy as np
import logging
import random
from datetime import datetime, timedelta

logger = logging.getLogger('SpaceAgriRL.SpaceEnvironment')

class SpaceEnvironmentSimulator:
    """Simulator for space environment effects on plant growth"""
    
    def __init__(self, seed=None):
        """
        Initialize the space environment simulator
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Space Weather Impact Parameters
        self.solar_cycle_periods = {
            'solar_minimum': (0.2, 1.5),   # Low radiation impact
            'solar_maximum': (1.5, 3.0),   # High radiation impact
            'solar_transition': (0.8, 2.0)  # Moderate radiation impact
        }
        
        # Cosmic Ray Intensity Mapping
        self.cosmic_ray_intensity = {
            'low': (0.1, 0.5),    # Minimal DNA damage
            'moderate': (0.5, 1.0),  # Some cellular stress
            'high': (1.0, 2.0)    # Significant genetic disruption
        }
        
        # Microgravity Adaptation Factors
        self.microgravity_adaptation = {
            'Dwarf Wheat': 0.8,
            'Lettuce': 0.9,
            'Cherry Tomato': 0.7,
            'Microgreens': 0.95,
            'Space Potato': 0.75,
            'Radish': 0.85,
            'Spinach': 0.88,
            'Space Basil': 0.92
        }
        
        # Initialize current space weather conditions
        self.current_conditions = {
            'solar_cycle_phase': 'solar_minimum',
            'cosmic_ray_intensity': 'low',
            'solar_flare_active': False,
            'solar_flare_data': None,
            'current_day': 0,
            'radiation_history': [],
            'flare_history': []
        }
        
        # Solar cycle starts at random phase (0-11 years in cycle)
        cycle_start = random.randint(0, 4015)  # days (11 years * 365 days)
        self.current_conditions['cycle_day'] = cycle_start
        self._update_solar_cycle_phase()
    
    def advance_time(self, days=1):
        """
        Advance simulator time and update space weather conditions
        
        Args:
            days: Number of days to advance
            
        Returns:
            Dictionary with updated space weather conditions
        """
        # Update current day
        self.current_conditions['current_day'] += days
        
        # Update solar cycle day and phase
        self.current_conditions['cycle_day'] = (self.current_conditions['cycle_day'] + days) % 4015  # 11-year cycle
        self._update_solar_cycle_phase()
        
        # Check for solar flare occurrence
        self._check_solar_flare()
        
        # Update cosmic ray intensity (changes more slowly)
        if random.random() < 0.1 * days:  # 10% chance per day to change
            self._update_cosmic_ray_intensity()
        
        # Update radiation levels based on current conditions
        self._update_radiation_levels()
        
        # Return current conditions
        return self.current_conditions.copy()
    
    def _update_solar_cycle_phase(self):
        """Update solar cycle phase based on cycle day"""
        cycle_day = self.current_conditions['cycle_day']
        
        # Solar maximum occurs around years 4-7 of the 11-year cycle
        if 1460 <= cycle_day <= 2555:  # days 4*365 to 7*365
            self.current_conditions['solar_cycle_phase'] = 'solar_maximum'
        elif (1095 <= cycle_day < 1460) or (2555 < cycle_day <= 2920):  # 1 year transition on either side
            self.current_conditions['solar_cycle_phase'] = 'solar_transition'
        else:
            self.current_conditions['solar_cycle_phase'] = 'solar_minimum'
    
    def _check_solar_flare(self):
        """Check for solar flare occurrence and update conditions"""
        # Solar flare probability depends on solar cycle phase
        if self.current_conditions['solar_cycle_phase'] == 'solar_maximum':
            flare_probability = 0.15  # Higher during solar maximum
        elif self.current_conditions['solar_cycle_phase'] == 'solar_transition':
            flare_probability = 0.08  # Moderate during transition
        else:
            flare_probability = 0.03  # Lower during solar minimum
        
        # If solar flare was active, check if it ends
        if self.current_conditions['solar_flare_active']:
            # Get current flare data
            flare_data = self.current_conditions['solar_flare_data']
            
            # Check if flare has ended
            if flare_data['current_hour'] >= flare_data['duration_hours']:
                # Flare has ended
                self.current_conditions['solar_flare_active'] = False
                self.current_conditions['solar_flare_data'] = None
                
                # Add to history
                self.current_conditions['flare_history'].append({
                    'day': self.current_conditions['current_day'],
                    'class': flare_data['class'],
                    'duration_hours': flare_data['duration_hours'],
                    'max_intensity': flare_data['max_intensity']
                })
                
                # Keep history manageable
                if len(self.current_conditions['flare_history']) > 20:
                    self.current_conditions['flare_history'] = self.current_conditions['flare_history'][-20:]
            else:
                # Update flare hour
                flare_data['current_hour'] += 24  # Advance 24 hours
                self.current_conditions['solar_flare_data'] = flare_data
        
        # Check for new flare if none active
        elif random.random() < flare_probability:
            # Generate a new solar flare
            self.current_conditions['solar_flare_active'] = True
            self.current_conditions['solar_flare_data'] = self.generate_solar_flare_event()
    
    def _update_cosmic_ray_intensity(self):
        """Update cosmic ray intensity"""
        # Cosmic ray intensity is anticorrelated with solar activity
        # Higher solar activity (solar maximum) = lower cosmic rays due to stronger solar magnetic field
        
        if self.current_conditions['solar_cycle_phase'] == 'solar_maximum':
            # During solar maximum, cosmic rays are lower
            probabilities = {'low': 0.7, 'moderate': 0.25, 'high': 0.05}
        elif self.current_conditions['solar_cycle_phase'] == 'solar_transition':
            # During transition, more variability
            probabilities = {'low': 0.4, 'moderate': 0.4, 'high': 0.2}
        else:
            # During solar minimum, cosmic rays are higher
            probabilities = {'low': 0.2, 'moderate': 0.5, 'high': 0.3}
        
        # Select intensity based on probabilities
        intensity_levels = list(probabilities.keys())
        weights = [probabilities[level] for level in intensity_levels]
        self.current_conditions['cosmic_ray_intensity'] = random.choices(intensity_levels, weights=weights)[0]
    
    def _update_radiation_levels(self):
        """Update radiation levels based on current space weather conditions"""
        # Base radiation level depends on cosmic ray intensity
        cosmic_intensity = self.cosmic_ray_intensity[self.current_conditions['cosmic_ray_intensity']]
        base_radiation = random.uniform(cosmic_intensity[0], cosmic_intensity[1]) * 10  # Scale to 0-20 range
        
        # Add solar flare contribution if active
        solar_flare_contribution = 0
        if self.current_conditions['solar_flare_active']:
            flare_data = self.current_conditions['solar_flare_data']
            solar_flare_contribution = flare_data['current_intensity'] * 15  # Scale to 0-75 range for X-class flares
        
        # Total radiation level
        radiation_level = base_radiation + solar_flare_contribution
        
        # Record in history
        self.current_conditions['radiation_history'].append({
            'day': self.current_conditions['current_day'],
            'cosmic_ray': base_radiation,
            'solar_flare': solar_flare_contribution,
            'total': radiation_level
        })
        
        # Keep history manageable
        if len(self.current_conditions['radiation_history']) > 100:
            self.current_conditions['radiation_history'] = self.current_conditions['radiation_history'][-100:]
    
    def generate_solar_flare_event(self, duration_hours=None):
        """
        Simulate a solar flare event with varying intensity
        
        Args:
            duration_hours: Optional duration in hours, if None, will be randomly determined
            
        Returns:
            Dictionary with solar flare event data
        """
        # Solar flare characteristics
        flare_classes = {
            'A': (0.1, 0.3),   # Minimal impact
            'B': (0.3, 0.5),   # Low impact
            'C': (0.5, 1.0),   # Moderate impact
            'M': (1.0, 2.0),   # Strong impact
            'X': (2.0, 5.0)    # Extreme impact
        }
        
        # Determine flare duration if not specified
        if duration_hours is None:
            # Duration based on class (larger flares last longer)
            class_durations = {
                'A': (1, 6),
                'B': (2, 12),
                'C': (4, 24),
                'M': (8, 36),
                'X': (12, 48)
            }
            
            # Randomly select flare class with probabilities based on solar cycle
            if self.current_conditions['solar_cycle_phase'] == 'solar_maximum':
                class_probs = {'A': 0.2, 'B': 0.3, 'C': 0.3, 'M': 0.15, 'X': 0.05}
            elif self.current_conditions['solar_cycle_phase'] == 'solar_transition':
                class_probs = {'A': 0.3, 'B': 0.4, 'C': 0.2, 'M': 0.08, 'X': 0.02}
            else:
                class_probs = {'A': 0.5, 'B': 0.3, 'C': 0.15, 'M': 0.04, 'X': 0.01}
            
            class_names = list(class_probs.keys())
            class_weights = [class_probs[c] for c in class_names]
            class_name = random.choices(class_names, weights=class_weights)[0]
            
            # Determine duration
            min_duration, max_duration = class_durations[class_name]
            duration_hours = random.randint(min_duration, max_duration)
        else:
            # If duration is provided, select appropriate class
            if duration_hours <= 6:
                class_probs = {'A': 0.6, 'B': 0.3, 'C': 0.1, 'M': 0.0, 'X': 0.0}
            elif duration_hours <= 12:
                class_probs = {'A': 0.3, 'B': 0.5, 'C': 0.15, 'M': 0.05, 'X': 0.0}
            elif duration_hours <= 24:
                class_probs = {'A': 0.1, 'B': 0.3, 'C': 0.4, 'M': 0.15, 'X': 0.05}
            elif duration_hours <= 36:
                class_probs = {'A': 0.0, 'B': 0.1, 'C': 0.3, 'M': 0.5, 'X': 0.1}
            else:
                class_probs = {'A': 0.0, 'B': 0.0, 'C': 0.2, 'M': 0.5, 'X': 0.3}
            
            class_names = list(class_probs.keys())
            class_weights = [class_probs[c] for c in class_names]
            class_name = random.choices(class_names, weights=class_weights)[0]
        
        # Get intensity range for this class
        min_intensity, max_intensity = flare_classes[class_name]
        
        # Randomly determine peak intensity within class range
        peak_intensity = random.uniform(min_intensity, max_intensity)
        
        # Generate intensity profile over time
        # Typically rises quickly, peaks, then decays more slowly
        hours = list(range(duration_hours))
        rise_proportion = 0.3  # First 30% of duration is rise
        rise_end = int(duration_hours * rise_proportion)
        
        intensity_profile = []
        for hour in hours:
            if hour <= rise_end:
                # Rising phase - follows roughly half a sine wave
                progress = hour / rise_end
                intensity = peak_intensity * np.sin(progress * np.pi / 2)
            else:
                # Decay phase - exponential decay
                decay_progress = (hour - rise_end) / (duration_hours - rise_end)
                intensity = peak_intensity * np.exp(-decay_progress * 2)
            
            # Add random fluctuations
            fluctuation = random.uniform(-0.1, 0.1) * intensity
            intensity = max(0, intensity + fluctuation)
            
            intensity_profile.append(float(intensity))
        
        # Flare data
        flare_data = {
            'class': class_name,
            'duration_hours': duration_hours,
            'current_hour': 0,
            'peak_intensity': peak_intensity,
            'current_intensity': intensity_profile[0],
            'intensity_profile': intensity_profile,
            'hours': hours
        }
        
        return flare_data
    
    def simulate_cosmic_ray_impact(self, species, duration_days=30):
        """
        Simulate cosmic ray impact on plant genetic stability
        
        Args:
            species: Plant species name
            duration_days: Duration of simulation in days
            
        Returns:
            Dictionary with cosmic ray impact data
        """
        # Verify species exists in adaptation factors
        if species not in self.microgravity_adaptation:
            # Use average adaptation if species not found
            species_adaptation = 0.85
            logger.warning(f"Species '{species}' not found in microgravity adaptation data, using default value")
        else:
            species_adaptation = self.microgravity_adaptation[species]
        
        # Base cosmic ray intensity
        intensity_name = self.current_conditions['cosmic_ray_intensity']
        min_intensity, max_intensity = self.cosmic_ray_intensity[intensity_name]
        
        # Calculate cosmic ray damage probability over time
        damage_probability = []
        genetic_stability = []
        cumulative_damage = []
        
        # Start with initial damage probability
        current_damage_prob = np.random.weibull(2.0) * (1 - species_adaptation) * max_intensity * 0.1
        current_damage_prob = min(current_damage_prob, 0.05)  # Cap initial damage
        
        # Simulate over requested duration
        for day in range(duration_days):
            # Update cosmic ray intensity based on current conditions
            self.advance_time(1)
            intensity_name = self.current_conditions['cosmic_ray_intensity']
            min_intensity, max_intensity = self.cosmic_ray_intensity[intensity_name]
            
            # Factor in solar flare effects if active
            flare_factor = 1.0
            if self.current_conditions['solar_flare_active']:
                flare_data = self.current_conditions['solar_flare_data']
                flare_factor = 1.0 + flare_data['current_intensity']
            
            # Calculate damage probability for this day
            day_damage_prob = np.random.weibull(2.0) * (1 - species_adaptation) * max_intensity * 0.1 * flare_factor
            day_damage_prob = min(day_damage_prob, 0.1)  # Cap daily damage
            
            # Apply some persistence (damage probability changes gradually)
            current_damage_prob = current_damage_prob * 0.7 + day_damage_prob * 0.3
            
            # Add to history
            damage_probability.append(float(current_damage_prob))
            
            # Calculate cumulative damage
            if not cumulative_damage:
                cumulative_damage.append(current_damage_prob)
            else:
                # Damage accumulates but can also be repaired
                prev_damage = cumulative_damage[-1]
                new_damage = prev_damage + current_damage_prob - (prev_damage * species_adaptation * 0.1)
                new_damage = min(new_damage, 1.0)  # Cap at 100% damage
                cumulative_damage.append(float(new_damage))
            
            # Calculate genetic stability (inverse of cumulative damage)
            genetic_stability.append(float(1.0 - cumulative_damage[-1]))
        
        return {
            'species': species,
            'duration_days': duration_days,
            'cosmic_ray_intensity': intensity_name,
            'species_adaptation': species_adaptation,
            'damage_probability': damage_probability,
            'genetic_stability': genetic_stability,
            'cumulative_damage': cumulative_damage,
            'final_stability': genetic_stability[-1]
        }
    
    def microgravity_growth_model(self, species, baseline_growth, duration_days=30):
        """
        Simulate plant growth under microgravity conditions
        
        Args:
            species: Plant species name
            baseline_growth: Baseline growth rate in Earth gravity
            duration_days: Duration of simulation in days
            
        Returns:
            Dictionary with microgravity growth data
        """
        # Verify species exists in adaptation factors
        if species not in self.microgravity_adaptation:
            # Use average adaptation if species not found
            adaptation_factor = 0.85
            logger.warning(f"Species '{species}' not found in microgravity adaptation data, using default value")
        else:
            adaptation_factor = self.microgravity_adaptation[species]
        
        # Initialize growth arrays
        baseline_growth_array = []
        adjusted_growth_array = []
        cumulative_baseline = []
        cumulative_adjusted = []
        
        # Initial values
        cumulative_baseline_val = 0
        cumulative_adjusted_val = 0
        
        # Simulate over requested duration
        for day in range(duration_days):
            # Baseline growth has small random variations
            day_baseline = baseline_growth * (1 + np.random.normal(0, 0.05))
            
            # Get environment factors from space weather
            self.advance_time(1)
            
            # Factor in cosmic ray effects on growth
            cosmic_effect = 1.0
            if self.current_conditions['cosmic_ray_intensity'] == 'high':
                cosmic_effect = 0.9  # 10% reduction
            elif self.current_conditions['cosmic_ray_intensity'] == 'moderate':
                cosmic_effect = 0.95  # 5% reduction
            
            # Factor in solar flare effects on growth
            flare_effect = 1.0
            if self.current_conditions['solar_flare_active']:
                flare_data = self.current_conditions['solar_flare_data']
                if flare_data['class'] in ['M', 'X']:
                    flare_effect = 0.9  # 10% reduction for strong flares
                elif flare_data['class'] == 'C':
                    flare_effect = 0.95  # 5% reduction for moderate flares
            
            # Calculate microgravity-adjusted growth
            # Adaptation factor reduces the negative effects of microgravity
            microgravity_factor = adaptation_factor
            
            # Apply environment effects
            environment_factor = cosmic_effect * flare_effect
            
            # Calculate adjusted growth
            adjusted = day_baseline * microgravity_factor * environment_factor
            
            # Add to history
            baseline_growth_array.append(float(day_baseline))
            adjusted_growth_array.append(float(adjusted))
            
            # Update cumulative values
            cumulative_baseline_val += day_baseline
            cumulative_adjusted_val += adjusted
            
            cumulative_baseline.append(float(cumulative_baseline_val))
            cumulative_adjusted.append(float(cumulative_adjusted_val))
        
        return {
            'species': species,
            'duration_days': duration_days,
            'adaptation_factor': adaptation_factor,
            'baseline_growth': baseline_growth_array,
            'microgravity_adjusted_growth': adjusted_growth_array,
            'cumulative_baseline_growth': cumulative_baseline,
            'cumulative_adjusted_growth': cumulative_adjusted,
            'growth_ratio': cumulative_adjusted[-1] / cumulative_baseline[-1] if cumulative_baseline[-1] > 0 else 0
        }
    
    def apply_space_environment_effects(self, plant_state, species, days=1):
        """
        Apply space environment effects to plant state
        
        Args:
            plant_state: Current plant state dictionary
            species: Plant species name
            days: Number of days to simulate
            
        Returns:
            Updated plant state with space environment effects
        """
        # Make a copy of plant state to avoid modifying the original
        new_state = plant_state.copy()
        
        # Advance space weather simulation
        self.advance_time(days)
        
        # Get current space weather conditions
        cosmic_intensity = self.current_conditions['cosmic_ray_intensity']
        solar_flare_active = self.current_conditions['solar_flare_active']
        solar_flare_data = self.current_conditions['solar_flare_data']
        
        # Get species adaptation factor
        if species in self.microgravity_adaptation:
            adaptation_factor = self.microgravity_adaptation[species]
        else:
            adaptation_factor = 0.85  # Default
            logger.warning(f"Species '{species}' not found in microgravity adaptation data, using default value")
        
        # Calculate space environment effects
        
        # 1. Microgravity effects on growth
        # Microgravity generally reduces growth rate but depends on species adaptation
        if 'height' in new_state:
            growth_modifier = adaptation_factor * (1 + np.random.normal(0, 0.05))
            
            # Apply microgravity growth modifier to height changes
            height_change = new_state.get('height_change', 0)
            new_state['height_change'] = height_change * growth_modifier
            
            # Adjust height directly if no height_change parameter
            if height_change == 0 and new_state['height'] > 0:
                # Estimate daily growth - assuming 2% increase per day for mature plants
                estimated_growth = new_state['height'] * 0.02 * days
                # Apply microgravity modifier
                new_state['height'] += estimated_growth * growth_modifier
        
        # 2. Cosmic ray effects
        # Cosmic rays affect genetic stability and can reduce plant health
        if 'health_score' in new_state:
            cosmic_effect = 1.0
            if cosmic_intensity == 'high':
                cosmic_effect = 0.95  # 5% health reduction
            elif cosmic_intensity == 'moderate':
                cosmic_effect = 0.98  # 2% health reduction
            
            # Apply species adaptation as resistance to cosmic rays
            cosmic_effect = 1 - ((1 - cosmic_effect) * (1 - adaptation_factor * 0.5))
            new_state['health_score'] *= cosmic_effect
        
        # 3. Solar flare effects
        # Solar flares can cause acute radiation damage
        if solar_flare_active and 'health_score' in new_state:
            flare_class = solar_flare_data['class']
            flare_intensity = solar_flare_data['current_intensity']
            
            flare_effect = 1.0
            if flare_class == 'X':
                flare_effect = 0.9 - (flare_intensity * 0.02)  # Up to 10% health reduction for X-class
            elif flare_class == 'M':
                flare_effect = 0.95 - (flare_intensity * 0.01)  # Up to 5% health reduction for M-class
            elif flare_class == 'C':
                flare_effect = 0.98 - (flare_intensity * 0.005)  # Up to 2% health reduction for C-class
            
            # Apply species adaptation as resistance to flares
            flare_effect = 1 - ((1 - flare_effect) * (1 - adaptation_factor * 0.7))
            new_state['health_score'] *= flare_effect
        
        # 4. Apply effective radiation level to the environment
        if 'radiation_level' in new_state:
            # Get the latest radiation history entry
            if self.current_conditions['radiation_history']:
                latest_radiation = self.current_conditions['radiation_history'][-1]['total']
                # Scale to the appropriate range for the simulation
                # Assuming the default range is 0-100
                scaled_radiation = min(100, latest_radiation * 5)  # Scale factor might need adjustment
                # Update radiation level
                new_state['radiation_level'] = scaled_radiation
        
        # Ensure health score stays within valid range
        if 'health_score' in new_state:
            new_state['health_score'] = max(0.0, min(1.0, new_state['health_score']))
        
        return new_state
    
    def get_current_space_weather_summary(self):
        """
        Get human-readable summary of current space weather conditions
        
        Returns:
            Dictionary with space weather summary
        """
        # Get current conditions
        solar_phase = self.current_conditions['solar_cycle_phase']
        cosmic_intensity = self.current_conditions['cosmic_ray_intensity']
        solar_flare_active = self.current_conditions['solar_flare_active']
        
        # Solar cycle description
        if solar_phase == 'solar_maximum':
            solar_desc = "Solar Maximum - high solar activity with increased risk of solar flares and CMEs"
        elif solar_phase == 'solar_transition':
            solar_desc = "Solar Transition - moderate solar activity with variable space weather conditions"
        else:
            solar_desc = "Solar Minimum - low solar activity with reduced solar flare risk"
        
        # Cosmic ray description
        if cosmic_intensity == 'high':
            cosmic_desc = "High cosmic ray intensity - increased radiation risk to plants and equipment"
        elif cosmic_intensity == 'moderate':
            cosmic_desc = "Moderate cosmic ray intensity - some radiation effects on sensitive systems"
        else:
            cosmic_desc = "Low cosmic ray intensity - minimal radiation effects expected"
        
        # Solar flare description
        if solar_flare_active:
            flare_data = self.current_conditions['solar_flare_data']
            flare_desc = f"Active {flare_data['class']}-class solar flare in progress - increased radiation levels detected"
            
            # Additional warnings for strong flares
            if flare_data['class'] in ['M', 'X']:
                flare_desc += " - protective measures recommended for sensitive plants"
        else:
            flare_desc = "No active solar flares detected"
        
        # Get radiation history
        radiation_history = []
        for entry in self.current_conditions['radiation_history'][-7:]:  # Last week
            radiation_history.append({
                'day': entry['day'],
                'total': entry['total'],
                'level': 'High' if entry['total'] > 15 else 'Moderate' if entry['total'] > 8 else 'Low'
            })
        
        # Get recent flare history
        flare_history = self.current_conditions['flare_history'][-5:]  # Last 5 flares
        
        return {
            'day': self.current_conditions['current_day'],
            'solar_cycle': {
                'phase': solar_phase,
                'description': solar_desc,
                'cycle_day': self.current_conditions['cycle_day'],
                'cycle_progress': self.current_conditions['cycle_day'] / 4015
            },
            'cosmic_rays': {
                'intensity': cosmic_intensity,
                'description': cosmic_desc
            },
            'solar_flare': {
                'active': solar_flare_active,
                'description': flare_desc,
                'data': self.current_conditions['solar_flare_data'] if solar_flare_active else None
            },
            'radiation': {
                'current_level': self.current_conditions['radiation_history'][-1]['total'] if self.current_conditions['radiation_history'] else 0,
                'recent_history': radiation_history
            },
            'flare_history': flare_history,
            'recommendations': self._generate_weather_recommendations()
        }
    
    def _generate_weather_recommendations(self):
        """Generate recommendations based on current space weather conditions"""
        recommendations = []
        
        # Solar flare recommendations
        if self.current_conditions['solar_flare_active']:
            flare_data = self.current_conditions['solar_flare_data']
            
            if flare_data['class'] in ['M', 'X']:
                recommendations.append({
                    'priority': 'high',
                    'category': 'radiation_protection',
                    'action': "Increase radiation shielding to maximum levels until flare subsides"
                })
                recommendations.append({
                    'priority': 'high',
                    'category': 'monitoring',
                    'action': "Increase health monitoring frequency for radiation-sensitive plant species"
                })
            elif flare_data['class'] == 'C':
                recommendations.append({
                    'priority': 'medium',
                    'category': 'radiation_protection',
                    'action': "Consider temporary increase in radiation shielding"
                })
            else:
                recommendations.append({
                    'priority': 'low',
                    'category': 'monitoring',
                    'action': "Monitor radiation levels for any significant increases"
                })
        
        # Cosmic ray recommendations
        if self.current_conditions['cosmic_ray_intensity'] == 'high':
            recommendations.append({
                'priority': 'medium',
                'category': 'plant_protection',
                'action': "Provide additional antioxidant supplements to reduce cellular damage"
            })
            recommendations.append({
                'priority': 'medium',
                'category': 'growth_management',
                'action': "Adjust growth expectations downward due to cosmic ray stress"
            })
        elif self.current_conditions['cosmic_ray_intensity'] == 'moderate':
            recommendations.append({
                'priority': 'low',
                'category': 'monitoring',
                'action': "Monitor sensitive plant species for signs of radiation stress"
            })
        
        # Solar cycle recommendations
        if self.current_conditions['solar_cycle_phase'] == 'solar_maximum':
            recommendations.append({
                'priority': 'medium',
                'category': 'planning',
                'action': "Maintain heightened readiness for solar events over the next months"
            })
        elif self.current_conditions['solar_cycle_phase'] == 'solar_minimum':
            recommendations.append({
                'priority': 'low',
                'category': 'radiation_protection',
                'action': "Focus on cosmic ray protection rather than solar flare shielding"
            })
        
        return recommendations