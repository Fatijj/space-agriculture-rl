"""
Space Agriculture XAI (Explainable AI) System
Implementation of advanced explainability techniques for RL-based decision making
"""

import numpy as np
import logging
import json
from datetime import datetime
import random

logger = logging.getLogger('SpaceAgriRL.XAI')

class SpaceAgricultureXAI:
    """
    Explainable AI system for space agriculture decision-making
    """
    
    def __init__(self, ml_model=None, feature_names=None):
        """
        Initialize the XAI system
        
        Args:
            ml_model: The machine learning model to explain
            feature_names: List of feature names for the model inputs
        """
        self.model = ml_model
        self.features = feature_names or [
            'temperature', 'light_intensity', 'water_content', 'radiation_level',
            'co2_level', 'o2_level', 'humidity', 'nitrogen_level',
            'phosphorus_level', 'potassium_level', 'height', 'health_score'
        ]
        
        # Initialize feature importance (would be calculated from SHAP values in a real system)
        self.feature_importance = self._initialize_feature_importance()
        
        # Store explanation history
        self.explanation_history = []
    
    def _initialize_feature_importance(self):
        """Initialize placeholder feature importance"""
        # In a real system, this would be calculated using SHAP or other methods
        importance = {}
        # Assign importance values to all features (random for simulation)
        for feature in self.features:
            importance[feature] = random.uniform(0, 1)
        
        # Normalize to sum to 1.0
        total = sum(importance.values())
        for feature in importance:
            importance[feature] /= total
        
        return importance
    
    def generate_global_explanation(self, dataset=None):
        """
        Create global model interpretability report
        
        Args:
            dataset: Dataset to use for generating explanations
            
        Returns:
            Dictionary with global explanation information
        """
        # In a real system, dataset would be used to calculate SHAP values
        # For demonstration, we'll use the pre-initialized values
        
        # Rank features by importance
        feature_importance = self.rank_feature_importance()
        
        # Detect feature interactions
        interactions = self.detect_feature_interactions()
        
        # Generate decision boundaries visualization data
        decision_boundaries = self.visualize_decision_boundaries()
        
        # Combine into explanation report
        explanation_report = {
            'feature_importance': feature_importance,
            'interaction_effects': interactions,
            'model_decision_boundaries': decision_boundaries,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.explanation_history.append({
            'type': 'global',
            'explanation': explanation_report
        })
        
        return explanation_report
    
    def rank_feature_importance(self):
        """
        Rank features by importance
        
        Returns:
            Dictionary of features sorted by importance
        """
        # Sort features by importance value
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to dictionary for easy access
        ranked_importance = {
            feature: {
                'importance': importance,
                'rank': i + 1,
                'percentage': importance * 100
            }
            for i, (feature, importance) in enumerate(sorted_features)
        }
        
        return ranked_importance
    
    def detect_feature_interactions(self):
        """
        Detect interactions between features
        
        Returns:
            Dictionary of detected feature interactions
        """
        # In a real system, this would use SHAP interaction values
        # For demonstration, we'll create simulated interactions
        
        # List of potential interactions to detect
        potential_interactions = [
            ('temperature', 'humidity'),
            ('light_intensity', 'temperature'),
            ('water_content', 'humidity'),
            ('nitrogen_level', 'phosphorus_level'),
            ('co2_level', 'light_intensity')
        ]
        
        # Generate interaction strengths
        interactions = {}
        for feature1, feature2 in potential_interactions:
            if feature1 in self.features and feature2 in self.features:
                # Random interaction strength (would be calculated in real system)
                strength = random.uniform(0, 1) ** 2  # Square to make strong interactions less common
                
                if strength > 0.1:  # Only include non-trivial interactions
                    interaction_name = f"{feature1}_{feature2}"
                    interactions[interaction_name] = {
                        'features': [feature1, feature2],
                        'strength': strength,
                        'effect': self._describe_interaction_effect(feature1, feature2, strength)
                    }
        
        return interactions
    
    def _describe_interaction_effect(self, feature1, feature2, strength):
        """Generate description for an interaction effect"""
        # Known interaction effects
        if (feature1 == 'temperature' and feature2 == 'humidity') or (feature2 == 'temperature' and feature1 == 'humidity'):
            return "Temperature and humidity interact to affect plant water requirements"
        
        elif (feature1 == 'light_intensity' and feature2 == 'temperature') or (feature2 == 'light_intensity' and feature1 == 'temperature'):
            return "Light intensity and temperature together impact photosynthesis efficiency"
        
        elif (feature1 == 'water_content' and feature2 == 'humidity') or (feature2 == 'water_content' and feature1 == 'humidity'):
            return "Water content and humidity jointly influence transpiration rates"
        
        elif (feature1 == 'nitrogen_level' and feature2 == 'phosphorus_level') or (feature2 == 'nitrogen_level' and feature1 == 'phosphorus_level'):
            return "Nitrogen and phosphorus levels have synergistic effects on growth"
        
        elif (feature1 == 'co2_level' and feature2 == 'light_intensity') or (feature2 == 'co2_level' and feature1 == 'light_intensity'):
            return "CO2 and light intensity interact to determine photosynthesis rate"
        
        # Generic description for other interactions
        return f"Interaction between {feature1} and {feature2} affects plant growth outcomes"
    
    def visualize_decision_boundaries(self):
        """
        Generate data for visualizing decision boundaries
        
        Returns:
            Dictionary with decision boundary visualization data
        """
        # In a real system, this would use the model to predict outcomes across different parameter ranges
        # For demonstration, we'll create sample decision regions
        
        # Choose two most important features for visualization
        top_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        feature1, importance1 = top_features[0]
        feature2, importance2 = top_features[1]
        
        # Generate grid of values for the two features
        grid_size = 20
        
        # Determine feature ranges (would come from data in real system)
        ranges = {
            'temperature': (15, 35),
            'light_intensity': (500, 1500),
            'water_content': (40, 90),
            'radiation_level': (0, 50),
            'co2_level': (400, 1600),
            'o2_level': (15, 25),
            'humidity': (40, 90),
            'nitrogen_level': (40, 90),
            'phosphorus_level': (40, 90),
            'potassium_level': (40, 90),
            'height': (0, 50),
            'health_score': (0, 1)
        }
        
        # Create grid of values
        feature1_values = np.linspace(
            ranges.get(feature1, (0, 1))[0],
            ranges.get(feature1, (0, 1))[1],
            grid_size
        )
        
        feature2_values = np.linspace(
            ranges.get(feature2, (0, 1))[0],
            ranges.get(feature2, (0, 1))[1],
            grid_size
        )
        
        # Initialize decision region matrix (would be model predictions in real system)
        decision_regions = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                # Simulate a prediction (0-1 value)
                # In a real system this would be model.predict([feature_values])
                # Here we create a pattern based on the feature values
                f1_norm = (feature1_values[i] - ranges.get(feature1, (0, 1))[0]) / (ranges.get(feature1, (0, 1))[1] - ranges.get(feature1, (0, 1))[0])
                f2_norm = (feature2_values[j] - ranges.get(feature2, (0, 1))[0]) / (ranges.get(feature2, (0, 1))[1] - ranges.get(feature2, (0, 1))[0])
                
                # Create a pattern (e.g., quadratic function)
                prediction = 0.5 + 0.5 * (f1_norm - 0.5) ** 2 - 0.3 * (f2_norm - 0.5) ** 2
                prediction = max(0, min(1, prediction + random.uniform(-0.05, 0.05)))
                
                row.append(float(prediction))
            decision_regions.append(row)
        
        # Create visualization data
        visualization_data = {
            'features': [feature1, feature2],
            'feature1_values': feature1_values.tolist(),
            'feature2_values': feature2_values.tolist(),
            'decision_regions': decision_regions,
            'importance': {
                feature1: importance1,
                feature2: importance2
            }
        }
        
        return visualization_data
    
    def explain_individual_prediction(self, sample):
        """
        Generate detailed explanation for a single prediction
        
        Args:
            sample: Input sample to explain
            
        Returns:
            Dictionary with explanation for the prediction
        """
        # Verify sample format
        if isinstance(sample, np.ndarray):
            # Convert numpy array to dictionary
            if len(sample) == len(self.features):
                sample_dict = {feature: float(value) for feature, value in zip(self.features, sample)}
            else:
                raise ValueError(f"Sample length {len(sample)} doesn't match feature count {len(self.features)}")
        elif isinstance(sample, dict):
            sample_dict = sample
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}")
        
        # Generate force plot data
        force_plot = self.generate_force_plot(sample_dict)
        
        # Generate counterfactual scenarios
        counterfactuals = self.generate_counterfactuals(sample_dict)
        
        # Break down prediction components
        prediction_breakdown = self.breakdown_prediction_components(sample_dict)
        
        # Combine into local explanation
        local_explanation = {
            'sample': sample_dict,
            'shap_force_plot': force_plot,
            'counterfactual_scenarios': counterfactuals,
            'prediction_breakdown': prediction_breakdown,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to history
        self.explanation_history.append({
            'type': 'local',
            'explanation': local_explanation
        })
        
        return local_explanation
    
    def generate_force_plot(self, sample):
        """
        Generate SHAP force plot data for a single prediction
        
        Args:
            sample: Dictionary with sample feature values
            
        Returns:
            Dictionary with force plot data
        """
        # This would use SHAP values in a real system
        # For demonstration, we'll simulate feature contributions
        
        # Check for missing features
        for feature in self.features:
            if feature not in sample:
                sample[feature] = 0.0
        
        # Calculate baseline prediction (average model output)
        baseline = 0.5  # Simulated baseline (would be calculated from data)
        
        # Calculate contribution for each feature
        contributions = {}
        for feature in self.features:
            # Simulate contribution based on feature value and importance
            # In real system this would come from SHAP library
            
            # Determine feature range
            ranges = {
                'temperature': (15, 35),
                'light_intensity': (500, 1500),
                'water_content': (40, 90),
                'radiation_level': (0, 50),
                'co2_level': (400, 1600),
                'o2_level': (15, 25),
                'humidity': (40, 90),
                'nitrogen_level': (40, 90),
                'phosphorus_level': (40, 90),
                'potassium_level': (40, 90),
                'height': (0, 50),
                'health_score': (0, 1)
            }
            
            feature_range = ranges.get(feature, (0, 1))
            
            # Normalize feature value to 0-1 scale
            if feature_range[1] - feature_range[0] > 0:
                normalized_value = (sample[feature] - feature_range[0]) / (feature_range[1] - feature_range[0])
            else:
                normalized_value = 0.5
            
            # Calculate deviation from optimal (0.5)
            deviation = normalized_value - 0.5
            
            # Apply feature importance to scale contribution
            contribution = deviation * self.feature_importance.get(feature, 0) * 2.0
            
            # Add to contributions dictionary
            contributions[feature] = contribution
        
        # Calculate final prediction
        prediction = baseline + sum(contributions.values())
        prediction = max(0, min(1, prediction))  # Clip to 0-1 range
        
        # Sort contributions by absolute magnitude
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Determine feature impact directions
        positive_features = []
        negative_features = []
        
        for feature, contribution in sorted_contributions:
            if contribution > 0.01:  # Only include non-trivial contributions
                positive_features.append({
                    'feature': feature,
                    'contribution': contribution
                })
            elif contribution < -0.01:
                negative_features.append({
                    'feature': feature,
                    'contribution': contribution
                })
        
        # Create force plot data
        force_plot = {
            'baseline': baseline,
            'prediction': prediction,
            'contributions': contributions,
            'positive_features': positive_features,
            'negative_features': negative_features
        }
        
        return force_plot
    
    def generate_counterfactuals(self, sample):
        """
        Generate counterfactual explanations for a sample
        
        Args:
            sample: Dictionary with sample feature values
            
        Returns:
            List of counterfactual scenarios
        """
        # Identify most important features for counterfactuals
        force_plot = self.generate_force_plot(sample)
        contributions = force_plot['contributions']
        
        # Sort features by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate counterfactuals for the top features
        counterfactuals = []
        for feature, contribution in sorted_features[:3]:  # Top 3 features
            if abs(contribution) > 0.01:  # Only for significant contributions
                # Create a counterfactual by adjusting this feature
                counterfactual = sample.copy()
                
                # Determine optimal adjustment direction
                if contribution < 0:
                    # Negative contribution, need to increase
                    adjustment_direction = "increase"
                    
                    # Ranges for different features
                    ranges = {
                        'temperature': (15, 35),
                        'light_intensity': (500, 1500),
                        'water_content': (40, 90),
                        'radiation_level': (0, 50),
                        'co2_level': (400, 1600),
                        'o2_level': (15, 25),
                        'humidity': (40, 90),
                        'nitrogen_level': (40, 90),
                        'phosphorus_level': (40, 90),
                        'potassium_level': (40, 90),
                        'height': (0, 50),
                        'health_score': (0, 1)
                    }
                    
                    feature_range = ranges.get(feature, (0, 1))
                    current = sample[feature]
                    
                    # Calculate new optimal value (move 50% toward upper bound)
                    new_value = current + (feature_range[1] - current) * 0.5
                    
                    # Special case for radiation which is better when lower
                    if feature == 'radiation_level':
                        adjustment_direction = "decrease"
                        new_value = current - (current - feature_range[0]) * 0.5
                    
                    counterfactual[feature] = new_value
                    
                else:
                    # Positive contribution, or need to decrease
                    adjustment_direction = "decrease"
                    
                    # Ranges for different features
                    ranges = {
                        'temperature': (15, 35),
                        'light_intensity': (500, 1500),
                        'water_content': (40, 90),
                        'radiation_level': (0, 50),
                        'co2_level': (400, 1600),
                        'o2_level': (15, 25),
                        'humidity': (40, 90),
                        'nitrogen_level': (40, 90),
                        'phosphorus_level': (40, 90),
                        'potassium_level': (40, 90),
                        'height': (0, 50),
                        'health_score': (0, 1)
                    }
                    
                    feature_range = ranges.get(feature, (0, 1))
                    current = sample[feature]
                    
                    # Calculate new optimal value (move 50% toward lower bound)
                    new_value = current - (current - feature_range[0]) * 0.5
                    
                    # Special case for radiation which is better when lower
                    if feature == 'radiation_level':
                        adjustment_direction = "increase"
                        new_value = current + (feature_range[1] - current) * 0.5
                    
                    counterfactual[feature] = new_value
                
                # Calculate expected outcome for counterfactual
                cf_force_plot = self.generate_force_plot(counterfactual)
                cf_prediction = cf_force_plot['prediction']
                
                # Calculate improvement
                improvement = cf_prediction - force_plot['prediction']
                
                counterfactuals.append({
                    'feature': feature,
                    'current_value': sample[feature],
                    'suggested_value': counterfactual[feature],
                    'adjustment_direction': adjustment_direction,
                    'expected_improvement': improvement,
                    'explanation': self._generate_counterfactual_explanation(feature, adjustment_direction, improvement)
                })
        
        return counterfactuals
    
    def _generate_counterfactual_explanation(self, feature, direction, improvement):
        """Generate natural language explanation for a counterfactual"""
        # Generate explanation based on feature and direction
        if improvement > 0:
            magnitude = "significantly" if improvement > 0.1 else "slightly"
            
            if feature == 'temperature':
                if direction == 'increase':
                    return f"{magnitude.capitalize()} increasing the temperature would improve plant health"
                else:
                    return f"{magnitude.capitalize()} decreasing the temperature would improve plant health"
            
            elif feature == 'light_intensity':
                if direction == 'increase':
                    return f"{magnitude.capitalize()} increasing the light intensity would improve photosynthesis"
                else:
                    return f"{magnitude.capitalize()} decreasing the light intensity would prevent light stress"
            
            elif feature == 'water_content':
                if direction == 'increase':
                    return f"{magnitude.capitalize()} increasing the water content would reduce drought stress"
                else:
                    return f"{magnitude.capitalize()} decreasing the water content would prevent overwatering"
            
            elif feature == 'radiation_level':
                if direction == 'decrease':
                    return f"{magnitude.capitalize()} reducing radiation exposure would improve plant health"
                else:
                    return f"{magnitude.capitalize()} increasing radiation exposure would stimulate adaptive responses"
            
            elif 'level' in feature:
                if direction == 'increase':
                    return f"{magnitude.capitalize()} increasing {feature.replace('_level', '')} levels would address deficiency"
                else:
                    return f"{magnitude.capitalize()} decreasing {feature.replace('_level', '')} levels would prevent toxicity"
            
            else:
                if direction == 'increase':
                    return f"{magnitude.capitalize()} increasing {feature.replace('_', ' ')} would improve outcomes"
                else:
                    return f"{magnitude.capitalize()} decreasing {feature.replace('_', ' ')} would improve outcomes"
        else:
            return f"Adjusting {feature.replace('_', ' ')} would not significantly improve outcomes"
    
    def breakdown_prediction_components(self, sample):
        """
        Break down prediction into component contributions
        
        Args:
            sample: Dictionary with sample feature values
            
        Returns:
            Dictionary with prediction breakdown components
        """
        # This uses the force plot data to create a more structured breakdown
        force_plot = self.generate_force_plot(sample)
        
        # Group contributions by category
        categories = {
            'environmental': ['temperature', 'light_intensity', 'humidity', 'co2_level', 'o2_level'],
            'water_management': ['water_content'],
            'nutrients': ['nitrogen_level', 'phosphorus_level', 'potassium_level'],
            'stress_factors': ['radiation_level'],
            'plant_status': ['height', 'health_score']
        }
        
        # Calculate category contributions
        category_contributions = {}
        for category, features in categories.items():
            # Sum contributions for features in this category
            contribution = sum(force_plot['contributions'].get(feature, 0) for feature in features if feature in force_plot['contributions'])
            
            category_contributions[category] = {
                'contribution': contribution,
                'percentage': abs(contribution) / sum(abs(c) for c in force_plot['contributions'].values()) * 100 if sum(force_plot['contributions'].values()) != 0 else 0,
                'features': {feature: force_plot['contributions'].get(feature, 0) for feature in features if feature in force_plot['contributions']}
            }
        
        # Calculate overall prediction components
        baseline = force_plot['baseline']
        prediction = force_plot['prediction']
        
        # Create breakdown
        breakdown = {
            'baseline': baseline,
            'prediction': prediction,
            'category_contributions': category_contributions,
            'key_drivers': self._identify_key_drivers(force_plot)
        }
        
        return breakdown
    
    def _identify_key_drivers(self, force_plot):
        """Identify key drivers of the prediction"""
        # Extract contributions
        contributions = force_plot['contributions']
        
        # Sort by absolute magnitude
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Take top contributors
        key_drivers = []
        for feature, contribution in sorted_contributions[:3]:  # Top 3 features
            if abs(contribution) > 0.01:  # Only for significant contributions
                driver = {
                    'feature': feature,
                    'contribution': contribution,
                    'direction': 'positive' if contribution > 0 else 'negative',
                    'impact': 'high' if abs(contribution) > 0.1 else 'medium' if abs(contribution) > 0.05 else 'low'
                }
                key_drivers.append(driver)
        
        return key_drivers
    
    def create_astronaut_friendly_report(self, explanation, role="botanist"):
        """
        Transform technical explanations into human-readable format
        
        Args:
            explanation: Technical explanation dictionary
            role: Target role for the explanation (botanist, commander, engineer)
            
        Returns:
            Dictionary with natural language explanation
        """
        # Handle different explanation types
        if 'feature_importance' in explanation:
            # Global explanation
            return self._create_global_friendly_report(explanation, role)
        elif 'shap_force_plot' in explanation:
            # Local explanation
            return self._create_local_friendly_report(explanation, role)
        else:
            return {"error": "Unknown explanation type"}
    
    def _create_global_friendly_report(self, explanation, role):
        """Create user-friendly report for global explanations"""
        # Extract components
        feature_importance = explanation.get('feature_importance', {})
        interactions = explanation.get('interaction_effects', {})
        
        # Sort features by importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
        
        # Create natural language key factors
        key_factors = []
        
        # Different detail levels for different roles
        if role == "botanist":
            # Detailed botanical explanation
            for feature, data in top_features[:5]:
                key_factors.append({
                    'factor': feature.replace('_', ' ').title(),
                    'importance': f"{data['percentage']:.1f}%",
                    'explanation': self._generate_botanical_explanation(feature, data['importance'])
                })
            
            # Add interaction explanations
            for interaction_name, interaction_data in list(interactions.items())[:3]:
                key_factors.append({
                    'factor': f"Interaction: {interaction_data['features'][0].replace('_', ' ').title()} + {interaction_data['features'][1].replace('_', ' ').title()}",
                    'importance': f"{interaction_data['strength']*100:.1f}%",
                    'explanation': interaction_data['effect']
                })
                
        elif role == "commander":
            # High-level mission-focused explanation
            for feature, data in top_features[:3]:
                key_factors.append({
                    'factor': feature.replace('_', ' ').title(),
                    'importance': 'High' if data['importance'] > 0.2 else 'Medium' if data['importance'] > 0.1 else 'Low',
                    'explanation': self._generate_commander_explanation(feature, data['importance'])
                })
                
        elif role == "engineer":
            # Technical system-focused explanation
            for feature, data in top_features[:5]:
                key_factors.append({
                    'factor': feature.replace('_', ' ').title(),
                    'importance': f"{data['percentage']:.1f}%",
                    'explanation': self._generate_engineer_explanation(feature, data['importance'])
                })
        
        # Generate recommendations
        recommendations = self._generate_recommendations_from_global(explanation, role)
        
        return {
            'key_factors': key_factors,
            'recommended_actions': recommendations,
            'summary': self._generate_global_summary(explanation, role)
        }
    
    def _create_local_friendly_report(self, explanation, role):
        """Create user-friendly report for local explanations"""
        # Extract components
        force_plot = explanation.get('shap_force_plot', {})
        counterfactuals = explanation.get('counterfactual_scenarios', [])
        breakdown = explanation.get('prediction_breakdown', {})
        
        # Create natural language key factors
        key_factors = []
        
        # Add factors based on prediction components
        category_contributions = breakdown.get('category_contributions', {})
        for category, data in category_contributions.items():
            if abs(data['contribution']) > 0.01:  # Only include significant categories
                key_factors.append({
                    'factor': category.replace('_', ' ').title(),
                    'impact': 'Positive' if data['contribution'] > 0 else 'Negative',
                    'explanation': self._generate_category_explanation(category, data, role)
                })
        
        # Generate recommendations from counterfactuals
        recommendations = []
        for cf in counterfactuals:
            if cf['expected_improvement'] > 0.02:  # Only include worthwhile improvements
                recommendations.append({
                    'action': f"{cf['adjustment_direction'].capitalize()} {cf['feature'].replace('_', ' ')}",
                    'from': cf['current_value'],
                    'to': cf['suggested_value'],
                    'expected_benefit': 'Significant' if cf['expected_improvement'] > 0.1 else 'Moderate' if cf['expected_improvement'] > 0.05 else 'Minor',
                    'explanation': cf['explanation']
                })
        
        return {
            'key_factors': key_factors,
            'recommended_actions': recommendations,
            'summary': self._generate_local_summary(explanation, role)
        }
    
    def _generate_botanical_explanation(self, feature, importance):
        """Generate botanical explanation for a feature"""
        if feature == 'temperature':
            return "Temperature affects enzyme activity, transpiration rates, and overall metabolic processes in the plant."
        elif feature == 'light_intensity':
            return "Light intensity directly impacts photosynthesis rates and photomorphogenesis (light-dependent development)."
        elif feature == 'water_content':
            return "Water content affects turgor pressure, nutrient transport, and cellular processes within the plant tissues."
        elif feature == 'radiation_level':
            return "Radiation can damage cellular DNA and proteins, requiring energy expenditure for repair mechanisms."
        elif feature == 'nitrogen_level':
            return "Nitrogen is essential for amino acid synthesis, protein formation, and chlorophyll production."
        elif feature == 'phosphorus_level':
            return "Phosphorus is crucial for energy transfer (ATP), nucleic acid synthesis, and root development."
        elif feature == 'potassium_level':
            return "Potassium regulates stomatal function, enzyme activation, and water relations within the plant."
        else:
            return f"{feature.replace('_', ' ').title()} plays a significant role in plant development and health."
    
    def _generate_commander_explanation(self, feature, importance):
        """Generate mission commander explanation for a feature"""
        if feature == 'temperature':
            return "Temperature control directly affects crop yield and mission resource efficiency."
        elif feature == 'light_intensity':
            return "Light management impacts power consumption and crop production timelines."
        elif feature == 'water_content':
            return "Water management is critical for sustainable resource utilization in the closed system."
        elif feature == 'radiation_level':
            return "Radiation shielding affects both plant health and habitat safety margins."
        elif 'level' in feature:
            return f"{feature.replace('_level', '').title()} management affects long-term sustainability of the food production system."
        else:
            return f"{feature.replace('_', ' ').title()} has significant implications for mission success metrics."
    
    def _generate_engineer_explanation(self, feature, importance):
        """Generate engineer explanation for a feature"""
        if feature == 'temperature':
            return "Temperature control system requires precise calibration and has significant environmental coupling effects."
        elif feature == 'light_intensity':
            return "Light system power consumption scales with intensity; spectrum optimization can improve efficiency."
        elif feature == 'water_content':
            return "Water delivery and reclamation systems must maintain precise soil moisture levels while minimizing losses."
        elif feature == 'radiation_level':
            return "Radiation shielding effectiveness varies with material density and configuration; monitoring required."
        elif 'level' in feature:
            return f"{feature.replace('_level', '').title()} delivery systems require regular calibration and precise dosing mechanisms."
        else:
            return f"{feature.replace('_', ' ').title()} monitoring systems provide critical feedback for system optimization."
    
    def _generate_category_explanation(self, category, data, role):
        """Generate explanation for a prediction category"""
        contribution = data['contribution']
        direction = "positive" if contribution > 0 else "negative"
        
        if category == 'environmental':
            if role == "botanist":
                return f"Environmental conditions are having a {direction} effect, particularly {self._most_significant_feature(data['features'])}."
            else:
                return f"Environmental control systems are contributing {direction}ly to plant outcomes."
        
        elif category == 'water_management':
            if direction == "positive":
                return f"Current water management is beneficial for plant growth."
            else:
                return f"Water content requires adjustment; current levels are suboptimal."
        
        elif category == 'nutrients':
            if direction == "positive":
                return f"Nutrient levels are well-balanced for current growth needs."
            else:
                return f"Nutrient imbalance detected, particularly {self._most_significant_feature(data['features'])}."
        
        elif category == 'stress_factors':
            if direction == "positive":
                return f"Plant is showing good resilience to current stress factors."
            else:
                return f"Stress factors are negatively impacting plant performance."
        
        elif category == 'plant_status':
            if direction == "positive":
                return f"Current plant development stage is favorable."
            else:
                return f"Plant metrics indicate potential growth challenges."
        
        return f"{category.replace('_', ' ').title()} factors are having a {direction} impact."
    
    def _most_significant_feature(self, features):
        """Find the most significant feature in a dictionary"""
        if not features:
            return "none"
        
        most_sig = max(features.items(), key=lambda x: abs(x[1]))
        return most_sig[0].replace('_', ' ')
    
    def _generate_recommendations_from_global(self, explanation, role):
        """Generate recommendations from global explanation"""
        # Extract feature importance
        feature_importance = explanation.get('feature_importance', {})
        
        # Focus on top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)[:3]
        
        # Generate recommendations
        recommendations = []
        for feature, data in top_features:
            recommendations.append({
                'focus_area': feature.replace('_', ' ').title(),
                'importance': 'High' if data['importance'] > 0.2 else 'Medium' if data['importance'] > 0.1 else 'Low',
                'suggestion': self._generate_feature_recommendation(feature, role)
            })
        
        return recommendations
    
    def _generate_feature_recommendation(self, feature, role):
        """Generate recommendation for a feature"""
        if role == "botanist":
            if feature == 'temperature':
                return "Monitor diurnal temperature fluctuations and adjust for optimal enzymatic activity during key growth phases."
            elif feature == 'light_intensity':
                return "Consider adjusting light spectrum along with intensity for optimal photomorphogenic development."
            elif feature == 'water_content':
                return "Implement precise irrigation schedules based on growth stage and transpiration monitoring."
            elif feature == 'radiation_level':
                return "Monitor plant DNA repair mechanisms through gene expression markers when radiation events occur."
        
        elif role == "commander":
            if feature == 'temperature':
                return "Optimize temperature control to balance plant growth needs with power consumption constraints."
            elif feature == 'light_intensity':
                return "Schedule high-intensity lighting to align with solar power availability when possible."
            elif feature == 'water_content':
                return "Prioritize water recycling system maintenance to ensure consistent delivery capability."
            elif feature == 'radiation_level':
                return "Consider temporary additional shielding during solar events to protect both crew and crops."
        
        elif role == "engineer":
            if feature == 'temperature':
                return "Verify HVAC calibration and sensor placement for accurate microclimate monitoring."
            elif feature == 'light_intensity':
                return "Check LED efficiency metrics and consider adaptive control algorithms to optimize power usage."
            elif feature == 'water_content':
                return "Ensure moisture sensors are correctly calibrated and check for irrigation system pressure consistency."
            elif feature == 'radiation_level':
                return "Verify radiation shield integrity and monitor cumulative exposure metrics against established thresholds."
        
        # Generic recommendation if no specific one is available
        return f"Focus on optimizing {feature.replace('_', ' ')} based on plant response metrics."
    
    def _generate_global_summary(self, explanation, role):
        """Generate summary for global explanation"""
        if role == "botanist":
            return "The AI system has identified key physiological factors that influence plant growth and development in the space agriculture environment. Focus on these factors to optimize growth outcomes."
        elif role == "commander":
            return "This analysis highlights the most mission-critical factors affecting crop production efficiency. Prioritizing these areas will maximize resource utilization and yield reliability."
        elif role == "engineer":
            return "System analysis has identified the key technical parameters that require precise monitoring and control. Maintaining these parameters within optimal ranges will ensure peak performance."
        else:
            return "The AI has identified the most important factors affecting plant growth in the space agriculture system."
    
    def _generate_local_summary(self, explanation, role):
        """Generate summary for local explanation"""
        # Get predicted value
        prediction = explanation.get('shap_force_plot', {}).get('prediction', 0.5)
        
        # Determine if outcome is good or concerning
        outcome_status = "favorable" if prediction > 0.7 else "acceptable" if prediction > 0.5 else "concerning"
        
        if role == "botanist":
            return f"Current plant conditions are {outcome_status}. The analysis identifies specific biological factors contributing to this state and suggests targeted interventions to optimize plant physiological responses."
        elif role == "commander":
            return f"Crop status assessment is {outcome_status}. The analysis highlights key operational factors affecting current growth metrics and provides prioritized recommendations for resource allocation."
        elif role == "engineer":
            return f"System performance analysis indicates {outcome_status} conditions. Key technical parameters requiring attention have been identified, with specific calibration and adjustment recommendations."
        else:
            return f"Current plant growth conditions are {outcome_status}. The AI has identified factors that can be adjusted to improve outcomes."
    
    def suggest_interventions(self, explanation):
        """
        Suggest practical interventions based on an explanation
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Dictionary with intervention suggestions
        """
        # Handle different explanation types
        if 'shap_force_plot' in explanation:
            # Use counterfactuals for local explanations
            return self._suggest_from_counterfactuals(explanation.get('counterfactual_scenarios', []))
        else:
            # Use feature importance for global explanations
            return self._suggest_from_feature_importance(explanation.get('feature_importance', {}))
    
    def _suggest_from_counterfactuals(self, counterfactuals):
        """Generate intervention suggestions from counterfactuals"""
        suggestions = []
        
        for cf in sorted(counterfactuals, key=lambda x: x['expected_improvement'], reverse=True):
            if cf['expected_improvement'] > 0.02:  # Only worthwhile improvements
                suggestion = {
                    'parameter': cf['feature'],
                    'current_value': cf['current_value'],
                    'target_value': cf['suggested_value'],
                    'expected_benefit': cf['expected_improvement'],
                    'confidence': 'high' if cf['expected_improvement'] > 0.1 else 'medium' if cf['expected_improvement'] > 0.05 else 'low',
                    'explanation': cf['explanation']
                }
                suggestions.append(suggestion)
        
        return {
            'immediate_actions': suggestions[:2],  # Top 2 most impactful suggestions
            'additional_considerations': suggestions[2:],  # Any other valid suggestions
            'summary': self._intervention_summary(suggestions)
        }
    
    def _suggest_from_feature_importance(self, feature_importance):
        """Generate intervention suggestions from feature importance"""
        suggestions = []
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
        
        # Generate suggestions for top features
        for feature, data in sorted_features[:5]:
            if data['importance'] > 0.05:  # Only for relatively important features
                # Create suggestion based on feature
                suggestion = {
                    'parameter': feature,
                    'importance': data['importance'],
                    'focus_level': 'high' if data['importance'] > 0.2 else 'medium' if data['importance'] > 0.1 else 'low',
                    'suggested_approach': self._generate_monitoring_suggestion(feature, data['importance'])
                }
                suggestions.append(suggestion)
        
        return {
            'monitoring_priorities': suggestions[:3],  # Top 3 priorities
            'secondary_considerations': suggestions[3:],  # Any other suggestions
            'summary': "Focus monitoring and control efforts on these key parameters to optimize growing conditions."
        }
    
    def _generate_monitoring_suggestion(self, feature, importance):
        """Generate monitoring suggestion for a feature"""
        if feature == 'temperature':
            return "Implement precise temperature control with diurnal variation based on plant growth stage."
        elif feature == 'light_intensity':
            return "Optimize light intensity and spectrum to match current growth stage requirements."
        elif feature == 'water_content':
            return "Establish irrigation schedule based on plant transpiration rates and growth phase."
        elif feature == 'radiation_level':
            return "Monitor radiation levels and adjust shielding during solar events to protect sensitive growth stages."
        elif 'level' in feature:
            nutrient = feature.replace('_level', '')
            return f"Regularly monitor {nutrient} levels and adjust nutrient solution based on plant uptake rates."
        else:
            return f"Closely monitor {feature.replace('_', ' ')} and make adjustments based on plant response metrics."
    
    def _intervention_summary(self, suggestions):
        """Generate summary for intervention suggestions"""
        if not suggestions:
            return "No significant interventions required at this time."
        
        top_suggestion = suggestions[0]
        feature = top_suggestion['parameter']
        
        if len(suggestions) == 1:
            return f"Focus on adjusting {feature.replace('_', ' ')} for optimal growing conditions."
        else:
            features = [s['parameter'].replace('_', ' ') for s in suggestions[:2]]
            return f"Prioritize adjustments to {features[0]} and {features[1]} for maximum improvement in growing conditions."