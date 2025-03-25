"""
Plant Disease Detection Module for Space Agriculture
Advanced version for space-relevant plant diseases with support for Plant Pathology 2020 dataset
"""

import numpy as np
import base64
from io import BytesIO
import streamlit as st
import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using basic image processing")

# Import plant pathology dataset module
try:
    from plant_pathology_dataset import PlantPathologyDataset, extract_image_features, analyze_disease_patterns
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Plant Pathology dataset module not available - using default disease detection")

# Dictionary of plant diseases and their characteristics specific to space agriculture
PLANT_DISEASES = {
    "healthy": {
        "description": "Plant shows no signs of disease or stress",
        "remediation": "Continue with current growing conditions",
        "confidence_threshold": 0.7,
        "reward_modifier": 1.0,  # Multiplier for the RL agent's reward
        "symptoms": "Vibrant color, uniform growth, no discoloration or lesions"
    },
    "scab": {
        "description": "Fungal disease causing dark, crusty lesions on leaves",
        "remediation": "Increase air circulation, reduce humidity, apply fungicide treatment",
        "confidence_threshold": 0.6,
        "reward_modifier": 0.7,  # Penalty for disease presence
        "symptoms": "Dark, raised lesions on leaves with yellowish halos"
    },
    "rust": {
        "description": "Fungal infection causing orange-brown pustules on leaves",
        "remediation": "Reduce leaf wetness, increase spacing between plants, apply fungicide",
        "confidence_threshold": 0.6,
        "reward_modifier": 0.7,
        "symptoms": "Reddish-brown pustules on leaf undersides, yellow spots on upper surfaces"
    },
    "microgravity_stress": {
        "description": "Stress due to microgravity conditions affecting plant growth orientation",
        "remediation": "Implement directional lighting, increase mechanical stimulation, adjust nutrient delivery",
        "confidence_threshold": 0.65,
        "reward_modifier": 0.8,
        "symptoms": "Irregular growth patterns, weakened stems, unusual branching angles"
    },
    "radiation_damage": {
        "description": "Cellular damage caused by cosmic radiation exposure",
        "remediation": "Increase radiation shielding, supplement with antioxidants, consider selective breeding for resistance",
        "confidence_threshold": 0.6,
        "reward_modifier": 0.6,
        "symptoms": "Mottled discoloration, leaf curling, stunted growth, genetic abnormalities"
    },
    "nutrient_deficiency": {
        "description": "Insufficient essential nutrients for proper growth",
        "remediation": "Adjust nutrient solution composition, check delivery system, consider foliar application",
        "confidence_threshold": 0.7,
        "reward_modifier": 0.8,
        "symptoms": "Chlorosis (yellowing), necrotic spots, stunted growth, specific patterns on older/newer leaves"
    },
    "multiple_diseases": {
        "description": "Plant shows signs of multiple infections or stress factors",
        "remediation": "Immediate isolation and comprehensive treatment needed",
        "confidence_threshold": 0.5,
        "reward_modifier": 0.5,
        "symptoms": "Combined symptoms from multiple conditions, overall declining health"
    }
}

# Simplified model for disease detection
class PlantDiseaseDetector:
    """
    Plant disease detector with support for Plant Pathology 2020 dataset.
    Can use either basic feature extraction or TensorFlow if available.
    """
    def __init__(self):
        """Initialize the plant disease detector"""
        # Initialize with all possible disease classes
        self.disease_classes = list(PLANT_DISEASES.keys())
        
        # Load Plant Pathology dataset if available
        self.dataset = None
        self.disease_patterns = None
        
        if DATASET_AVAILABLE:
            try:
                self.dataset = PlantPathologyDataset()
                self.disease_patterns = analyze_disease_patterns(self.dataset)
                logging.info("Plant Pathology dataset and patterns loaded successfully")
            except Exception as e:
                logging.error(f"Error loading Plant Pathology dataset: {str(e)}")
        else:
            logging.info("Plant Pathology dataset not available, using built-in detection")
    
    def preprocess_image(self, img_array):
        """
        Preprocess the image for the model
        
        Args:
            img_array: Numpy array of the image
            
        Returns:
            Preprocessed image array
        """
        # Resize to expected input dimensions (224x224 for many CNN models)
        # This is a simplified version - in a real implementation we would use OpenCV or TensorFlow
        # for proper preprocessing including normalization
        try:
            # Ensure the image is in RGB format and properly sized
            img_resized = img_array
            
            # Return the preprocessed image
            return img_resized
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, img_array):
        """
        Predict the disease class of the plant image
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            Dictionary with disease probabilities
        """
        # In a real implementation, this would run the image through a trained model
        try:
            # Check if we should use the Plant Pathology dataset-based prediction
            if DATASET_AVAILABLE and self.dataset is not None:
                return self._predict_with_dataset(img_array)
            else:
                return self._predict_with_features(img_array)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            logging.error(f"Prediction error: {str(e)}")
            # Fallback to uniform distribution
            return {disease: 1.0/len(self.disease_classes) for disease in self.disease_classes}
            
    def _predict_with_dataset(self, img_array):
        """
        Predict using the Plant Pathology 2020 dataset
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            Dictionary with disease probabilities
        """
        try:
            # Extract features from the image
            features = extract_image_features(img_array)
            
            # Log the extracted features
            logging.info(f"Extracted image features: {features}")
            
            # Get predictions from the dataset model
            plant_path_predictions = self.dataset.get_disease_probabilities(features)
            
            # Get our standard space agriculture predictions as well
            space_agri_predictions = self._predict_with_features(img_array)
            
            # Initialize the combined prediction dictionary
            prediction = {}
            
            # First, include the standard plant pathology diseases
            for disease in ['healthy', 'rust', 'scab', 'multiple_diseases']:
                if disease in plant_path_predictions:
                    # Use plant pathology dataset predictions for these diseases with higher weight
                    prediction[disease] = plant_path_predictions[disease] * 0.7 + space_agri_predictions.get(disease, 0) * 0.3
                elif disease in space_agri_predictions:
                    prediction[disease] = space_agri_predictions[disease]
            
            # Then, include the space-specific diseases with their original probabilities
            for disease in ['microgravity_stress', 'radiation_damage', 'nutrient_deficiency']:
                if disease in space_agri_predictions:
                    prediction[disease] = space_agri_predictions[disease]
            
            # Normalize probabilities to ensure they sum to 1.0
            total = sum(prediction.values())
            for key in prediction:
                prediction[key] = prediction[key] / total
                
            # Log the combined predictions
            logging.info(f"Combined predictions (dataset + features): {prediction}")
            
            return prediction
        
        except Exception as e:
            logging.error(f"Error in dataset-based prediction: {str(e)}")
            # Fallback to feature-based prediction
            return self._predict_with_features(img_array)
            
    def _predict_with_features(self, img_array):
        """
        Predict disease based on extracted image features
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            Dictionary with disease probabilities
        """
        # Extract multiple features that correlate with different disease patterns
        
        # Basic image statistics
        brightness = np.mean(img_array)
        variance = np.var(img_array)
        
        # Color channel analysis (RGB)
        r_channel = img_array[:,:,0] if img_array.ndim >= 3 else img_array
        g_channel = img_array[:,:,1] if img_array.ndim >= 3 else img_array
        b_channel = img_array[:,:,2] if img_array.ndim >= 3 else img_array
        
        r_mean = np.mean(r_channel)
        g_mean = np.mean(g_channel)
        b_mean = np.mean(b_channel)
        
        # Color ratios - useful for detecting chlorosis and other color-based diseases
        rg_ratio = r_mean / (g_mean + 1e-10)  # Avoid division by zero
        rb_ratio = r_mean / (b_mean + 1e-10)
        
        # Texture analysis - approximated by local variance 
        # Higher values might indicate spots, lesions, or irregular growth
        texture_variance = np.var(img_array - np.mean(img_array, axis=(0,1)))
        
        # Edge detection approximation - look for sudden changes in intensity
        # High edge values might indicate lesions, abnormal growth patterns
        h_gradient = np.mean(np.abs(np.diff(g_channel, axis=1)))
        v_gradient = np.mean(np.abs(np.diff(g_channel, axis=0)))
        edge_intensity = (h_gradient + v_gradient) / 2
        
        # Initialize prediction dictionary with all disease types
        prediction = {disease: 0.1 for disease in self.disease_classes}
        
        # Enhanced prediction logic based on multiple features
        
        # Healthy plants typically have strong green channel, balanced texture
        greenness = g_mean / (r_mean + b_mean + 1e-10)
        if greenness > 0.4 and variance < 1500 and texture_variance < 1000:
            prediction["healthy"] = 0.7 + np.random.random() * 0.2
        else:
            prediction["healthy"] = 0.2 + np.random.random() * 0.3
        
        # Scab often shows as dark lesions (low brightness in spots, high variance)
        if brightness < 100 and variance > 1500 and edge_intensity > 20:
            prediction["scab"] = 0.5 + np.random.random() * 0.3
        else:
            prediction["scab"] = 0.1 + np.random.random() * 0.2
        
        # Rust shows as orange-brown spots (high red channel, medium green)
        if rg_ratio > 1.2 and rb_ratio > 1.2 and variance > 1200:
            prediction["rust"] = 0.6 + np.random.random() * 0.3
        else:
            prediction["rust"] = 0.1 + np.random.random() * 0.2
        
        # Microgravity stress often results in irregular growth patterns
        # Higher texture variance and edge values might indicate this
        if texture_variance > 1200 and edge_intensity > 25:
            prediction["microgravity_stress"] = 0.5 + np.random.random() * 0.3
        else:
            prediction["microgravity_stress"] = 0.1 + np.random.random() * 0.15
        
        # Radiation damage often shows as mottled discoloration
        # High variance across all channels without clear patterns
        channel_variance = np.var([r_mean, g_mean, b_mean])
        if channel_variance > 500 and texture_variance > 1000:
            prediction["radiation_damage"] = 0.5 + np.random.random() * 0.3
        else:
            prediction["radiation_damage"] = 0.1 + np.random.random() * 0.1
        
        # Nutrient deficiency often shows as yellowing (high red-green ratio)
        if rg_ratio < 0.8 and g_mean < 100 and brightness < 120:
            prediction["nutrient_deficiency"] = 0.6 + np.random.random() * 0.3
        else:
            prediction["nutrient_deficiency"] = 0.1 + np.random.random() * 0.2
        
        # Calculate probability of multiple issues based on the other predictions
        # Higher if we have significant probabilities for more than one condition
        sorted_predictions = sorted([(k, v) for k, v in prediction.items() if k != "multiple_diseases"], 
                                   key=lambda x: x[1], reverse=True)
        
        if len(sorted_predictions) >= 2 and sorted_predictions[1][1] > 0.3:
            # If second highest prediction is significant
            top_two_sum = sorted_predictions[0][1] + sorted_predictions[1][1]
            prediction["multiple_diseases"] = min(0.2 + top_two_sum / 3, 0.7)
        else:
            prediction["multiple_diseases"] = 0.05 + np.random.random() * 0.1
        
        # Normalize to ensure sum is close to 1.0
        total = sum(prediction.values())
        for key in prediction:
            prediction[key] = prediction[key] / total
        
        # Log the feature values
        logging.debug(f"Image features: brightness={brightness:.2f}, variance={variance:.2f}, " +
                    f"rg_ratio={rg_ratio:.2f}, texture_var={texture_variance:.2f}")
        
        return prediction
    
    def get_diagnosis(self, predictions):
        """
        Convert model predictions to a comprehensive diagnosis report
        
        Args:
            predictions: Dictionary with disease probabilities
            
        Returns:
            Dictionary with detailed diagnosis information
        """
        # Find the highest probability disease
        predicted_class = max(predictions, key=predictions.get)
        confidence = predictions[predicted_class]
        
        # Get information about the predicted disease
        disease_info = PLANT_DISEASES[predicted_class]
        
        # Check if confidence exceeds threshold for this disease
        is_confident = confidence >= disease_info["confidence_threshold"]
        
        # Find secondary conditions (any other condition with significant probability)
        secondary_conditions = []
        for disease, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            if disease != predicted_class and prob > 0.15:  # Only include significant probabilities
                secondary_conditions.append({
                    "name": disease,
                    "probability": prob,
                    "description": PLANT_DISEASES[disease]["description"]
                })
        
        # Generate specific recommendations based on the detected condition
        recommendations = []
        
        # Start with the primary recommendation from the disease info
        primary_recommendation = disease_info["remediation"]
        recommendations.append(primary_recommendation)
        
        # Add specific recommendations based on the condition
        if predicted_class == "healthy":
            recommendations.append("Continue monitoring for early signs of stress or disease")
            recommendations.append("Maintain optimal environmental conditions for the specific plant species")
        
        elif predicted_class == "scab" or predicted_class == "rust":
            recommendations.append("Consider adjusting humidity levels to discourage fungal growth")
            recommendations.append("Inspect nearby plants for signs of infection")
            recommendations.append("If possible, remove and isolate affected plant parts")
        
        elif predicted_class == "microgravity_stress":
            recommendations.append("Implement mechanical stimulation to strengthen plant tissues")
            recommendations.append("Consider using directional lighting to guide plant growth")
            recommendations.append("Adjust nutrient solution flow rates to improve distribution in microgravity")
        
        elif predicted_class == "radiation_damage":
            recommendations.append("Relocate plants to a more shielded area if possible")
            recommendations.append("Supplement nutrients with antioxidants to mitigate cellular damage")
            recommendations.append("Monitor genetic stability in subsequent generations")
        
        elif predicted_class == "nutrient_deficiency":
            recommendations.append("Perform a detailed nutrient solution analysis")
            recommendations.append("Check for precipitates or blockages in the delivery system")
            recommendations.append("Consider a foliar application of micronutrients as a quick intervention")
        
        elif predicted_class == "multiple_diseases":
            recommendations.append("Isolate affected plants immediately to prevent cross-contamination")
            recommendations.append("Implement comprehensive environmental control measures")
            recommendations.append("Consider genetic testing for precise diagnosis")
        
        # Determine a health status based on the condition and confidence
        if predicted_class == "healthy" and is_confident:
            health_status = "Healthy"
        elif predicted_class == "healthy" and not is_confident:
            health_status = "Moderate Risk"
        elif not is_confident:
            health_status = "Moderate Risk"
        elif predicted_class in ["scab", "rust", "nutrient_deficiency"]:
            health_status = "Moderate Risk"
        else:  # More serious conditions or multiple issues
            health_status = "Severe Risk"
            
        # Calculate disease severity as an inverse of the reward modifier
        # (1.0 means no impact, lower values mean more severe)
        disease_severity = 1.0 - disease_info["reward_modifier"] if predicted_class != "healthy" else 0.0
        
        # Enhanced diagnosis with more details
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "description": disease_info["description"],
            "symptoms": disease_info["symptoms"],
            "remediation": disease_info["remediation"],
            "recommendations": recommendations,
            "is_confident": is_confident,
            "health_status": health_status,
            "disease_name": "None" if predicted_class == "healthy" else predicted_class.replace("_", " ").title(),
            "disease_severity": disease_severity,
            "secondary_conditions": secondary_conditions,
            "reward_modifier": disease_info["reward_modifier"] if is_confident else 1.0
        }

def generate_report(diagnosis):
    """
    Generate a comprehensive formatted report from the diagnosis
    
    Args:
        diagnosis: Dictionary with diagnosis information
        
    Returns:
        Formatted report string with markdown formatting
    """
    # Determine the status color for visual indication
    if diagnosis['health_status'] == "Healthy":
        status_color = "green"
        status_emoji = "✅"
    elif diagnosis['health_status'] == "Moderate Risk":
        status_color = "orange"
        status_emoji = "⚠️"
    else:  # Severe Risk
        status_color = "red"
        status_emoji = "🚨"
    
    # Format the confidence as a percentage
    confidence_pct = diagnosis['confidence'] * 100
    
    # Create the header and overall assessment
    report = f"""
    ## Plant Health Diagnosis
    
    ### Overall Assessment
    <div style="color:{status_color}; font-size:1.2em; font-weight:bold;">
    {status_emoji} Status: {diagnosis['health_status']}
    </div>
    
    **Condition**: {diagnosis['predicted_class'].replace('_', ' ').title()}  
    **Confidence**: {confidence_pct:.1f}%  
    **Severity**: {(diagnosis['disease_severity'] * 100):.1f}% (if applicable)
    
    ### Description
    {diagnosis['description']}
    
    ### Observed Symptoms
    {diagnosis['symptoms']}
    """
    
    # Add secondary conditions if present
    if diagnosis.get('secondary_conditions'):
        report += "\n\n### Potential Secondary Conditions\n"
        for condition in diagnosis['secondary_conditions']:
            report += f"- **{condition['name'].replace('_', ' ').title()}** ({condition['probability']*100:.1f}%): {condition['description']}\n"
    
    # Add recommendations
    report += "\n\n### Recommended Actions\n"
    for i, recommendation in enumerate(diagnosis['recommendations']):
        report += f"{i+1}. {recommendation}\n"
    
    # Add AI confidence assessment
    report += f"""
    ### AI Confidence Assessment
    {'✅ High confidence diagnosis' if diagnosis['is_confident'] else '⚠️ Low confidence - Please verify manually'}
    """
    
    # Add environmental impact information if disease is detected
    if diagnosis['predicted_class'] != "healthy":
        report += f"""
        ### Environmental Impact
        Reward Modifier: {diagnosis['reward_modifier']:.2f} (affects RL agent optimization)
        """
    
    return report

def image_to_base64(img_array):
    """
    Convert image array to base64 string for display
    
    Args:
        img_array: Numpy array of the image
        
    Returns:
        Base64 encoded string
    """
    try:
        from PIL import Image
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_array.astype('uint8'))
        
        # Save to BytesIO object
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        
        # Get base64 encoded string
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Error converting image to base64: {e}")
        return None

def apply_diagnosis_to_environment(env, diagnosis):
    """
    Apply diagnosis results to the environment state
    
    Args:
        env: Space agriculture environment
        diagnosis: Dictionary with diagnosis information
        
    Returns:
        Modified environment
    """
    # In a production implementation, this would update the environment state
    # based on the detected plant disease
    
    # Apply reward modifier to the environment's reward calculation
    reward_modifier = diagnosis["reward_modifier"]
    
    # Store the reward modifier in the environment for use during reward calculation
    env.disease_reward_modifier = reward_modifier
    
    # If there's a disease, we might also want to modify the plant's health directly
    if diagnosis["predicted_class"] != "healthy" and diagnosis["is_confident"]:
        # Reduce plant health proportionally to disease severity (1.0 - reward_modifier)
        health_reduction = (1.0 - reward_modifier) * 0.2  # Scale the effect
        env.state["health_score"] = max(0.1, env.state["health_score"] - health_reduction)
    
    return env