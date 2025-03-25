"""
Plant Disease Detection Module for Space Agriculture
Based on the Plant Pathology 2020 dataset approach
"""

import numpy as np
import base64
from io import BytesIO
import streamlit as st

# Dictionary of plant diseases and their characteristics
PLANT_DISEASES = {
    "healthy": {
        "description": "Plant shows no signs of disease",
        "remediation": "Continue with current growing conditions",
        "confidence_threshold": 0.7,
        "reward_modifier": 1.0  # Multiplier for the RL agent's reward
    },
    "scab": {
        "description": "Fungal disease causing dark, crusty lesions on leaves",
        "remediation": "Increase air circulation, reduce humidity, apply fungicide treatment",
        "confidence_threshold": 0.6,
        "reward_modifier": 0.7  # Penalty for disease presence
    },
    "rust": {
        "description": "Fungal infection causing orange-brown pustules on leaves",
        "remediation": "Reduce leaf wetness, increase spacing between plants, apply fungicide",
        "confidence_threshold": 0.6,
        "reward_modifier": 0.7
    },
    "multiple_diseases": {
        "description": "Plant shows signs of multiple infections",
        "remediation": "Immediate isolation and comprehensive treatment needed",
        "confidence_threshold": 0.5,
        "reward_modifier": 0.5
    }
}

# Simplified model for disease detection
class PlantDiseaseDetector:
    """
    A simplified plant disease detector that mimics a trained CNN model.
    In a full implementation, this would use a trained TensorFlow/PyTorch model.
    """
    def __init__(self):
        """Initialize the plant disease detector"""
        # We would load a pre-trained model here in a full implementation
        self.disease_classes = list(PLANT_DISEASES.keys())
    
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
        # Here we simulate the prediction with a probabilistic approach based on image characteristics
        
        # For demo purposes, we'll generate "predictions" based on simple image properties
        try:
            # Generate simple random predictions 
            # In a real implementation, this would be the output of the model
            # Here we're just simulating output for demonstration
            
            # Analyze image characteristics 
            brightness = np.mean(img_array)
            variance = np.var(img_array)
            
            # Use image characteristics to bias our simulated predictions
            # This is just a demonstration of how image features might correlate with diseases
            
            # Initialize prediction scores
            prediction = {}
            
            # Higher brightness often correlates with healthy leaves
            if brightness > 100:
                prediction["healthy"] = 0.7 + np.random.random() * 0.2
            else:
                prediction["healthy"] = 0.2 + np.random.random() * 0.3
                
            # Higher variance might indicate lesions or spots
            if variance > 1000:
                prediction["scab"] = 0.5 + np.random.random() * 0.3
                prediction["rust"] = 0.4 + np.random.random() * 0.3
            else:
                prediction["scab"] = 0.1 + np.random.random() * 0.2
                prediction["rust"] = 0.1 + np.random.random() * 0.2
                
            # Multiple diseases probability
            prediction["multiple_diseases"] = min(
                0.2 + (prediction["scab"] + prediction["rust"]) / 4,
                0.8
            )
            
            # Normalize to ensure sum is close to 1.0
            total = sum(prediction.values())
            for key in prediction:
                prediction[key] = prediction[key] / total
                
            return prediction
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return {disease: 0.25 for disease in self.disease_classes}
    
    def get_diagnosis(self, predictions):
        """
        Convert model predictions to a diagnosis report
        
        Args:
            predictions: Dictionary with disease probabilities
            
        Returns:
            Dictionary with diagnosis information
        """
        # Find the highest probability disease
        predicted_class = max(predictions, key=predictions.get)
        confidence = predictions[predicted_class]
        
        # Get information about the predicted disease
        disease_info = PLANT_DISEASES[predicted_class]
        
        # Check if confidence exceeds threshold for this disease
        is_confident = confidence >= disease_info["confidence_threshold"]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "description": disease_info["description"],
            "remediation": disease_info["remediation"],
            "is_confident": is_confident,
            "reward_modifier": disease_info["reward_modifier"] if is_confident else 1.0
        }

def generate_report(diagnosis):
    """
    Generate a formatted report from the diagnosis
    
    Args:
        diagnosis: Dictionary with diagnosis information
        
    Returns:
        Formatted report string
    """
    report = f"""
    ## Plant Health Diagnosis
    
    **Condition**: {diagnosis['predicted_class'].replace('_', ' ').title()}
    **Confidence**: {diagnosis['confidence']:.2f}
    
    ### Description
    {diagnosis['description']}
    
    ### Recommended Actions
    {diagnosis['remediation']}
    
    ### AI Confidence Assessment
    {'✅ High confidence diagnosis' if diagnosis['is_confident'] else '⚠️ Low confidence - Please verify'}
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