"""
Plant Image Generator for Space Agriculture
Advanced version for generating synthetic plant images with space-relevant conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import logging

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using basic image processing for plant generation")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlantImageGenerator:
    """
    Generates synthetic plant images for different plant conditions in space agriculture
    """
    def __init__(self, image_size=(224, 224, 3)):
        """
        Initialize the image generator
        
        Args:
            image_size: Tuple of (height, width, channels) for generated images
        """
        self.image_size = image_size
        self.conditions = {
            "healthy": self._generate_healthy_plant,
            "scab": self._generate_scab_infection,
            "rust": self._generate_rust_infection,
            "microgravity_stress": self._generate_microgravity_stress,
            "radiation_damage": self._generate_radiation_damage,
            "nutrient_deficiency": self._generate_nutrient_deficiency,
            "multiple_diseases": self._generate_multiple_diseases
        }
        
    def generate_image(self, condition="healthy", species="generic"):
        """
        Generate a synthetic plant image for the specified condition
        
        Args:
            condition: String indicating the plant condition/disease
            species: String indicating plant species (for variations)
            
        Returns:
            NumPy array representing the image
        """
        if condition not in self.conditions:
            logger.warning(f"Unknown condition '{condition}', defaulting to 'healthy'")
            condition = "healthy"
            
        try:
            return self.conditions[condition](species)
        except Exception as e:
            logger.error(f"Error generating {condition} image: {e}")
            # Return a fallback image (gray with text)
            return self._generate_fallback_image(f"Error: {str(e)}")
            
    def _generate_base_plant(self, species="generic"):
        """Generate base plant image with species variations"""
        # Start with random noise - using numpy for compatibility
        if TENSORFLOW_AVAILABLE:
            try:
                noise = tf.random.normal([self.image_size[0], self.image_size[1], self.image_size[2]])
                noise = noise.numpy()  # Convert to numpy array
            except:
                # Fallback to numpy if TensorFlow operation fails
                noise = np.random.normal(0, 1, self.image_size)
        else:
            noise = np.random.normal(0, 1, self.image_size)
        
        # Create a green background
        # Adjust the green channel based on species
        if species == "Dwarf Wheat":
            green_intensity = 200
            variation = 30
            background_color = [100, green_intensity, 50]
        elif species == "Cherry Tomato":
            green_intensity = 180
            variation = 40
            background_color = [90, green_intensity, 40]
        elif species == "Lettuce":
            green_intensity = 220
            variation = 25
            background_color = [70, green_intensity, 30]
        elif species == "Space Potato":
            green_intensity = 170
            variation = 35
            background_color = [110, green_intensity, 60]
        else:  # Generic
            green_intensity = 190
            variation = 30
            background_color = [80, green_intensity, 40]
        
        # Create base image with green vegetation
        base_image = np.zeros(self.image_size, dtype=np.float32)
        
        # Fill with base color
        for c in range(3):
            base_image[:, :, c] = background_color[c]
        
        # Add some natural variation
        variation_mask = np.random.normal(0, 1, self.image_size) * variation
        base_image += variation_mask
        
        # Add plant structures (simplified)
        # Add stems
        stem_width = int(self.image_size[1] * 0.05)
        stem_center = self.image_size[1] // 2
        
        # Main stem
        base_image[self.image_size[0]//2:, stem_center-stem_width//2:stem_center+stem_width//2, 0] = 60
        base_image[self.image_size[0]//2:, stem_center-stem_width//2:stem_center+stem_width//2, 1] = 120
        base_image[self.image_size[0]//2:, stem_center-stem_width//2:stem_center+stem_width//2, 2] = 30
        
        # Clip to valid range
        base_image = np.clip(base_image, 0, 255)
        
        return base_image.astype(np.uint8)
    
    def _generate_healthy_plant(self, species="generic"):
        """Generate a healthy plant image"""
        plant_image = self._generate_base_plant(species)
        
        # For healthy plants, enhance the green channel
        plant_image[:, :, 1] = np.clip(plant_image[:, :, 1] * 1.1, 0, 255)
        
        return plant_image
    
    def _generate_scab_infection(self, species="generic"):
        """Generate a plant with scab infection"""
        plant_image = self._generate_base_plant(species)
        
        # Add dark lesions for scab
        num_lesions = np.random.randint(5, 15)
        for _ in range(num_lesions):
            x = np.random.randint(10, self.image_size[0] - 10)
            y = np.random.randint(10, self.image_size[1] - 10)
            radius = np.random.randint(5, 15)
            
            # Create mask for the lesion area
            for i in range(max(0, x-radius), min(self.image_size[0], x+radius)):
                for j in range(max(0, y-radius), min(self.image_size[1], y+radius)):
                    if (i - x)**2 + (j - y)**2 <= radius**2:
                        # Dark brown lesion
                        plant_image[i, j, 0] = np.clip(plant_image[i, j, 0] * 0.7, 0, 255)  # Reduce red
                        plant_image[i, j, 1] = np.clip(plant_image[i, j, 1] * 0.5, 0, 255)  # Reduce green
                        plant_image[i, j, 2] = np.clip(plant_image[i, j, 2] * 0.4, 0, 255)  # Reduce blue
                        
                        # Add yellowish halo around lesion
                        halo_radius = radius + 3
                        if (i - x)**2 + (j - y)**2 <= halo_radius**2 and (i - x)**2 + (j - y)**2 > radius**2:
                            plant_image[i, j, 0] = np.clip(plant_image[i, j, 0] * 1.5, 0, 255)  # Increase red
                            plant_image[i, j, 1] = np.clip(plant_image[i, j, 1] * 1.5, 0, 255)  # Increase green
                            plant_image[i, j, 2] = np.clip(plant_image[i, j, 2] * 0.5, 0, 255)  # Reduce blue
                            
        return plant_image
    
    def _generate_rust_infection(self, species="generic"):
        """Generate a plant with rust infection"""
        plant_image = self._generate_base_plant(species)
        
        # Add orange-brown pustules for rust
        num_pustules = np.random.randint(10, 30)
        for _ in range(num_pustules):
            x = np.random.randint(10, self.image_size[0] - 10)
            y = np.random.randint(10, self.image_size[1] - 10)
            radius = np.random.randint(3, 8)
            
            # Create mask for the pustule area
            for i in range(max(0, x-radius), min(self.image_size[0], x+radius)):
                for j in range(max(0, y-radius), min(self.image_size[1], y+radius)):
                    if (i - x)**2 + (j - y)**2 <= radius**2:
                        # Orange-brown pustule
                        plant_image[i, j, 0] = np.clip(plant_image[i, j, 0] * 2.0, 0, 255)  # Increase red
                        plant_image[i, j, 1] = np.clip(plant_image[i, j, 1] * 0.8, 0, 255)  # Reduce green
                        plant_image[i, j, 2] = np.clip(plant_image[i, j, 2] * 0.3, 0, 255)  # Reduce blue
        
        return plant_image
    
    def _generate_microgravity_stress(self, species="generic"):
        """Generate a plant with microgravity stress"""
        plant_image = self._generate_base_plant(species)
        
        # Apply geometric distortion to simulate irregular growth
        distortion_map_x = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32)
        distortion_map_y = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32)
        
        # Create warping effect
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                distortion_map_x[i, j] = j + np.sin(i/20) * 10
                distortion_map_y[i, j] = i + np.cos(j/20) * 10
        
        # Apply distortion
        distorted_image = np.zeros_like(plant_image)
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                src_x = int(distortion_map_x[i, j])
                src_y = int(distortion_map_y[i, j])
                
                if 0 <= src_x < self.image_size[1] and 0 <= src_y < self.image_size[0]:
                    distorted_image[i, j, :] = plant_image[src_y, src_x, :]
        
        # Weaken stems (reduce green, increase yellowing)
        stem_center = self.image_size[1] // 2
        stem_width = int(self.image_size[1] * 0.15)  # Wider area to represent weakened stem
        
        distorted_image[self.image_size[0]//2:, stem_center-stem_width//2:stem_center+stem_width//2, 0] += 30  # More red
        distorted_image[self.image_size[0]//2:, stem_center-stem_width//2:stem_center+stem_width//2, 1] -= 30  # Less green
        
        # Clip to valid range
        distorted_image = np.clip(distorted_image, 0, 255)
        
        return distorted_image.astype(np.uint8)
    
    def _generate_radiation_damage(self, species="generic"):
        """Generate a plant with radiation damage"""
        plant_image = self._generate_base_plant(species)
        
        # Add mottled discoloration and leaf curling effects
        
        # Create mottled pattern
        mottle_mask = np.random.normal(0, 1, (self.image_size[0], self.image_size[1]))
        mottle_mask = np.clip((mottle_mask + 0.5) * 2, 0, 1)  # Scale to 0-1
        
        # Apply varying levels of discoloration
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                if mottle_mask[i, j] > 0.7:  # High discoloration areas
                    plant_image[i, j, 0] = np.clip(plant_image[i, j, 0] * 1.4, 0, 255)  # More red
                    plant_image[i, j, 1] = np.clip(plant_image[i, j, 1] * 0.6, 0, 255)  # Less green
                    plant_image[i, j, 2] = np.clip(plant_image[i, j, 2] * 1.2, 0, 255)  # More blue
        
        # Add leaf curling effects (simplified via edge distortion)
        # Add some darker spots to represent necrotic areas from radiation damage
        num_spots = np.random.randint(5, 15)
        for _ in range(num_spots):
            x = np.random.randint(10, self.image_size[0] - 10)
            y = np.random.randint(10, self.image_size[1] - 10)
            radius = np.random.randint(5, 12)
            
            for i in range(max(0, x-radius), min(self.image_size[0], x+radius)):
                for j in range(max(0, y-radius), min(self.image_size[1], y+radius)):
                    if (i - x)**2 + (j - y)**2 <= radius**2:
                        # Dark spot with purple hue (radiation damage)
                        plant_image[i, j, 0] = np.clip(plant_image[i, j, 0] * 0.6, 0, 255)  # Reduce red
                        plant_image[i, j, 1] = np.clip(plant_image[i, j, 1] * 0.4, 0, 255)  # Reduce green
                        plant_image[i, j, 2] = np.clip(plant_image[i, j, 2] * 0.9, 0, 255)  # Keep blue
        
        return plant_image
    
    def _generate_nutrient_deficiency(self, species="generic"):
        """Generate a plant with nutrient deficiency"""
        plant_image = self._generate_base_plant(species)
        
        # Chlorosis (yellowing) effect - reduce green, increase red/yellow
        plant_image[:, :, 0] = np.clip(plant_image[:, :, 0] * 1.3, 0, 255)  # Increase red
        plant_image[:, :, 1] = np.clip(plant_image[:, :, 1] * 0.7, 0, 255)  # Decrease green
        
        # Add interveinal chlorosis pattern (yellowing between leaf veins)
        # Simplified approach using random pattern
        
        # Create pattern of veins (darker green lines)
        for i in range(0, self.image_size[0], 10):
            thickness = np.random.randint(1, 3)
            for t in range(thickness):
                if i+t < self.image_size[0]:
                    # Keep veins more green
                    plant_image[i+t, :, 0] = np.clip(plant_image[i+t, :, 0] * 0.8, 0, 255)  # Less red in veins
                    plant_image[i+t, :, 1] = np.clip(plant_image[i+t, :, 1] * 1.2, 0, 255)  # More green in veins
        
        # Add some necrotic spots (brown/black) on leaf edges
        num_spots = np.random.randint(3, 10)
        for _ in range(num_spots):
            # Focus on edges of the image to represent leaf edges
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])
            
            if edge == 'top':
                x = np.random.randint(0, 30)
                y = np.random.randint(0, self.image_size[1])
            elif edge == 'bottom':
                x = np.random.randint(self.image_size[0] - 30, self.image_size[0])
                y = np.random.randint(0, self.image_size[1])
            elif edge == 'left':
                x = np.random.randint(0, self.image_size[0])
                y = np.random.randint(0, 30)
            else:  # right
                x = np.random.randint(0, self.image_size[0])
                y = np.random.randint(self.image_size[1] - 30, self.image_size[1])
                
            radius = np.random.randint(5, 15)
            
            for i in range(max(0, x-radius), min(self.image_size[0], x+radius)):
                for j in range(max(0, y-radius), min(self.image_size[1], y+radius)):
                    if (i - x)**2 + (j - y)**2 <= radius**2:
                        # Dark brown necrotic spot
                        plant_image[i, j, 0] = np.clip(plant_image[i, j, 0] * 0.7, 0, 255)
                        plant_image[i, j, 1] = np.clip(plant_image[i, j, 1] * 0.4, 0, 255)
                        plant_image[i, j, 2] = np.clip(plant_image[i, j, 2] * 0.3, 0, 255)
        
        return plant_image
    
    def _generate_multiple_diseases(self, species="generic"):
        """Generate a plant with multiple disease conditions"""
        # Start with base plant
        plant_image = self._generate_base_plant(species)
        
        # Randomly combine 2-3 conditions
        conditions = list(self.conditions.keys())
        conditions.remove("multiple_diseases")  # Don't include this one
        num_conditions = np.random.randint(2, 4)
        selected_conditions = np.random.choice(conditions, num_conditions, replace=False)
        
        # Apply milder versions of each selected condition
        for condition in selected_conditions:
            if condition == "healthy":
                continue  # Skip healthy
                
            # Generate the condition image
            condition_image = self.conditions[condition](species)
            
            # Blend 30-50% of the condition into the base image
            blend_factor = np.random.uniform(0.3, 0.5)
            plant_image = plant_image * (1 - blend_factor) + condition_image * blend_factor
        
        # Ensure image is in valid range and correct type
        plant_image = np.clip(plant_image, 0, 255).astype(np.uint8)
        
        return plant_image
    
    def _generate_fallback_image(self, message="Error"):
        """Generate a fallback image with text message"""
        # Create gray background
        img = np.ones(self.image_size) * 128
        
        # Add text would go here in a full implementation
        # For this simplified version, we just return the gray image
        
        return img.astype(np.uint8)
    
    def save_image(self, image, filename):
        """Save the generated image to disk"""
        try:
            plt.imsave(filename, image)
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

    def image_to_base64(self, image):
        """Convert image array to base64 string for display"""
        try:
            # Save to BytesIO object
            buffered = BytesIO()
            plt.imsave(buffered, image, format="PNG")
            buffered.seek(0)
            
            # Get base64 encoded string
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        except Exception as e:
            logger.error(f"Error converting to base64: {e}")
            return None

def generate_example_images(output_dir="example_images"):
    """Generate example images for all conditions"""
    generator = PlantImageGenerator()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate example for each condition and each species
    species_list = ["Dwarf Wheat", "Cherry Tomato", "Lettuce", "Space Potato"]
    
    for condition in generator.conditions.keys():
        for species in species_list:
            try:
                image = generator.generate_image(condition, species)
                filename = f"{output_dir}/{species.replace(' ', '_')}_{condition}.png"
                generator.save_image(image, filename)
                logger.info(f"Generated {filename}")
            except Exception as e:
                logger.error(f"Error generating {condition} for {species}: {e}")

if __name__ == "__main__":
    # Generate example images when script is run directly
    generate_example_images()