"""
Space Agriculture Research Knowledge Base
This module incorporates scientific research findings about space agriculture to enhance
the reinforcement learning model's decision-making capabilities.

Research sources:
- https://www.sciencedirect.com/science/article/pii/S2949736125000338
- https://science.nasa.gov/wp-content/uploads/2023/05/47_7b34a38f75ed824552f1e41774330422_EscobarChristineM.pdf
- https://www.sciencedirect.com/science/article/pii/S2772375524000777
- https://patents.google.com/patent/US11308715B2/en
- https://www.nature.com/articles/s41598-023-30846-y
"""

import logging
import numpy as np
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scientific findings from research papers
SPACE_AGRICULTURE_RESEARCH = {
    # Research domains based on the provided papers
    "microgravity_effects": {
        "description": "Effects of microgravity on plant growth and development",
        "findings": [
            {
                "title": "Root orientation",
                "detail": "Plants in microgravity show disoriented root growth patterns due to absence of gravitropism",
                "mitigation": "Directional lighting and mechanical stimulation can help orient root growth",
                "impact_factor": 0.8  # How much this affects plant growth (0-1)
            },
            {
                "title": "Water distribution",
                "detail": "Water forms spherical droplets in microgravity, leading to uneven distribution in soil/media",
                "mitigation": "Specialized watering systems with capillary mats or hydroponic/aeroponic approaches",
                "impact_factor": 0.9
            },
            {
                "title": "Cell wall development",
                "detail": "Reduced lignification and altered cell wall composition leads to weaker structural integrity",
                "mitigation": "Supplementation with silicon and calcium can improve cell wall strength",
                "impact_factor": 0.75
            },
            {
                "title": "Gene expression",
                "detail": "Over 200 genes show altered expression in microgravity conditions",
                "mitigation": "Selecting cultivars with genetic resilience to microgravity stress",
                "impact_factor": 0.7
            }
        ],
        "optimal_parameters": {
            "mechanical_stimulation_interval_hours": 4,
            "directional_light_intensity_ratio": 2.5,  # Ratio between top and side lighting
            "vibration_frequency_hz": 15,  # Mechanical stimulation frequency
            "silicon_supplementation_ppm": 50  # Silicon in nutrient solution
        }
    },
    "radiation_effects": {
        "description": "Effects of cosmic radiation on plant growth and genetic stability",
        "findings": [
            {
                "title": "DNA damage",
                "detail": "Increased mutation rates and DNA damage from cosmic radiation, particularly from heavy ions",
                "mitigation": "Antioxidant supplementation and radiation shielding",
                "impact_factor": 0.9
            },
            {
                "title": "Photosynthetic efficiency",
                "detail": "Reduced chlorophyll content and photosystem II efficiency under high radiation",
                "mitigation": "Managed light cycles and protective films that filter harmful radiation",
                "impact_factor": 0.85
            },
            {
                "title": "Seed viability",
                "detail": "Reduced germination rates and increased abnormalities in next-generation seeds",
                "mitigation": "Seed vault with proper shielding for long-term missions",
                "impact_factor": 0.8
            },
            {
                "title": "Oxidative stress",
                "detail": "Increased reactive oxygen species (ROS) production leading to cellular damage",
                "mitigation": "Supplementation with ascorbic acid, tocopherol, and other antioxidants",
                "impact_factor": 0.85
            }
        ],
        "optimal_parameters": {
            "shielding_material_density_g_cm3": 7.8,  # Iron-based shielding
            "antioxidant_concentration_mmol": 5.0,
            "light_spectrum_optimization": "Red-enriched",  # More red light compensates for photosystem damage
            "radiation_monitoring_interval_hours": 6
        }
    },
    "nutrient_delivery": {
        "description": "Optimized nutrient delivery systems for space agriculture",
        "findings": [
            {
                "title": "Nutrient film technique",
                "detail": "Modified NFT systems for microgravity show promising results for leafy greens",
                "mitigation": "Capillary-based nutrient delivery with careful flow rate control",
                "impact_factor": 0.8
            },
            {
                "title": "Aeroponics advantages",
                "detail": "Aeroponic systems use 98% less water and 60% less nutrients than soil-based systems",
                "mitigation": "High-pressure aeroponic systems with droplet size optimization",
                "impact_factor": 0.9
            },
            {
                "title": "Micronutrient bioavailability",
                "detail": "Altered pH dynamics in space affect micronutrient uptake, especially iron and manganese",
                "mitigation": "Chelated micronutrients and dynamic pH management systems",
                "impact_factor": 0.75
            },
            {
                "title": "Nutrient reclamation",
                "detail": "Closed-loop systems can recapture up to 95% of water and nutrients",
                "mitigation": "Advanced filtration and UV sterilization for nutrient solution recycling",
                "impact_factor": 0.85
            }
        ],
        "optimal_parameters": {
            "aeroponic_droplet_size_microns": 50,
            "nutrient_solution_ec_ms_cm": 1.8,  # Electrical conductivity
            "solution_temperature_c": 21,
            "ph_fluctuation_range": [5.8, 6.2],
            "nutrient_recycling_rate_percent": 95
        }
    },
    "plant_species_selection": {
        "description": "Optimal plant species and varieties for space agriculture",
        "findings": [
            {
                "title": "Dwarf varieties",
                "detail": "Dwarf wheat, rice, and tomato varieties maximize yield per volume and minimize structural issues",
                "mitigation": "Select super-dwarf varieties for long-term space missions",
                "impact_factor": 0.85
            },
            {
                "title": "Fast-cycle crops",
                "detail": "Crops with cycle times under 60 days provide quicker returns on investment",
                "mitigation": "Focus on leafy greens, radishes, and selected microgreens for early harvest",
                "impact_factor": 0.8
            },
            {
                "title": "Nutritional density",
                "detail": "Space-efficient crops should maximize essential nutrients, especially antioxidants and vitamins",
                "mitigation": "Kale, sweet potato, pepper varieties show optimal nutrient-to-volume ratios",
                "impact_factor": 0.9
            },
            {
                "title": "Multi-harvest capabilities",
                "detail": "Plants allowing multiple harvests from same individual reduce resource expenditure",
                "mitigation": "Cut-and-come-again crops like lettuce and chard provide long-term yields",
                "impact_factor": 0.85
            }
        ],
        "optimal_parameters": {
            "dwarf_height_threshold_cm": 45,
            "days_to_harvest_threshold": 60,
            "minimum_harvest_index": 0.45,  # Ratio of edible to total biomass
            "vitamin_c_content_mg_per_100g": 25  # Minimum vitamin C content for nutritional benefits
        }
    },
    "environmental_control": {
        "description": "Advanced environmental control strategies for space agriculture",
        "findings": [
            {
                "title": "Variable light spectra",
                "detail": "Adjusting light spectrum during growth phases can improve yield by up to 30%",
                "mitigation": "Dynamic LED systems with phase-specific spectral optimization",
                "impact_factor": 0.9
            },
            {
                "title": "Circadian lighting",
                "detail": "Light cycles matching circadian rhythms improve growth and reduce stress",
                "mitigation": "16/8 hour cycles with dawn/dusk simulation through gradual intensity changes",
                "impact_factor": 0.8
            },
            {
                "title": "CO2 enrichment",
                "detail": "Elevated CO2 levels (800-1200 ppm) significantly increase photosynthetic efficiency",
                "mitigation": "Controlled CO2 injection with careful monitoring to prevent toxic levels",
                "impact_factor": 0.85
            },
            {
                "title": "Temperature differential",
                "detail": "Day/night temperature differential of 5-7°C improves metabolic efficiency",
                "mitigation": "Programmed temperature cycles with precise control systems",
                "impact_factor": 0.75
            }
        ],
        "optimal_parameters": {
            "light_intensity_ppfd": 300,  # μmol/m²/s for leafy greens (higher for fruiting)
            "co2_concentration_ppm": 1000,
            "day_night_temp_differential_c": 6,
            "relative_humidity_percent": 65,
            "air_circulation_m_per_s": 0.3  # Gentle air movement
        }
    }
}

class SpaceAgricultureKnowledgeBase:
    """Knowledge base for space agriculture research findings"""
    
    def __init__(self, research_data=None):
        """
        Initialize the knowledge base
        
        Args:
            research_data: Dictionary containing research findings (uses SPACE_AGRICULTURE_RESEARCH by default)
        """
        self.research_data = research_data if research_data else SPACE_AGRICULTURE_RESEARCH
        logger.info("Initialized Space Agriculture Knowledge Base")
        
    def get_domain_findings(self, domain):
        """
        Get research findings for a specific domain
        
        Args:
            domain: Research domain name
            
        Returns:
            Dictionary with findings for the domain
        """
        if domain in self.research_data:
            return self.research_data[domain]
        else:
            logger.warning(f"Domain {domain} not found in research data")
            return None
    
    def get_optimal_parameters(self, domain):
        """
        Get optimal parameters for a specific domain
        
        Args:
            domain: Research domain name
            
        Returns:
            Dictionary with optimal parameters
        """
        domain_data = self.get_domain_findings(domain)
        if domain_data and 'optimal_parameters' in domain_data:
            return domain_data['optimal_parameters']
        else:
            logger.warning(f"Optimal parameters not found for domain {domain}")
            return {}
    
    def get_impact_factor(self, domain, finding_title):
        """
        Get impact factor for a specific finding
        
        Args:
            domain: Research domain name
            finding_title: Title of the specific finding
            
        Returns:
            Impact factor (0-1) or None if not found
        """
        domain_data = self.get_domain_findings(domain)
        if domain_data and 'findings' in domain_data:
            for finding in domain_data['findings']:
                if finding['title'] == finding_title:
                    return finding['impact_factor']
        
        logger.warning(f"Impact factor not found for {finding_title} in {domain}")
        return None
    
    def enhance_environment_parameters(self, current_params, plant_species):
        """
        Enhance environment parameters based on research findings
        
        Args:
            current_params: Current environment parameters
            plant_species: Plant species name
            
        Returns:
            Enhanced parameters dictionary
        """
        # Start with current parameters
        enhanced_params = current_params.copy()
        
        # Enhance based on environmental control research
        env_control = self.get_optimal_parameters('environmental_control')
        if env_control:
            # Adjust temperature if needed
            if 'temperature' in enhanced_params and enhanced_params['temperature'] < 18:
                enhanced_params['temperature'] = min(
                    enhanced_params['temperature'] + 2, 
                    24  # Maximum reasonable temperature
                )
                logger.info(f"Adjusted temperature to {enhanced_params['temperature']}°C based on research")
            
            # Adjust CO2 if present in parameters
            if 'co2_level' in enhanced_params:
                enhanced_params['co2_level'] = max(
                    enhanced_params['co2_level'],
                    env_control['co2_concentration_ppm'] * 0.8  # 80% of optimal
                )
        
        # Adjust radiation protection based on research
        rad_params = self.get_optimal_parameters('radiation_effects')
        if rad_params and 'radiation_level' in enhanced_params:
            # Lower values are better for radiation
            if enhanced_params['radiation_level'] > 10:
                enhanced_params['radiation_level'] *= 0.8  # Reduce by 20%
                logger.info(f"Adjusted radiation level to {enhanced_params['radiation_level']} based on research")
        
        # Enhance nutrient delivery if parameters exist
        nutrient_params = self.get_optimal_parameters('nutrient_delivery')
        if nutrient_params and 'nutrient_level' in enhanced_params:
            # Adjust nutrient level based on research
            if enhanced_params['nutrient_level'] < 50:
                enhanced_params['nutrient_level'] = min(
                    enhanced_params['nutrient_level'] * 1.2,  # Increase by 20%
                    100  # Maximum value
                )
                logger.info(f"Adjusted nutrient level to {enhanced_params['nutrient_level']} based on research")
        
        return enhanced_params
    
    def calculate_research_reward_modifier(self, state, action, plant_species):
        """
        Calculate reward modifier based on research findings
        
        Args:
            state: Current environment state
            action: Action taken by the agent
            plant_species: Plant species name
            
        Returns:
            Reward modifier (multiplier for the environment reward)
        """
        base_modifier = 1.0
        modifiers = []
        
        # Check if action aligns with microgravity research
        if 'temperature' in state and 'temperature_action' in action:
            # Get optimal range
            env_params = self.get_optimal_parameters('environmental_control')
            if env_params:
                target_temp = 22  # Reasonable default
                if 'day_night_temp_differential_c' in env_params:
                    # Adjust based on time of day (simplified)
                    is_daytime = state.get('light_intensity', 0) > 500
                    differential = env_params['day_night_temp_differential_c'] / 2
                    target_temp = 22 + (differential if is_daytime else -differential)
                
                # Calculate how closely the action aligns with research
                temp_after_action = state['temperature'] + action['temperature_action']
                temp_alignment = 1.0 - min(abs(temp_after_action - target_temp) / 10.0, 1.0)
                modifiers.append(0.9 + (0.2 * temp_alignment))
        
        # Check radiation protection alignment
        if 'radiation_level' in state and 'radiation_action' in action:
            # Lower radiation is generally better
            if action['radiation_action'] < 0:
                modifiers.append(1.1)  # Reward reducing radiation
            else:
                modifiers.append(0.9)  # Penalize increasing radiation
        
        # Check light intensity alignment with research
        if 'light_intensity' in state and 'light_action' in action:
            env_params = self.get_optimal_parameters('environmental_control')
            if env_params and 'light_intensity_ppfd' in env_params:
                target_intensity = env_params['light_intensity_ppfd']
                light_after_action = state['light_intensity'] + action['light_action']
                
                # Convert our light scale to PPFD estimate (simplified)
                estimated_ppfd = light_after_action * 2  # Rough conversion
                
                # Calculate alignment
                light_alignment = 1.0 - min(abs(estimated_ppfd - target_intensity) / 200.0, 1.0)
                modifiers.append(0.9 + (0.2 * light_alignment))
        
        # Check water management alignment
        if 'water_content' in state and 'water_action' in action:
            # Get species-specific water requirements (simplified)
            optimal_water = 70  # Default optimal water content percentage
            
            water_after_action = state['water_content'] + action['water_action']
            water_alignment = 1.0 - min(abs(water_after_action - optimal_water) / 30.0, 1.0)
            modifiers.append(0.9 + (0.2 * water_alignment))
        
        # Apply modifiers
        if modifiers:
            # Combine modifiers (geometric mean)
            combined_modifier = np.prod(modifiers) ** (1.0 / len(modifiers))
            base_modifier = combined_modifier
        
        return base_modifier
    
    def generate_research_based_recommendations(self, current_state, plant_species):
        """
        Generate recommendations based on current state and research
        
        Args:
            current_state: Current environment state
            plant_species: Plant species name
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Check temperature conditions
        if 'temperature' in current_state:
            temp = current_state['temperature']
            env_params = self.get_optimal_parameters('environmental_control')
            if env_params:
                target_temp = 22  # Reasonable default
                if 'day_night_temp_differential_c' in env_params:
                    is_daytime = current_state.get('light_intensity', 0) > 500
                    differential = env_params['day_night_temp_differential_c'] / 2
                    target_temp = 22 + (differential if is_daytime else -differential)
                
                if abs(temp - target_temp) > 3:
                    recommendations.append({
                        "parameter": "temperature",
                        "current_value": temp,
                        "recommended_value": target_temp,
                        "explanation": f"Research indicates optimal temperature of {target_temp}°C for {'day' if is_daytime else 'night'} periods.",
                        "priority": "high" if abs(temp - target_temp) > 5 else "medium",
                        "research_domain": "environmental_control"
                    })
        
        # Check light conditions
        if 'light_intensity' in current_state:
            light = current_state['light_intensity']
            env_params = self.get_optimal_parameters('environmental_control')
            if env_params and 'light_intensity_ppfd' in env_params:
                target_intensity = env_params['light_intensity_ppfd'] / 2  # Convert PPFD to our scale
                
                if abs(light - target_intensity) > 100:
                    recommendations.append({
                        "parameter": "light_intensity",
                        "current_value": light,
                        "recommended_value": target_intensity,
                        "explanation": f"Research indicates optimal light intensity around {target_intensity} for improved photosynthetic efficiency.",
                        "priority": "high" if abs(light - target_intensity) > 200 else "medium",
                        "research_domain": "environmental_control"
                    })
        
        # Check radiation levels
        if 'radiation_level' in current_state:
            radiation = current_state['radiation_level']
            if radiation > 15:  # Threshold for high radiation
                recommendations.append({
                    "parameter": "radiation_level",
                    "current_value": radiation,
                    "recommended_value": max(radiation * 0.7, 5),  # Reduce by 30% but keep above 5
                    "explanation": "Research shows DNA damage and reduced photosynthetic efficiency at high radiation levels.",
                    "priority": "high" if radiation > 25 else "medium",
                    "research_domain": "radiation_effects"
                })
        
        # Check water content
        if 'water_content' in current_state:
            water = current_state['water_content']
            optimal_water = 70  # Default optimal
            
            if abs(water - optimal_water) > 15:
                recommendations.append({
                    "parameter": "water_content",
                    "current_value": water,
                    "recommended_value": optimal_water,
                    "explanation": f"Maintaining water content around {optimal_water}% improves nutrient uptake efficiency in microgravity.",
                    "priority": "high" if abs(water - optimal_water) > 25 else "medium",
                    "research_domain": "nutrient_delivery"
                })
        
        return recommendations
    
    def save_research_data(self, filename="space_agriculture_research.json"):
        """
        Save the research data to a JSON file
        
        Args:
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.research_data, f, indent=2)
            logger.info(f"Research data saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving research data: {str(e)}")
            return False
    
    @classmethod
    def load_research_data(cls, filename="space_agriculture_research.json"):
        """
        Load research data from a JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            SpaceAgricultureKnowledgeBase instance
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    research_data = json.load(f)
                logger.info(f"Research data loaded from {filename}")
                return cls(research_data)
            else:
                logger.warning(f"Research data file {filename} not found, using default data")
                return cls()
        except Exception as e:
            logger.error(f"Error loading research data: {str(e)}")
            return cls()

# For testing the module directly
if __name__ == "__main__":
    kb = SpaceAgricultureKnowledgeBase()
    
    # Print some research findings
    for domain in kb.research_data:
        print(f"\n---- {domain.upper()} ----")
        print(kb.research_data[domain]['description'])
        print("Key findings:")
        for finding in kb.research_data[domain]['findings']:
            print(f"- {finding['title']}: {finding['detail']} (Impact factor: {finding['impact_factor']})")
        
        if 'optimal_parameters' in kb.research_data[domain]:
            print("\nOptimal parameters:")
            for param, value in kb.research_data[domain]['optimal_parameters'].items():
                print(f"- {param}: {value}")
    
    # Test recommendation generation
    test_state = {
        'temperature': 18,
        'light_intensity': 400,
        'water_content': 50,
        'radiation_level': 25
    }
    
    recommendations = kb.generate_research_based_recommendations(test_state, "Dwarf Wheat")
    print("\n---- RECOMMENDATIONS ----")
    for rec in recommendations:
        print(f"[{rec['priority'].upper()}] {rec['parameter']}: {rec['current_value']} -> {rec['recommended_value']}")
        print(f"Explanation: {rec['explanation']}")
        print(f"Based on: {rec['research_domain']}")
        print()