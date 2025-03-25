"""
Advanced Biosensor System for Space Agriculture
Implementation of multi-modal sensing and biofeedback loops
"""

import numpy as np
import logging
import json
import random
from datetime import datetime

logger = logging.getLogger('SpaceAgriRL.Biosensors')

class SensorDevice:
    """Base class for sensor devices"""
    
    def __init__(self, sensor_id, sensor_type, calibration_params=None):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.calibration_params = calibration_params or {}
        self.last_reading = None
        self.reading_history = []
        self.last_calibration = datetime.now()
        
    def read(self, raw_input=None):
        """Read sensor data with calibration"""
        # Generate simulated reading if no input provided
        if raw_input is None:
            raw_input = self._generate_simulated_reading()
        
        # Apply calibration
        calibrated_reading = self._apply_calibration(raw_input)
        
        # Store reading
        self.last_reading = calibrated_reading
        self.reading_history.append({
            'timestamp': datetime.now(),
            'value': calibrated_reading
        })
        
        # Limit history size
        if len(self.reading_history) > 1000:
            self.reading_history = self.reading_history[-1000:]
        
        return calibrated_reading
    
    def _generate_simulated_reading(self):
        """Generate a simulated sensor reading"""
        # Override in subclasses
        return 0.0
    
    def _apply_calibration(self, raw_value):
        """Apply calibration parameters to raw sensor value"""
        # Basic linear calibration
        offset = self.calibration_params.get('offset', 0.0)
        scale = self.calibration_params.get('scale', 1.0)
        
        return (raw_value + offset) * scale
    
    def calibrate(self, reference_values):
        """Calibrate sensor using reference values"""
        # Simple calibration - just store offsets
        if reference_values:
            # Measure current raw values
            current_values = [self._generate_simulated_reading() for _ in range(5)]
            current_avg = sum(current_values) / len(current_values)
            
            # Calculate offset
            reference_avg = sum(reference_values) / len(reference_values)
            offset = reference_avg - current_avg
            
            # Update calibration parameters
            self.calibration_params['offset'] = offset
            self.last_calibration = datetime.now()
            
            return {
                'status': 'success',
                'new_offset': offset,
                'reference_avg': reference_avg,
                'current_avg': current_avg
            }
        
        return {'status': 'error', 'message': 'No reference values provided'}


class ChlorophyllSensor(SensorDevice):
    """Sensor for measuring chlorophyll fluorescence"""
    
    def __init__(self, sensor_id="chloro_1"):
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="chlorophyll_fluorescence",
            calibration_params={
                'offset': 0.0,
                'scale': 1.0,
                'wavelength': 685  # nm
            }
        )
    
    def _generate_simulated_reading(self):
        """Generate simulated chlorophyll fluorescence reading"""
        # Baseline value
        baseline = 0.72  # Typical Fv/Fm value for healthy plant
        
        # Add random variation
        variation = np.random.normal(0, 0.05)
        
        # Return simulated reading
        return max(0, min(1, baseline + variation))
    
    def calculate_photosynthesis_efficiency(self):
        """Calculate photosynthetic efficiency from fluorescence"""
        if not self.last_reading:
            return None
        
        # Convert Fv/Fm to quantum yield (Simplified model)
        efficiency = self.last_reading * 0.8
        
        # Classify efficiency
        if efficiency > 0.7:
            status = "excellent"
        elif efficiency > 0.5:
            status = "good"
        elif efficiency > 0.3:
            status = "fair"
        else:
            status = "poor"
        
        return {
            'efficiency': efficiency,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }


class MetaboliteTracker(SensorDevice):
    """Sensor for tracking plant metabolites"""
    
    def __init__(self, sensor_id="metab_1"):
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="metabolite_tracker",
            calibration_params={
                'offset': 0.0,
                'scale': 1.0,
                'detection_threshold': 0.01  # μmol/L
            }
        )
        
        # Initialize metabolite profiles
        self.metabolite_profiles = {
            'stress_markers': {
                'proline': 0.0,
                'abscisic_acid': 0.0,
                'reactive_oxygen_species': 0.0
            },
            'growth_markers': {
                'auxin': 0.0,
                'cytokinin': 0.0,
                'gibberellin': 0.0
            },
            'defense_markers': {
                'salicylic_acid': 0.0,
                'jasmonic_acid': 0.0,
                'ethylene': 0.0
            }
        }
    
    def _generate_simulated_reading(self):
        """Generate simulated metabolite readings"""
        # Update all metabolite levels with some random variation
        for category in self.metabolite_profiles:
            for metabolite in self.metabolite_profiles[category]:
                # Get current value
                current = self.metabolite_profiles[category][metabolite]
                
                # Add random variation
                variation = np.random.normal(0, 0.05)
                
                # Update value (keep between 0 and 1)
                self.metabolite_profiles[category][metabolite] = max(0, min(1, current + variation))
        
        # Return average stress marker level as the main reading
        stress_markers = self.metabolite_profiles['stress_markers']
        avg_stress = sum(stress_markers.values()) / len(stress_markers)
        
        return avg_stress
    
    def get_stress_markers(self):
        """Get current stress marker levels"""
        # Ensure we have a reading
        if self.last_reading is None:
            self.read()
        
        # Return all stress markers
        stress_markers = self.metabolite_profiles['stress_markers']
        
        # Calculate stress index (0-100)
        stress_index = sum(stress_markers.values()) / len(stress_markers) * 100
        
        return {
            'markers': stress_markers,
            'stress_index': stress_index,
            'interpretation': self._interpret_stress_level(stress_index)
        }
    
    def _interpret_stress_level(self, stress_index):
        """Interpret stress index value"""
        if stress_index < 20:
            return "minimal_stress"
        elif stress_index < 40:
            return "low_stress"
        elif stress_index < 60:
            return "moderate_stress"
        elif stress_index < 80:
            return "high_stress"
        else:
            return "severe_stress"


class QuantumDotStressDetector(SensorDevice):
    """Quantum dot nanosensor for cellular stress detection"""
    
    def __init__(self, sensor_id="qdot_1"):
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="quantum_dot_stress_detector",
            calibration_params={
                'offset': 0.0,
                'scale': 1.0,
                'fluorescence_threshold': 0.2
            }
        )
        
        # Initialize stress response signatures
        self.stress_signatures = {
            'heat_shock_response': 0.0,
            'osmotic_stress': 0.0,
            'oxidative_stress': 0.0,
            'nutrient_deficiency': 0.0,
            'cellular_damage': 0.0
        }
    
    def _generate_simulated_reading(self):
        """Generate simulated quantum dot readings"""
        # Update all stress signatures with some random variation
        for stress_type in self.stress_signatures:
            # Get current value
            current = self.stress_signatures[stress_type]
            
            # Add random variation
            variation = np.random.normal(0, 0.03)
            
            # Update value (keep between 0 and 1)
            self.stress_signatures[stress_type] = max(0, min(1, current + variation))
        
        # Return maximum stress signature as the main reading
        return max(self.stress_signatures.values())
    
    def detect_stress_patterns(self):
        """Detect specific stress patterns from quantum dot signals"""
        # Ensure we have a reading
        if self.last_reading is None:
            self.read()
        
        # Find dominant stress type
        dominant_stress = max(self.stress_signatures.items(), key=lambda x: x[1])
        
        # Calculate overall stress severity
        severity = self.last_reading * 100
        
        return {
            'stress_signatures': self.stress_signatures,
            'dominant_stress': dominant_stress[0],
            'severity': severity,
            'stress_level': self._stress_level_classification(severity),
            'timestamp': datetime.now().isoformat()
        }
    
    def _stress_level_classification(self, severity):
        """Classify stress severity"""
        if severity < 20:
            return "minimal"
        elif severity < 40:
            return "low"
        elif severity < 60:
            return "moderate"
        elif severity < 80:
            return "high"
        else:
            return "extreme"


class HyperspectralImager(SensorDevice):
    """Hyperspectral imaging sensor"""
    
    def __init__(self, sensor_id="hyper_1"):
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="hyperspectral_imager",
            calibration_params={
                'offset': 0.0,
                'scale': 1.0,
                'wavelength_range': (400, 2500),  # nm
                'spectral_resolution': 10  # nm
            }
        )
        
        # Initialize spectral signatures for different plant characteristics
        self.spectral_signatures = {
            'chlorophyll': [],
            'water_content': [],
            'nutrient_status': [],
            'disease_markers': [],
            'stress_indicators': []
        }
        
        # Generate initial spectral data
        self._initialize_spectral_data()
    
    def _initialize_spectral_data(self):
        """Initialize spectral data with realistic signatures"""
        # Wavelength range and resolution
        wl_min = self.calibration_params['wavelength_range'][0]
        wl_max = self.calibration_params['wavelength_range'][1]
        resolution = self.calibration_params['spectral_resolution']
        
        # Generate wavelength bands
        wavelengths = list(range(wl_min, wl_max, resolution))
        
        # Generate spectra for each signature type
        for signature_type in self.spectral_signatures:
            # Create empty spectrum
            spectrum = []
            
            # Fill with realistic values for each wavelength
            for wl in wavelengths:
                # Different peak wavelengths for different signatures
                if signature_type == 'chlorophyll' and (650 < wl < 700 or 400 < wl < 500):
                    # Chlorophyll absorption peaks
                    value = 0.8 + np.random.normal(0, 0.05)
                elif signature_type == 'water_content' and 1400 < wl < 1900:
                    # Water absorption bands
                    value = 0.7 + np.random.normal(0, 0.05)
                elif signature_type == 'nutrient_status' and 700 < wl < 1300:
                    # NIR plateau for leaf structure
                    value = 0.6 + np.random.normal(0, 0.05)
                elif signature_type == 'disease_markers' and 500 < wl < 800:
                    # Visible to NIR transition
                    value = 0.5 + np.random.normal(0, 0.05)
                elif signature_type == 'stress_indicators' and 530 < wl < 570:
                    # Green peak reflectance
                    value = 0.4 + np.random.normal(0, 0.05)
                else:
                    # Background reflectance
                    value = 0.2 + np.random.normal(0, 0.05)
                
                # Clip to valid range
                value = max(0, min(1, value))
                spectrum.append(value)
            
            # Store spectrum
            self.spectral_signatures[signature_type] = spectrum
    
    def _generate_simulated_reading(self):
        """Generate simulated hyperspectral reading"""
        # Choose a random signature type
        signature_type = random.choice(list(self.spectral_signatures.keys()))
        
        # Get that spectrum
        spectrum = self.spectral_signatures[signature_type]
        
        # Add random variation to the spectrum
        varied_spectrum = [max(0, min(1, val + np.random.normal(0, 0.03))) for val in spectrum]
        
        # Update the stored spectrum
        self.spectral_signatures[signature_type] = varied_spectrum
        
        # Return average value as the main reading
        return sum(varied_spectrum) / len(varied_spectrum)
    
    def detect_stress_patterns(self):
        """Detect plant stress patterns from hyperspectral data"""
        # Ensure we have data
        if self.last_reading is None:
            self.read()
        
        # Calculate stress indices from hyperspectral data
        # These would be complex algorithms in a real implementation
        # Simplified for demonstration
        
        # Photochemical Reflectance Index (PRI)
        stress_indicators = self.spectral_signatures['stress_indicators']
        mid_point = len(stress_indicators) // 2
        pri = (stress_indicators[mid_point-5] - stress_indicators[mid_point+5]) / (stress_indicators[mid_point-5] + stress_indicators[mid_point+5])
        
        # Normalized Difference Vegetation Index (NDVI)
        chlorophyll = self.spectral_signatures['chlorophyll']
        red_band = sum(chlorophyll[:len(chlorophyll)//3]) / (len(chlorophyll)//3)
        nir_band = sum(chlorophyll[len(chlorophyll)*2//3:]) / (len(chlorophyll)//3)
        ndvi = (nir_band - red_band) / (nir_band + red_band) if (nir_band + red_band) > 0 else 0
        
        # Water Band Index (WBI)
        water_content = self.spectral_signatures['water_content']
        mid_point = len(water_content) // 2
        wbi = water_content[mid_point-10] / water_content[mid_point+10] if water_content[mid_point+10] > 0 else 1
        
        return {
            'indices': {
                'PRI': pri,
                'NDVI': ndvi,
                'WBI': wbi
            },
            'stress_level': self._calculate_stress_level(pri, ndvi, wbi),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_stress_level(self, pri, ndvi, wbi):
        """Calculate overall stress level from spectral indices"""
        # Convert indices to stress indicators (0-1 scale)
        pri_stress = abs(pri) * 2  # Higher absolute PRI indicates stress
        ndvi_stress = 1 - ndvi  # Lower NDVI indicates stress
        wbi_stress = abs(wbi - 1)  # WBI further from 1 indicates water stress
        
        # Combine stress indicators (weighted average)
        overall_stress = (pri_stress * 0.3 + ndvi_stress * 0.5 + wbi_stress * 0.2)
        
        # Classify stress level
        if overall_stress < 0.2:
            return "minimal"
        elif overall_stress < 0.4:
            return "low"
        elif overall_stress < 0.6:
            return "moderate"
        elif overall_stress < 0.8:
            return "high"
        else:
            return "severe"


class TerahertzScanner(SensorDevice):
    """Terahertz scanner for non-invasive tissue analysis"""
    
    def __init__(self, sensor_id="thz_1"):
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="terahertz_scanner",
            calibration_params={
                'offset': 0.0,
                'scale': 1.0,
                'frequency_range': (0.1, 10.0),  # THz
                'resolution': 0.01  # THz
            }
        )
        
        # Initialize tissue analysis parameters
        self.tissue_parameters = {
            'water_distribution': [],
            'cell_wall_integrity': [],
            'internal_structure': []
        }
        
        # Initialize parameter data
        self._initialize_parameter_data()
    
    def _initialize_parameter_data(self):
        """Initialize tissue parameter data"""
        # Frequency range and resolution
        freq_min = self.calibration_params['frequency_range'][0]
        freq_max = self.calibration_params['frequency_range'][1]
        resolution = self.calibration_params['resolution']
        
        # Generate frequency points
        frequencies = list(np.arange(freq_min, freq_max, resolution))
        
        # Generate data for each parameter
        for param in self.tissue_parameters:
            # Create empty data array
            data = []
            
            # Fill with realistic values for each frequency
            for freq in frequencies:
                # Different frequency responses for different parameters
                if param == 'water_distribution':
                    # Water has strong absorption at higher THz frequencies
                    if freq > 5.0:
                        value = 0.8 + np.random.normal(0, 0.05)
                    else:
                        value = 0.4 + np.random.normal(0, 0.05)
                elif param == 'cell_wall_integrity':
                    # Cell wall features are more prominent at mid-range
                    if 2.0 < freq < 6.0:
                        value = 0.7 + np.random.normal(0, 0.05)
                    else:
                        value = 0.3 + np.random.normal(0, 0.05)
                elif param == 'internal_structure':
                    # Internal structure visible across range
                    value = 0.5 + 0.2 * np.sin(freq) + np.random.normal(0, 0.05)
                else:
                    value = 0.2 + np.random.normal(0, 0.05)
                
                # Clip to valid range
                value = max(0, min(1, value))
                data.append(value)
            
            # Store data
            self.tissue_parameters[param] = data
    
    def _generate_simulated_reading(self):
        """Generate simulated terahertz scan"""
        # Choose a random parameter
        param = random.choice(list(self.tissue_parameters.keys()))
        
        # Get that parameter's data
        data = self.tissue_parameters[param]
        
        # Add random variation
        varied_data = [max(0, min(1, val + np.random.normal(0, 0.02))) for val in data]
        
        # Update the stored data
        self.tissue_parameters[param] = varied_data
        
        # Return average value as the main reading
        return sum(varied_data) / len(varied_data)
    
    def analyze_tissue_integrity(self):
        """Analyze tissue integrity from terahertz scan data"""
        # Ensure we have data
        if self.last_reading is None:
            self.read()
        
        # Calculate integrity metrics
        water_distribution = self.tissue_parameters['water_distribution']
        cell_wall_data = self.tissue_parameters['cell_wall_integrity']
        structure_data = self.tissue_parameters['internal_structure']
        
        # Water content uniformity (lower variance = more uniform)
        water_variance = np.var(water_distribution)
        water_uniformity = max(0, min(1, 1 - water_variance * 5))
        
        # Cell wall integrity (average value)
        cell_integrity = sum(cell_wall_data) / len(cell_wall_data)
        
        # Internal structure coherence (correlation between adjacent points)
        coherence_sum = sum(abs(structure_data[i] - structure_data[i-1]) for i in range(1, len(structure_data)))
        structural_coherence = max(0, min(1, 1 - coherence_sum / len(structure_data)))
        
        # Overall integrity score
        integrity_score = (water_uniformity * 0.3 + cell_integrity * 0.4 + structural_coherence * 0.3) * 100
        
        return {
            'metrics': {
                'water_uniformity': water_uniformity,
                'cell_integrity': cell_integrity,
                'structural_coherence': structural_coherence
            },
            'integrity_score': integrity_score,
            'status': self._classify_integrity(integrity_score),
            'timestamp': datetime.now().isoformat()
        }
    
    def _classify_integrity(self, score):
        """Classify tissue integrity score"""
        if score < 40:
            return "poor"
        elif score < 60:
            return "fair"
        elif score < 80:
            return "good"
        else:
            return "excellent"


class PiezoelectricStressGauge(SensorDevice):
    """Piezoelectric sensor for cellular wall stress monitoring"""
    
    def __init__(self, sensor_id="piezo_1"):
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="piezoelectric_stress_gauge",
            calibration_params={
                'offset': 0.0,
                'scale': 1.0,
                'sensitivity': 0.05  # mV/Pa
            }
        )
        
        # Initialize stress readings
        self.strain_readings = {
            'turgor_pressure': 0.0,  # MPa
            'mechanical_strain': 0.0,  # %
            'cellular_elasticity': 0.0  # MPa^-1
        }
    
    def _generate_simulated_reading(self):
        """Generate simulated piezoelectric stress reading"""
        # Update stress parameters with random variation
        
        # Turgor pressure (0.5-1.0 MPa for healthy cells)
        current_pressure = self.strain_readings['turgor_pressure']
        if current_pressure == 0:
            # Initialize to a normal value
            current_pressure = 0.75
        pressure_variation = np.random.normal(0, 0.05)
        self.strain_readings['turgor_pressure'] = max(0.1, min(1.2, current_pressure + pressure_variation))
        
        # Mechanical strain (0-5% normally)
        current_strain = self.strain_readings['mechanical_strain']
        strain_variation = np.random.normal(0, 0.2)
        self.strain_readings['mechanical_strain'] = max(0, min(10, current_strain + strain_variation))
        
        # Cellular elasticity (inverse of Young's modulus, higher = more elastic)
        current_elasticity = self.strain_readings['cellular_elasticity']
        if current_elasticity == 0:
            # Initialize to a normal value
            current_elasticity = 0.5
        elasticity_variation = np.random.normal(0, 0.03)
        self.strain_readings['cellular_elasticity'] = max(0.1, min(1.0, current_elasticity + elasticity_variation))
        
        # Calculate overall stress as a function of these parameters
        # Higher strain and lower turgor pressure indicate stress
        stress = (self.strain_readings['mechanical_strain'] / 10) * (1 - self.strain_readings['turgor_pressure'] / 1.2)
        
        return stress
    
    def measure_strain(self):
        """Measure cellular strain from piezoelectric data"""
        # Ensure we have data
        if self.last_reading is None:
            self.read()
        
        # Calculate strain metrics
        turgor_pressure = self.strain_readings['turgor_pressure']  # MPa
        mechanical_strain = self.strain_readings['mechanical_strain']  # %
        elasticity = self.strain_readings['cellular_elasticity']  # MPa^-1
        
        # Calculate stress-strain relationship
        # In a real system, this would be based on complex mechanics
        stress = mechanical_strain / elasticity if elasticity > 0 else 0
        
        # Determine if plant is experiencing drought stress based on turgor pressure
        drought_stress = turgor_pressure < 0.6
        
        # Determine if plant is experiencing mechanical stress
        mechanical_stress = mechanical_strain > 3.0
        
        # Calculate overall strain index (0-100)
        strain_index = (stress * 10) + (5 * (1 - turgor_pressure / 1.2) * 10)
        strain_index = max(0, min(100, strain_index))
        
        return {
            'readings': self.strain_readings,
            'calculated': {
                'stress': stress,
                'strain_index': strain_index
            },
            'stress_indicators': {
                'drought_stress': drought_stress,
                'mechanical_stress': mechanical_stress
            },
            'strain_status': self._classify_strain(strain_index),
            'timestamp': datetime.now().isoformat()
        }
    
    def _classify_strain(self, strain_index):
        """Classify strain index"""
        if strain_index < 20:
            return "minimal"
        elif strain_index < 40:
            return "low"
        elif strain_index < 60:
            return "moderate"
        elif strain_index < 80:
            return "high"
        else:
            return "severe"


class PlantBiosensorSystem:
    """Comprehensive plant biosensor system with multiple sensing modalities"""
    
    def __init__(self, plant_species):
        self.plant_species = plant_species
        
        # Initialize molecular sensors
        self.molecular_sensors = {
            'chlorophyll_fluorescence': ChlorophyllSensor(),
            'metabolite_tracker': MetaboliteTracker(),
            'stress_detector': QuantumDotStressDetector()
        }
        
        # Initialize imaging sensors
        self.imaging_sensors = {
            'hyperspectral_camera': HyperspectralImager(),
            'terahertz_scanner': TerahertzScanner(),
            'cellular_stress_monitor': PiezoelectricStressGauge()
        }
        
        # Sensor readings history
        self.readings_history = []
        
        # Intervention history
        self.intervention_history = []
    
    def collect_multi_modal_data(self, plant_state=None):
        """
        Aggregate data from multiple sensor types
        
        Args:
            plant_state: Optional dictionary of plant state for more realistic simulations
            
        Returns:
            Dictionary of biosensor data
        """
        # If plant state is provided, use it to influence sensor readings
        if plant_state:
            # Adjust sensor readings based on plant state
            self._set_sensor_baselines_from_state(plant_state)
        
        # Collect data from all sensors
        biosensor_data = {
            'photosynthetic_efficiency': self.calculate_photosynthesis_efficiency(),
            'stress_metabolites': self.molecular_sensors['metabolite_tracker'].get_stress_markers(),
            'cellular_strain': self.imaging_sensors['cellular_stress_monitor'].measure_strain(),
            'spectral_stress_signature': self.imaging_sensors['hyperspectral_camera'].detect_stress_patterns()
        }
        
        # Add tissue integrity data
        biosensor_data['tissue_integrity'] = self.imaging_sensors['terahertz_scanner'].analyze_tissue_integrity()
        
        # Add molecular stress detection
        biosensor_data['molecular_stress'] = self.molecular_sensors['stress_detector'].detect_stress_patterns()
        
        # Calculate combined stress index
        biosensor_data['combined_stress_index'] = self._calculate_combined_stress(biosensor_data)
        
        # Store reading history
        self.readings_history.append({
            'timestamp': datetime.now().isoformat(),
            'biosensor_data': biosensor_data
        })
        
        # Limit history size
        if len(self.readings_history) > 100:
            self.readings_history = self.readings_history[-100:]
        
        return biosensor_data
    
    def _set_sensor_baselines_from_state(self, plant_state):
        """Set sensor baselines based on plant state"""
        if not plant_state:
            return
        
        # Extract relevant state parameters
        health_score = plant_state.get('health_score', 0.9)
        temperature = plant_state.get('temperature', 22.0)
        water_content = plant_state.get('water_content', 70.0)
        radiation_level = plant_state.get('radiation_level', 20.0)
        
        # Adjust chlorophyll fluorescence based on health
        cf_sensor = self.molecular_sensors['chlorophyll_fluorescence']
        cf_sensor.calibration_params['offset'] = (health_score - 0.9) * 0.2
        
        # Adjust metabolite tracking based on temperature stress
        met_sensor = self.molecular_sensors['metabolite_tracker']
        temp_stress = abs(temperature - 22.0) / 10.0  # 0-1 scale for temp stress
        
        # Update stress metabolites
        met_sensor.metabolite_profiles['stress_markers']['proline'] = temp_stress * 0.8
        met_sensor.metabolite_profiles['stress_markers']['abscisic_acid'] = (1 - water_content / 100) * 0.7
        met_sensor.metabolite_profiles['stress_markers']['reactive_oxygen_species'] = radiation_level / 100 * 0.9
        
        # Adjust quantum dot stress detector based on radiation
        qd_sensor = self.molecular_sensors['stress_detector']
        qd_sensor.stress_signatures['oxidative_stress'] = radiation_level / 100 * 0.8
        qd_sensor.stress_signatures['heat_shock_response'] = temp_stress * 0.7
        
        # Adjust cellular stress monitor based on water content
        cs_sensor = self.imaging_sensors['cellular_stress_monitor']
        cs_sensor.strain_readings['turgor_pressure'] = water_content / 100 * 1.2
    
    def calculate_photosynthesis_efficiency(self):
        """Calculate photosynthetic efficiency from chlorophyll fluorescence"""
        return self.molecular_sensors['chlorophyll_fluorescence'].calculate_photosynthesis_efficiency()
    
    def trigger_adaptive_interventions(self, biosensor_data, available_interventions=None):
        """
        Use multi-modal data to trigger precise interventions
        
        Args:
            biosensor_data: Comprehensive sensor data from collect_multi_modal_data
            available_interventions: Dictionary of available intervention types
            
        Returns:
            Dictionary of intervention recommendations
        """
        # Default interventions if none provided
        if available_interventions is None:
            available_interventions = {
                'light_spectrum': True,
                'nutrient_delivery': True,
                'humidity_control': True,
                'temperature_control': True
            }
        
        # Extract stress indicators
        combined_stress = biosensor_data.get('combined_stress_index', {'stress_level': 50, 'primary_stressor': 'unknown'})
        stress_level = combined_stress['stress_level']
        primary_stressor = combined_stress['primary_stressor']
        
        # Generate intervention recommendations
        intervention_recommendations = {
            'required_interventions': [],
            'optional_interventions': [],
            'intervention_priority': 'low'
        }
        
        # Set intervention priority based on stress level
        if stress_level > 70:
            intervention_recommendations['intervention_priority'] = 'high'
        elif stress_level > 40:
            intervention_recommendations['intervention_priority'] = 'medium'
        
        # Generate specific interventions based on primary stressor
        if primary_stressor == 'temperature_stress' and available_interventions.get('temperature_control', False):
            # Check if it's heat or cold stress
            temperature_direction = biosensor_data.get('spectral_stress_signature', {}).get('temperature_direction', 'unknown')
            
            if temperature_direction == 'high':
                intervention_recommendations['required_interventions'].append({
                    'type': 'temperature_control',
                    'action': 'decrease',
                    'magnitude': min(1.0, stress_level / 100 * 1.5)
                })
            else:
                intervention_recommendations['required_interventions'].append({
                    'type': 'temperature_control',
                    'action': 'increase',
                    'magnitude': min(1.0, stress_level / 100 * 1.5)
                })
        
        elif primary_stressor == 'water_stress' and available_interventions.get('nutrient_delivery', False):
            # Water stress intervention
            intervention_recommendations['required_interventions'].append({
                'type': 'nutrient_delivery',
                'action': 'increase_water',
                'magnitude': min(1.0, stress_level / 100 * 1.8)
            })
        
        elif primary_stressor == 'light_stress' and available_interventions.get('light_spectrum', False):
            # Light stress could be too much or too little
            light_direction = biosensor_data.get('spectral_stress_signature', {}).get('light_direction', 'unknown')
            
            if light_direction == 'high':
                intervention_recommendations['required_interventions'].append({
                    'type': 'light_spectrum',
                    'action': 'decrease_intensity',
                    'magnitude': min(1.0, stress_level / 100 * 1.2)
                })
            else:
                intervention_recommendations['required_interventions'].append({
                    'type': 'light_spectrum',
                    'action': 'increase_intensity',
                    'magnitude': min(1.0, stress_level / 100 * 1.2)
                })
        
        elif primary_stressor == 'nutrient_stress' and available_interventions.get('nutrient_delivery', False):
            # Nutrient stress intervention
            intervention_recommendations['required_interventions'].append({
                'type': 'nutrient_delivery',
                'action': 'adjust_nutrients',
                'magnitude': min(1.0, stress_level / 100),
                'specific_nutrients': self._identify_deficient_nutrients(biosensor_data)
            })
        
        # Add humidity control as an optional intervention if plant is water stressed
        if (stress_level > 30 and 
            'water_stress' in [primary_stressor, combined_stress.get('secondary_stressor', '')] and 
            available_interventions.get('humidity_control', False)):
            
            intervention_recommendations['optional_interventions'].append({
                'type': 'humidity_control',
                'action': 'increase',
                'magnitude': min(1.0, stress_level / 100)
            })
        
        # Store intervention recommendation
        self.intervention_history.append({
            'timestamp': datetime.now().isoformat(),
            'stress_level': stress_level,
            'primary_stressor': primary_stressor,
            'recommendations': intervention_recommendations
        })
        
        # Limit history size
        if len(self.intervention_history) > 100:
            self.intervention_history = self.intervention_history[-100:]
        
        return intervention_recommendations
    
    def _calculate_combined_stress(self, biosensor_data):
        """Calculate combined stress index from all sensors"""
        # Extract individual stress metrics
        photosynthetic_efficiency = biosensor_data.get('photosynthetic_efficiency', {}).get('efficiency', 0.8)
        stress_metabolites = biosensor_data.get('stress_metabolites', {}).get('stress_index', 20)
        cellular_strain = biosensor_data.get('cellular_strain', {}).get('calculated', {}).get('strain_index', 20)
        spectral_stress = self._convert_stress_level_to_value(
            biosensor_data.get('spectral_stress_signature', {}).get('stress_level', 'minimal')
        )
        tissue_integrity = biosensor_data.get('tissue_integrity', {}).get('integrity_score', 80)
        molecular_stress_level = self._convert_stress_level_to_value(
            biosensor_data.get('molecular_stress', {}).get('stress_level', 'minimal')
        )
        
        # Convert photosynthetic efficiency to stress (0-100, 100 = high stress)
        photosynthetic_stress = (1 - photosynthetic_efficiency) * 100
        
        # Convert tissue integrity to stress (0-100, 100 = high stress)
        tissue_stress = 100 - tissue_integrity
        
        # Calculate weighted average (adjust weights based on reliability/importance)
        stress_values = [
            photosynthetic_stress * 0.2,
            stress_metabolites * 0.2,
            cellular_strain * 0.15,
            spectral_stress * 0.15,
            tissue_stress * 0.15,
            molecular_stress_level * 0.15
        ]
        
        combined_stress = sum(stress_values)
        
        # Determine primary stressor (highest contribution to stress)
        stress_contributions = [
            ('photosynthetic_stress', photosynthetic_stress * 0.2),
            ('water_stress', cellular_strain * 0.15),
            ('metabolic_stress', stress_metabolites * 0.2),
            ('light_stress', spectral_stress * 0.15),
            ('structural_stress', tissue_stress * 0.15),
            ('molecular_stress', molecular_stress_level * 0.15)
        ]
        
        sorted_contributions = sorted(stress_contributions, key=lambda x: x[1], reverse=True)
        primary_stressor = sorted_contributions[0][0]
        secondary_stressor = sorted_contributions[1][0] if len(sorted_contributions) > 1 else None
        
        # Determine temperature or water stress direction
        temp_direction = 'unknown'
        light_direction = 'unknown'
        
        # Add temperature and light direction to spectral data
        if 'spectral_stress_signature' in biosensor_data and 'indices' in biosensor_data['spectral_stress_signature']:
            indices = biosensor_data['spectral_stress_signature']['indices']
            
            # Negative PRI can indicate excess light
            if 'PRI' in indices:
                light_direction = 'high' if indices['PRI'] < -0.2 else 'low'
            
            # Add to biosensor data
            biosensor_data['spectral_stress_signature']['light_direction'] = light_direction
            biosensor_data['spectral_stress_signature']['temperature_direction'] = temp_direction
        
        return {
            'stress_level': combined_stress,
            'primary_stressor': primary_stressor,
            'secondary_stressor': secondary_stressor,
            'stress_contributions': dict(sorted_contributions)
        }
    
    def _convert_stress_level_to_value(self, stress_level):
        """Convert text stress level to numeric value (0-100)"""
        stress_map = {
            'minimal': 10,
            'low': 30,
            'moderate': 50,
            'high': 70,
            'severe': 90,
            'extreme': 100
        }
        
        return stress_map.get(stress_level, 50)
    
    def _identify_deficient_nutrients(self, biosensor_data):
        """Identify specific nutrient deficiencies from sensor data"""
        # This would use complex spectral signatures in a real system
        # Simplified random implementation for demonstration
        
        # List of possible nutrients
        nutrients = ['nitrogen', 'phosphorus', 'potassium', 'calcium', 'magnesium', 'sulfur', 'iron']
        
        # Randomly select 1-3 "deficient" nutrients
        num_deficient = random.randint(1, 3)
        deficient_nutrients = random.sample(nutrients, num_deficient)
        
        # Create deficiency scores (0-1)
        deficiencies = {}
        for nutrient in deficient_nutrients:
            deficiencies[nutrient] = random.uniform(0.3, 0.8)
        
        return deficiencies