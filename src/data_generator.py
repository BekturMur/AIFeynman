"""
DataGenerator - Class for generating synthetic data for condensed matter systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class DataGenerator:
    """
    A class for generating synthetic data for condensed matter systems.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the DataGenerator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
    def generate_ising_model_data(self, 
                                size: int = 1000,
                                temperature_range: Tuple[float, float] = (0.1, 4.0),
                                field_range: Tuple[float, float] = (-2.0, 2.0)) -> pd.DataFrame:
        """
        Generate data for the Ising model.
        
        Args:
            size: Number of data points to generate
            temperature_range: Range of temperatures to sample from
            field_range: Range of magnetic fields to sample from
            
        Returns:
            DataFrame containing generated data
        """
        temperatures = np.random.uniform(*temperature_range, size)
        fields = np.random.uniform(*field_range, size)
        
        # Calculate magnetization using mean-field approximation
        magnetizations = np.tanh((fields + 1.0) / temperatures)
        
        data = pd.DataFrame({
            'temperature': temperatures,
            'field': fields,
            'magnetization': magnetizations
        })
        
        return data
    
    def generate_band_structure_data(self,
                                   size: int = 1000,
                                   k_range: Tuple[float, float] = (-np.pi, np.pi),
                                   t_range: Tuple[float, float] = (0.5, 2.0)) -> pd.DataFrame:
        """
        Generate data for electronic band structure.
        
        Args:
            size: Number of data points to generate
            k_range: Range of k-points to sample from
            t_range: Range of hopping parameters to sample from
            
        Returns:
            DataFrame containing generated data
        """
        k_points = np.random.uniform(*k_range, size)
        hopping = np.random.uniform(*t_range, size)
        
        # Calculate energy using tight-binding model
        energies = -2 * hopping * np.cos(k_points)
        
        data = pd.DataFrame({
            'k_point': k_points,
            'hopping': hopping,
            'energy': energies
        })
        
        return data
    
    def generate_superconductor_data(self,
                                   size: int = 1000,
                                   temperature_range: Tuple[float, float] = (0.1, 10.0),
                                   gap_range: Tuple[float, float] = (0.1, 2.0)) -> pd.DataFrame:
        """
        Generate data for superconducting properties.
        
        Args:
            size: Number of data points to generate
            temperature_range: Range of temperatures to sample from
            gap_range: Range of gap parameters to sample from
            
        Returns:
            DataFrame containing generated data
        """
        temperatures = np.random.uniform(*temperature_range, size)
        gaps = np.random.uniform(*gap_range, size)
        
        # Calculate critical temperature using BCS theory
        critical_temps = 1.14 * gaps
        
        data = pd.DataFrame({
            'temperature': temperatures,
            'gap': gaps,
            'critical_temperature': critical_temps
        })
        
        return data
    
    def save_data(self, data: pd.DataFrame, file_path: str):
        """
        Save generated data to a CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Path to save the data
        """
        data.to_csv(file_path, index=False) 