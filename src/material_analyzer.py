"""
MaterialAnalyzer - Main class for analyzing condensed matter systems using AIFeynman.
"""

import numpy as np
import pandas as pd
from aifeynman import Feynman
from typing import Dict, List, Tuple, Optional
import torch
from pathlib import Path

class MaterialAnalyzer:
    """
    A class for analyzing condensed matter systems using AIFeynman.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the MaterialAnalyzer.
        
        Args:
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        self.feynman = Feynman(device=device)
        self.data = None
        self.discovered_laws = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load experimental or simulation data from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing the data
            
        Returns:
            DataFrame containing the loaded data
        """
        self.data = pd.read_csv(file_path)
        return self.data
    
    def prepare_data(self, target_column: str, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for AIFeynman analysis.
        
        Args:
            target_column: Name of the target variable column
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (features, target) arrays
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        X = self.data[feature_columns].values
        y = self.data[target_column].values
        
        return X, y
    
    def discover_laws(self, 
                     target_column: str,
                     feature_columns: List[str],
                     max_terms: int = 5,
                     max_operations: int = 10) -> Dict:
        """
        Discover physical laws using AIFeynman.
        
        Args:
            target_column: Name of the target variable column
            feature_columns: List of feature column names
            max_terms: Maximum number of terms in the discovered equation
            max_operations: Maximum number of operations in the discovered equation
            
        Returns:
            Dictionary containing discovered laws and their metrics
        """
        X, y = self.prepare_data(target_column, feature_columns)
        
        # Run AIFeynman analysis
        results = self.feynman.fit(X, y, 
                                 max_terms=max_terms,
                                 max_operations=max_operations)
        
        self.discovered_laws = results
        return results
    
    def evaluate_laws(self, test_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Evaluate the discovered laws on test data.
        
        Args:
            test_data: Optional test data DataFrame. If None, uses the loaded data.
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.discovered_laws is None:
            raise ValueError("No laws discovered yet. Run discover_laws first.")
            
        data = test_data if test_data is not None else self.data
        
        # Implement evaluation logic here
        metrics = {
            'r2_score': None,
            'mse': None,
            'mae': None
        }
        
        return metrics
    
    def save_results(self, output_dir: str):
        """
        Save discovered laws and analysis results.
        
        Args:
            output_dir: Directory to save results in
        """
        if self.discovered_laws is None:
            raise ValueError("No laws discovered yet. Run discover_laws first.")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save discovered laws
        with open(output_path / 'discovered_laws.txt', 'w') as f:
            for law in self.discovered_laws:
                f.write(f"{law}\n")
                
        # Save metrics
        metrics = self.evaluate_laws()
        pd.DataFrame([metrics]).to_csv(output_path / 'metrics.csv', index=False) 