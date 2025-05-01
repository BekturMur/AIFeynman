"""
MaterialAnalyzer - Main class for analyzing condensed matter systems using AIFeynman.
"""

import numpy as np
import pandas as pd
import aifeynman
from typing import Dict, List, Tuple, Optional
import torch
from pathlib import Path
import os

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
                     features: np.ndarray,
                     target: np.ndarray,
                     feature_names: List[str] = None,
                     target_name: str = None,
                     max_terms: int = 5,
                     max_operations: int = 10) -> Dict:
        """
        Discover physical laws using AIFeynman.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            target: Target variable array (n_samples,)
            feature_names: Optional list of feature names
            target_name: Optional target variable name
            max_terms: Maximum number of terms in the discovered equation
            max_operations: Maximum number of operations in the discovered equation
            
        Returns:
            Dictionary containing discovered laws and their metrics
        """
        # Create results directory if it doesn't exist
        results_dir = os.path.abspath('results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save data in AIFeynman format
        data = np.column_stack((features, target))
        data_file = os.path.join(results_dir, 'input.txt')
        np.savetxt(data_file, data, delimiter='\t')
        
        # Remember current directory
        current_dir = os.getcwd()
        
        try:
            # Change to results directory
            os.chdir(results_dir)
            
            # Run AIFeynman analysis
            results = aifeynman.run_aifeynman(
                'input.txt',
                '.',
                BF_try_time=300,
                BF_ops_file_type='simple',
                polyfit_deg=2
            )
            
            # Parse the results file
            results_file = os.path.join(results_dir, 'results_pareto.txt')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.discovered_laws = f.readlines()
                    
                # Format the laws with variable names if provided
                if feature_names and target_name:
                    formatted_laws = []
                    for law in self.discovered_laws:
                        formatted_law = law
                        for i, name in enumerate(feature_names):
                            formatted_law = formatted_law.replace(f'x{i}', name)
                        formatted_law = formatted_law.replace('y', target_name)
                        formatted_laws.append(formatted_law)
                    self.discovered_laws = formatted_laws
            else:
                self.discovered_laws = ["No laws discovered"]
            
        except Exception as e:
            print(f"Error during AIFeynman analysis: {str(e)}")
            self.discovered_laws = ["Error during analysis"]
            
        finally:
            # Change back to original directory
            os.chdir(current_dir)
            
            # Clean up temporary file
            if os.path.exists(data_file):
                os.unlink(data_file)
        
        return self.discovered_laws
    
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