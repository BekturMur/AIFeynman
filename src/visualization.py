"""
Visualizer - Class for visualizing results from condensed matter analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import seaborn as sns
import os

class Visualizer:
    """
    A class for visualizing results from condensed matter analysis.
    """
    
    def __init__(self):
        """Initialize the Visualizer."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_phase_diagram(self, 
                         data: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         z_col: str,
                         title: str = "Phase Diagram",
                         save_path: Optional[str] = None):
        """
        Plot a 2D phase diagram.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            z_col: Column name for color/intensity
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data[x_col], data[y_col], c=data[z_col], cmap='viridis')
        plt.colorbar(scatter, label=z_col)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_band_structure(self,
                          data: pd.DataFrame,
                          k_col: str = 'k_point',
                          energy_col: str = 'energy',
                          title: str = "Band Structure",
                          save_path: Optional[str] = None):
        """
        Plot electronic band structure.
        
        Args:
            data: DataFrame containing the data
            k_col: Column name for k-points
            energy_col: Column name for energies
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data[k_col], data[energy_col], 'b-', linewidth=2)
        plt.xlabel('k-point')
        plt.ylabel('Energy')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_heatmap(self,
                               data: pd.DataFrame,
                               title: str = "Correlation Heatmap",
                               save_path: Optional[str] = None):
        """
        Plot correlation heatmap of variables.
        
        Args:
            data: DataFrame containing the data
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        correlation = data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_discovered_laws(self, laws: List[str], save_path: Optional[str] = None) -> None:
        """Plot the discovered physical laws.
        
        Args:
            laws: List of discovered laws
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        if laws and isinstance(laws, list):
            plt.text(0.1, 0.5, "\n".join([f"Law {i+1}: {law.strip()}" for i, law in enumerate(laws)]),
                    fontsize=12, verticalalignment='center')
        else:
            plt.text(0.1, 0.5, "No laws discovered", fontsize=12, verticalalignment='center')
        
        plt.title("Discovered Physical Laws")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
    def plot_error_analysis(self,
                          predictions: np.ndarray,
                          actual: np.ndarray,
                          title: str = "Error Analysis",
                          save_path: Optional[str] = None):
        """
        Plot error analysis of predictions.
        
        Args:
            predictions: Array of predicted values
            actual: Array of actual values
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predictions, alpha=0.5)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 