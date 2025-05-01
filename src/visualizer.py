import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn')
    
    def plot_correlation_heatmap(self, features, target, feature_names, target_name, output_file):
        """Plot correlation heatmap between features and target."""
        # Combine features and target
        data = np.column_stack((features, target))
        column_names = feature_names + [target_name]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f',
                   xticklabels=column_names,
                   yticklabels=column_names,
                   cmap='coolwarm',
                   center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def plot_3d_surface(self, features, target, feature_names, target_name, output_file):
        """Plot 3D surface of the data."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for surface plot
        x = features[:, 0]
        y = features[:, 1]
        z = target
        
        # Plot scatter points
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
        
        # Add labels
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(target_name)
        
        # Add colorbar
        plt.colorbar(scatter, label=target_name)
        
        plt.title('3D Surface Plot')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close() 