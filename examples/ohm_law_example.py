import numpy as np
from src.material_analyzer import MaterialAnalyzer
from src.data_generator import DataGenerator
from src.visualizer import Visualizer

def generate_ohm_law_data():
    """Generate synthetic data for Ohm's law analysis."""
    # Generate random resistances (R) from 1 to 100 ohms
    resistances = np.random.uniform(1, 100, 5000)
    
    # Generate random currents (I) from 0.1 to 10 amperes
    currents = np.random.uniform(0.1, 10, 5000)
    
    # Calculate voltages using Ohm's law: V = I * R
    voltages = currents * resistances
    
    # Add very small noise to make it more realistic
    noise = np.random.normal(0, 0.01 * voltages, len(voltages))
    voltages += noise
    
    # Create feature matrix
    features = np.column_stack((currents, resistances))
    
    return features, voltages

def main():
    # Initialize components
    analyzer = MaterialAnalyzer()
    visualizer = Visualizer()
    
    # Generate Ohm's law data
    features, target = generate_ohm_law_data()
    
    # Analyze data with AIFeynman
    print("Starting AIFeynman analysis...")
    analyzer.discover_laws(features, target, 
                          feature_names=['Current (A)', 'Resistance (Ω)'],
                          target_name='Voltage (V)')
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer.plot_correlation_heatmap(features, target,
                                      feature_names=['Current (A)', 'Resistance (Ω)'],
                                      target_name='Voltage (V)',
                                      output_file='output/ohm_law_correlation.png')
    
    # Plot 3D visualization of the data
    visualizer.plot_3d_surface(features, target,
                             feature_names=['Current (A)', 'Resistance (Ω)'],
                             target_name='Voltage (V)',
                             output_file='output/ohm_law_surface.png')

if __name__ == "__main__":
    main() 