"""
Example usage of the CondensedMatterAI package.
"""

import os
from src import MaterialAnalyzer, DataGenerator, Visualizer

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Initialize components
    data_gen = DataGenerator()
    analyzer = MaterialAnalyzer()
    visualizer = Visualizer()
    
    # Generate Ising model data
    print("Generating Ising model data...")
    ising_data = data_gen.generate_ising_model_data(size=1000)
    data_gen.save_data(ising_data, 'output/ising_data.csv')
    
    # Analyze Ising model data
    print("Analyzing Ising model data...")
    analyzer.load_data('output/ising_data.csv')
    laws = analyzer.discover_laws(
        target_column='magnetization',
        feature_columns=['temperature', 'field']
    )
    
    # Visualize results
    print("Visualizing results...")
    visualizer.plot_phase_diagram(
        ising_data,
        x_col='temperature',
        y_col='field',
        z_col='magnetization',
        title='Ising Model Phase Diagram',
        save_path='output/phase_diagram.png'
    )
    
    visualizer.plot_correlation_heatmap(
        ising_data,
        title='Ising Model Correlations',
        save_path='output/correlation_heatmap.png'
    )
    
    visualizer.plot_discovered_laws(
        laws=analyzer.discovered_laws,
        save_path='output/discovered_laws.png'
    )
    
    # Save analysis results
    print("Saving analysis results...")
    analyzer.save_results('output/results')
    
    print("Analysis complete! Results saved in the 'output' directory.")

if __name__ == "__main__":
    main() 