"""
Tests for the MaterialAnalyzer class.
"""

import pytest
import numpy as np
import pandas as pd
from src.material_analyzer import MaterialAnalyzer
from src.data_generator import DataGenerator

@pytest.fixture
def analyzer():
    return MaterialAnalyzer()

@pytest.fixture
def sample_data():
    data_gen = DataGenerator()
    return data_gen.generate_ising_model_data(size=100)

def test_initialization(analyzer):
    """Test MaterialAnalyzer initialization."""
    assert analyzer.data is None
    assert analyzer.discovered_laws is None

def test_load_data(analyzer, sample_data):
    """Test data loading functionality."""
    # Save sample data temporarily
    sample_data.to_csv('test_data.csv', index=False)
    
    # Test loading data
    loaded_data = analyzer.load_data('test_data.csv')
    assert isinstance(loaded_data, pd.DataFrame)
    assert not loaded_data.empty
    assert analyzer.data is not None
    
    # Clean up
    import os
    os.remove('test_data.csv')

def test_prepare_data(analyzer, sample_data):
    """Test data preparation functionality."""
    analyzer.data = sample_data
    X, y = analyzer.prepare_data(
        target_column='magnetization',
        feature_columns=['temperature', 'field']
    )
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)

def test_discover_laws(analyzer, sample_data):
    """Test law discovery functionality."""
    analyzer.data = sample_data
    laws = analyzer.discover_laws(
        target_column='magnetization',
        feature_columns=['temperature', 'field']
    )
    
    assert isinstance(laws, dict)
    assert analyzer.discovered_laws is not None

def test_evaluate_laws(analyzer, sample_data):
    """Test law evaluation functionality."""
    analyzer.data = sample_data
    analyzer.discover_laws(
        target_column='magnetization',
        feature_columns=['temperature', 'field']
    )
    
    metrics = analyzer.evaluate_laws()
    assert isinstance(metrics, dict)
    assert 'r2_score' in metrics
    assert 'mse' in metrics
    assert 'mae' in metrics

def test_save_results(analyzer, sample_data, tmp_path):
    """Test results saving functionality."""
    analyzer.data = sample_data
    analyzer.discover_laws(
        target_column='magnetization',
        feature_columns=['temperature', 'field']
    )
    
    output_dir = tmp_path / "results"
    analyzer.save_results(str(output_dir))
    
    assert (output_dir / 'discovered_laws.txt').exists()
    assert (output_dir / 'metrics.csv').exists() 