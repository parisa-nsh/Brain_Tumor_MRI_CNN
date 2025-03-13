import pytest
import tensorflow as tf
from src.model.model import BrainTumorClassifier

def test_model_creation():
    """Test if model can be created with default config"""
    config = BrainTumorClassifier.get_default_config()
    model = BrainTumorClassifier(config)
    assert model is not None
    assert model.model is not None

def test_model_output_shape():
    """Test if model outputs correct shape"""
    config = BrainTumorClassifier.get_default_config()
    model = BrainTumorClassifier(config)
    
    # Create dummy input
    batch_size = 1
    input_shape = config["input_shape"]
    dummy_input = tf.random.normal([batch_size, *input_shape])
    
    # Get prediction
    output = model.model(dummy_input)
    
    # Check output shape
    expected_shape = (batch_size, config["num_classes"])
    assert output.shape == expected_shape

def test_model_compile():
    """Test if model compiles successfully"""
    config = BrainTumorClassifier.get_default_config()
    model = BrainTumorClassifier(config)
    
    # Model should be compiled in __init__
    assert model.model.optimizer is not None
    assert model.model.loss is not None 