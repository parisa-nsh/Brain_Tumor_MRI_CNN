import pytest
import numpy as np
import tensorflow as tf
from src.data.preprocessing import BrainMRIPreprocessor

def test_preprocessor_creation():
    """Test if preprocessor can be created with default config"""
    config = BrainMRIPreprocessor.get_default_config()
    preprocessor = BrainMRIPreprocessor(config)
    assert preprocessor is not None

def test_image_preprocessing():
    """Test if image preprocessing works correctly"""
    config = BrainMRIPreprocessor.get_default_config()
    preprocessor = BrainMRIPreprocessor(config)
    
    # Create dummy image
    dummy_image = np.random.rand(256, 256, 3)
    
    # Process image
    processed_image = preprocessor.preprocess_image(dummy_image)
    
    # Check output shape and values
    assert processed_image.shape == (224, 224, 3)  # Default size in config
    assert processed_image.dtype == np.float32
    assert np.max(processed_image) <= 1.0
    assert np.min(processed_image) >= 0.0

def test_dataset_creation():
    """Test if dataset creation works correctly"""
    config = BrainMRIPreprocessor.get_default_config()
    preprocessor = BrainMRIPreprocessor(config)
    
    # Create dummy data
    num_samples = 5
    image_paths = ["dummy_path.jpg"] * num_samples
    labels = [0] * num_samples
    
    # Mock the image reading and processing
    def mock_load_and_preprocess(image_path, label):
        return np.zeros((224, 224, 3), dtype=np.float32), tf.one_hot(label, depth=4)
    
    # Patch the load_and_preprocess function
    preprocessor.load_and_preprocess = mock_load_and_preprocess
    
    # Create dataset
    dataset = preprocessor.create_dataset(image_paths, labels, is_training=True)
    
    # Check if dataset is created correctly
    assert isinstance(dataset, tf.data.Dataset)
    
    # Check batch size
    for images, labels in dataset.take(1):
        assert images.shape[0] == config["batch_size"] or images.shape[0] == num_samples
        assert labels.shape[0] == config["batch_size"] or labels.shape[0] == num_samples 