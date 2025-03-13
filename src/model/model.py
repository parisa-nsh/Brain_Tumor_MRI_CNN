import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BrainTumorClassifier:
    """Brain Tumor Classification Model
    
    A CNN model for classifying brain MRI scans into different tumor categories.
    The architecture is based on the paper [insert paper reference] with modifications
    for improved performance and stability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with given configuration
        
        Args:
            config: Dictionary containing model configuration
                - input_shape: Tuple of (height, width, channels)
                - num_classes: Number of tumor categories
                - dropout_rate: Dropout rate for regularization
                - learning_rate: Initial learning rate
        """
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> models.Model:
        """Builds and returns the CNN model architecture"""
        try:
            inputs = layers.Input(shape=self.config["input_shape"])
            
            # First Convolutional Block
            x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(self.config["dropout_rate"])(x)
            
            # Second Convolutional Block
            x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(self.config["dropout_rate"])(x)
            
            # Third Convolutional Block
            x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(self.config["dropout_rate"])(x)
            
            # Fourth Convolutional Block
            x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(self.config["dropout_rate"])(x)
            
            # Dense Layers
            x = layers.Flatten()(x)
            x = layers.Dense(512, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.config["dropout_rate"])(x)
            
            # Output Layer
            outputs = layers.Dense(self.config["num_classes"], activation="softmax")(x)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"])
            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC()]
            )
            
            logger.info("Model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def get_model(self) -> models.Model:
        """Returns the compiled Keras model"""
        return self.model
    
    def summary(self) -> None:
        """Prints model summary"""
        self.model.summary()
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Returns default model configuration"""
        return {
            "input_shape": (224, 224, 3),
            "num_classes": 4,
            "dropout_rate": 0.3,
            "learning_rate": 0.001
        }

if __name__ == "__main__":
    # Example usage
    config = BrainTumorClassifier.get_default_config()
    model = BrainTumorClassifier(config)
    model.summary() 