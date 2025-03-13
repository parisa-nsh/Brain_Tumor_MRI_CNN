import os
import sys
import hydra
from omegaconf import DictConfig
import mlflow
import logging
from pathlib import Path
from typing import Tuple, List
import tensorflow as tf

# Add src directory to python path
sys.path.append(str(Path(__file__).parent.parent))

from model.model import BrainTumorClassifier
from data.preprocessing import BrainMRIPreprocessor

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and evaluation with MLflow tracking"""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer with configuration
        
        Args:
            config: Hydra configuration object containing:
                - data: Data configuration
                - model: Model configuration
                - training: Training configuration
                - mlflow: MLflow configuration
        """
        self.config = config
        self._setup_mlflow()
        
        # Initialize preprocessor and model
        self.preprocessor = BrainMRIPreprocessor(config.data)
        self.model = BrainTumorClassifier(config.model)
        
        # Setup callbacks
        self.callbacks = self._get_callbacks()
        
    def _setup_mlflow(self) -> None:
        """Sets up MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        
    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Creates training callbacks"""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.training.checkpoint_dir, "model_{epoch:02d}.h5"),
                save_best_only=True,
                monitor="val_accuracy"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        return callbacks
        
    def _load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Loads and preprocesses training and validation data"""
        try:
            # Here you would implement your data loading logic
            # For example, reading from a data directory:
            train_data_dir = Path(self.config.data.train_data_dir)
            val_data_dir = Path(self.config.data.val_data_dir)
            
            # Get image paths and labels
            train_images = list(train_data_dir.glob("*/*.jpg"))
            train_labels = [str(p.parent.name) for p in train_images]
            
            val_images = list(val_data_dir.glob("*/*.jpg"))
            val_labels = [str(p.parent.name) for p in val_images]
            
            # Convert labels to indices
            label_to_idx = {label: idx for idx, label in enumerate(self.config.data.class_names)}
            train_labels = [label_to_idx[label] for label in train_labels]
            val_labels = [label_to_idx[label] for label in val_labels]
            
            # Create datasets
            train_dataset = self.preprocessor.create_dataset(train_images, train_labels, is_training=True)
            val_dataset = self.preprocessor.create_dataset(val_images, val_labels, is_training=False)
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def train(self) -> None:
        """Trains the model and logs metrics with MLflow"""
        try:
            # Load data
            train_dataset, val_dataset = self._load_data()
            
            # Start MLflow run
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    "model_type": self.config.model.model_type,
                    "learning_rate": self.config.model.learning_rate,
                    "batch_size": self.config.data.batch_size,
                    "epochs": self.config.training.epochs
                })
                
                # Train model
                history = self.model.get_model().fit(
                    train_dataset,
                    epochs=self.config.training.epochs,
                    validation_data=val_dataset,
                    callbacks=self.callbacks,
                    verbose=1
                )
                
                # Log metrics
                for epoch in range(len(history.history["loss"])):
                    mlflow.log_metrics({
                        "train_loss": history.history["loss"][epoch],
                        "train_accuracy": history.history["accuracy"][epoch],
                        "val_loss": history.history["val_loss"][epoch],
                        "val_accuracy": history.history["val_accuracy"][epoch]
                    }, step=epoch)
                
                # Save model
                model_path = os.path.join(self.config.training.model_dir, "final_model.h5")
                self.model.get_model().save(model_path)
                mlflow.log_artifact(model_path)
                
                logger.info("Training completed successfully")
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

@hydra.main(config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function
    
    Args:
        config: Hydra configuration
    """
    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 