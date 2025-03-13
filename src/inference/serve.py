import sys
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import io
from PIL import Image
import logging
import hydra
from omegaconf import DictConfig
import mlflow
from datetime import datetime

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import BrainMRIPreprocessor

# Setup logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Classification API",
    description="API for classifying brain tumor MRI images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelServer:
    """Handles model serving and prediction"""
    
    def __init__(self, config: DictConfig):
        """Initialize the model server
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.preprocessor = BrainMRIPreprocessor(config.data)
        self.model = self._load_model()
        self.class_names = config.data.class_names
        
    def _load_model(self) -> tf.keras.Model:
        """Loads the trained model"""
        try:
            model = tf.keras.models.load_model(self.config.serving.model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    async def predict(self, image: UploadFile) -> dict:
        """Make prediction on input image
        
        Args:
            image: Uploaded image file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Read and preprocess image
            contents = await image.read()
            image_array = np.array(Image.open(io.BytesIO(contents)))
            
            # Convert grayscale to RGB if needed
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            # Preprocess image
            processed_image = self.preprocessor.preprocess_image(image_array)
            processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            prediction = self.model.predict(processed_image)
            predicted_class = self.class_names[np.argmax(prediction[0])]
            confidence = float(np.max(prediction[0]))
            
            # Log prediction
            self._log_prediction(predicted_class, confidence)
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, prediction[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def _log_prediction(self, prediction: str, confidence: float) -> None:
        """Log prediction details to MLflow
        
        Args:
            prediction: Predicted class
            confidence: Prediction confidence
        """
        try:
            with mlflow.start_run(run_name="prediction"):
                mlflow.log_params({
                    "prediction": prediction,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")

# Global model server instance
model_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize model server on startup"""
    global model_server
    with hydra.initialize(config_path="../../configs"):
        config = hydra.compose(config_name="config")
        model_server = ModelServer(config)
    logger.info("Model server initialized")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        result = await model_server.predict(file)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    with hydra.initialize(config_path="../../configs"):
        config = hydra.compose(config_name="config")
        uvicorn.run(
            "serve:app",
            host=config.serving.host,
            port=config.serving.port,
            reload=True
        ) 