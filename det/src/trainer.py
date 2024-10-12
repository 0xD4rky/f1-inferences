from ultralytics import YOLO
import yaml
from pathlib import Path
import logging 
from typing import Dict, Optional, Union, List

class YOLO_trainer:

    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 yaml_path: str,
                 model_size: str = 'm',
                 project_name: str = 'safety'):
        
        """
        Initialize the YOLO Helmet Detection trainer.
        
        Args:
            train_path (str): Path to training data
            val_path (str): Path to validation data
            test_path (str): Path to test data
            model_size (str): YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            project_name (str): Name for the training project
        """

        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.yaml_path = Path(yaml_path)
        self.model_size = model_size
        self.project_name = project_name

        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.yaml_path = None
        self.model = None
        self.results = None

        self.class_names = {
            0: 'head',
            1: 'helmet',
            2: 'person'
        }
    
    def initialize(self) -> None:

        """Initialize the YOLOv8 model."""
        try:
            self.model = YOLO(f'yolov8{self.model_size}.pt')
            self.logger.info(f"Initialized YOLOv8{self.model_size} model")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
    
    def train(self, 
              epochs: int = 100,
              imgsz: int = 640,
              batch_size: int = 16,
              **kwargs) -> None:
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Input image size
            batch_size (int): Batch size
            **kwargs: Additional arguments to pass to the YOLO trainer
        """

        if self.model is None:
            self.initialize()

        try:

            self.results = self.model.train(
                data = str(self.yaml_path),
                epochs = epochs,
                imgsz = imgsz,
                batch = batch_size,
                name = self.project_name,
                **kwargs
            )
            self.logger.info("Training completed successfully")
        except Exception as e:
            self.logger.error(f'Error during training: {e}')
            raise

    def validate(self) -> Optional[Dict]:
        "validates the training model"
        if self.model is None:
            self.logger.error("No trained model available. Please train a model first")
            return None
        
        try:
            metrics = self.model.eval()
            self.logger.info("Validation completed successfully!")
            return metrics
        except Exception as e:

            self.logger.error(f"Error during validation: {e}")
            raise
    
    def export_model(self, format: str = 'onnx') -> Path:
        """
        Export the trained model to a specific format.
        
        Args:
            format (str): Format to export to ('onnx', 'torchscript', etc.)
        
        Returns:
            Path: Path to the exported model
        """
        if self.model is None:
            self.logger.error("No trained model available. Please train first.")
            return None
        
        try:
            exported_model = self.model.export(format=format)
            self.logger.info(f"Model exported successfully in {format} format")
            return exported_model
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            raise
    
    def predict(self, 
                source: Union[str, Path, List[str]], 
                conf: float = 0.25) -> None:
        """
        Run prediction on images or video.
        
        Args:
            source: Source for prediction (image/video path or directory)
            conf: Confidence threshold
        """
        if self.model is None:
            self.logger.error("No trained model available. Please train first.")
            return None
        
        try:
            results = self.model.predict(source=source, conf=conf)
            self.logger.info("Prediction completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise


        

