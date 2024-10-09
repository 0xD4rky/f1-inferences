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
                bathc = batch_size,
                name = self.project_name,
                **kwargs
            )
            self.logger.info("Training completed successfully")
        except Exception as e:
            self.logger.error(f'Error during training: {e}')
            raise

        
        

