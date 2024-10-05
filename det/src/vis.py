import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import random
import glob


def visulaization(image_folder, label_folder, num_images):

    """
    Visualize random samples from both training and validation sets, showing:

    1. The original images
    2. Bounding boxes around detected objects
    3. Green boxes for people with helmets
    4. Red boxes for people without helmets
    """

    
