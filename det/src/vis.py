import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import glob


def visualization(image_folder, label_folder, num_images = 9):

    """
    Visualize random samples from both training and validation sets, showing:

    1. The original images
    2. Bounding boxes around detected objects
    3. Green boxes for people with helmets
    4. Red boxes for people without helmets
    """

    image_files  = list(Path(image_folder).glob('*.jpg')) + list(Path(image_folder).glob('*png'))
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    fig,axes = plt.subplots(3,3, figsize = (15,15))
    axes = axes.ravel()

    class_colors = {
        0: (255, 0, 0),    # Red for head
        1: (0, 255, 0),    # Green for helmet
        2: (0, 0, 255)     # Blue for person
    }

    class_names = {
        0: 'head',
        1: 'helmet',
        2: 'person'
    }

    for idx, img_path in enumerate(selected_images):
        if idx >= num_images:
            break
            
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get corresponding label file
        label_file = Path(label_folder) / f"{img_path.stem}.txt"
        
        height, width = img.shape[:2]
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                class_id = int(class_id)
                
                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)
                
                color = class_colors[class_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add label text
                label = class_names[class_id]
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"Image {idx+1}")
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                 markerfacecolor=color, markersize=10, label=name)
                      for class_id, (color, name) in enumerate(zip(class_colors.values(), class_names.values()))]
    fig.legend(handles=legend_elements, loc='center right')
    
    plt.tight_layout()
    return fig

def analyze_dataset(image_folder, label_folder):
    class_counts = {0: 0, 1: 0, 2: 0}  # Initialize counts for head, helmet, person
    class_names = {0: 'head', 1: 'helmet', 2: 'person'}
    total_images = 0
    unlabeled_images = 0
    
    image_files = list(Path(image_folder).glob('*.jpg')) + list(Path(image_folder).glob('*.png'))
    
    for img_path in image_files:
        total_images += 1
        label_file = Path(label_folder) / f"{img_path.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
        else:
            unlabeled_images += 1
    
    return class_counts, class_names, total_images, unlabeled_images

def visualize_dataset(train_img_path, train_label_path, val_img_path, val_label_path):
    # Analyze training data
    train_class_counts, class_names, train_total, train_unlabeled = analyze_dataset(train_img_path, train_label_path)
    
    # Analyze validation data
    val_class_counts, _, val_total, val_unlabeled = analyze_dataset(val_img_path, val_label_path)
    
    print("Dataset Analysis:")
    print(f"\nTraining Set:")
    print(f"Total images: {train_total}")
    for class_id, count in train_class_counts.items():
        print(f"{class_names[class_id]}: {count} instances")
    print(f"Unlabeled images: {train_unlabeled}")
    
    print(f"\nValidation Set:")
    print(f"Total images: {val_total}")
    for class_id, count in val_class_counts.items():
        print(f"{class_names[class_id]}: {count} instances")
    print(f"Unlabeled images: {val_unlabeled}")
    
    # Visualize training images
    print("\nGenerating visualization of training images...")
    train_fig = visualization(train_img_path, train_label_path)
    
    # Visualize validation images
    print("Generating visualization of validation images...")
    val_fig = visualization(val_img_path, val_label_path)
    
    return train_fig, val_fig


def main():

    train_img_path = r'det/data/train/images'
    train_label_path = r'det/data/train/labels'
    val_img_path = r'det/data/valid/images'
    val_label_path = r'det/data/valid/labels'
    
    train_fig, val_fig = visualize_dataset(train_img_path, train_label_path, 
                                          val_img_path, val_label_path)
    plt.show()

if __name__ == "__main__":
    main()
