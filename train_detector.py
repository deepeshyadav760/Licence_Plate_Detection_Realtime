import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ultralytics import YOLO
import cv2

def convert_to_yolo_format(df, img_dir, output_dir):
    """
    Converts a pandas DataFrame with bounding box info to the YOLOv5 TXT format.
    - Creates a .txt file for each image.
    - Each line in the txt file is: <class_index> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    """
    # We only have one class: "license_plate"
    class_index = 0
    
    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting to YOLO format"):
        img_id = row['img_id']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        
        img_path = os.path.join(img_dir, img_id)
        
        # We need image dimensions to normalize the coordinates
        try:
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
        except Exception as e:
            print(f"Warning: Could not read image {img_path}. Skipping. Error: {e}")
            continue

        # Calculate YOLO coordinates
        dw = 1. / img_width
        dh = 1. / img_height
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        x_center_norm = x_center * dw
        y_center_norm = y_center * dh
        width_norm = width * dw
        height_norm = height * dh
        
        # Create the label file path
        label_filename = os.path.splitext(img_id)[0] + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        # Write the YOLO formatted line to the txt file
        with open(label_path, 'w') as f:
            f.write(f"{class_index} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

def train_yolo_detector():
    """
    Main function to prepare data and train the YOLOv8 detector.
    """
    # --- 1. Load Data ---
    df = pd.read_csv("licplatesdetection_train.csv")
    img_dir = "data/train"

    # --- 2. Create YOLO Directory Structure ---
    # This structure is required by the ultralytics library
    base_dir = "data/yolo_dataset"
    train_img_path = os.path.join(base_dir, "images/train")
    val_img_path = os.path.join(base_dir, "images/val")
    train_label_path = os.path.join(base_dir, "labels/train")
    val_label_path = os.path.join(base_dir, "labels/val")

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)
    print("YOLO directory structure created.")

    # --- 3. Split Data into Training and Validation Sets ---
    # We use a 80/20 split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Data split: {len(train_df)} training samples, {len(val_df)} validation samples.")

    # --- 4. Convert and Move Data ---
    # Convert training data
    convert_to_yolo_format(train_df, img_dir, train_label_path)
    # Convert validation data
    convert_to_yolo_format(val_df, img_dir, val_label_path)

    # Copy image files to the new YOLO structure
    print("Copying images to YOLO structure...")
    for img_id in tqdm(train_df['img_id'], desc="Copying train images"):
        os.system(f'copy "{os.path.join(img_dir, img_id)}" "{os.path.join(train_img_path, img_id)}"')
    for img_id in tqdm(val_df['img_id'], desc="Copying val images"):
        os.system(f'copy "{os.path.join(img_dir, img_id)}" "{os.path.join(val_img_path, img_id)}"')

    # --- 5. Create the YAML dataset configuration file ---
    # This file tells YOLO where to find the data
    yaml_data = {
        'train': os.path.abspath(train_img_path),
        'val': os.path.abspath(val_img_path),
        'nc': 1,  # Number of classes
        'names': ['license_plate']  # List of class names
    }

    yaml_path = os.path.join(base_dir, "license_plate_dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"YAML configuration file created at {yaml_path}")

    # --- 6. Train the YOLOv8 Model ---
    # Load a pre-trained model (yolov8n.pt is small and fast)
    model = YOLO('yolov8n.pt') 

    print("Starting YOLOv8 model training...")
    # Train the model
    results = model.train(
        data=yaml_path,        # Path to our YAML file
        epochs=50,             # Number of training epochs (50 is a good start)
        imgsz=640,             # Image size for training
        batch=8,               # Batch size
        name='yolov8_license_plate_detector' # Name for the training run
    )
    print("Training complete!")
    print("Your trained model is saved in the 'runs/detect/yolov8_license_plate_detector/' directory.")
    print("The best model is usually named 'best.pt'.")

if __name__ == '__main__':
    train_yolo_detector()