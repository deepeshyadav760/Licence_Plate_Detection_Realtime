import os
import cv2
import torch
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

# --- We need the same CRNN class and functions from our other scripts ---
from train import CRNN, CHARS, IMG_WIDTH, IMG_HEIGHT, LicensePlateDataset
from detect import decode_text

def run_evaluation():
    """
    Evaluates the full detection and recognition pipeline on the training dataset
    and calculates the final accuracy.
    """
    print("Starting evaluation on the training dataset...")
    
    # --- 1. Load Model and Detector ---
    model = CRNN(len(CHARS) + 1)
    try:
        model.load_state_dict(torch.load("models\crnn_augmented.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("ERROR: Could not find 'models/crnn.pth'. Please run train.py first.")
        return
    model.eval()
    print("Trained CRNN model loaded.")

    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    print("Haar Cascade detector loaded.")

    # --- 2. Load Training Data (which includes the ground truth labels) ---
    detection_df = pd.read_csv("licplatesdetection_train.csv")
    recognition_df = pd.read_csv("licplatesrecognition_train.csv")
    df = pd.merge(detection_df, recognition_df, on="img_id")
    
    train_image_dir = "data/train"
    
    correct_predictions = 0
    total_images = len(df)
    
    print(f"Evaluating on {total_images} images...")
    
    # --- 3. Process Each Training Image ---
    for index, row in tqdm(df.iterrows(), total=total_images, desc="Evaluating"):
        img_name = row['img_id']
        true_label = row['text']
        img_path = os.path.normpath(os.path.join(train_image_dir, img_name))

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect plates using the same lenient parameters
        plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
        
        predicted_text = ""
        if len(plates) > 0:
            # Assume the first detected plate is the correct one
            x, y, w, h = plates[0]
            plate_roi = gray_frame[y:y+h, x:x+w]
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                transforms.ToTensor(),
            ])
            plate_tensor = transform(plate_roi).unsqueeze(0)
            
            with torch.no_grad():
                output = model(plate_tensor)
                predicted_text = decode_text(output)
        
        # --- 4. Compare Prediction with True Label ---
        if predicted_text == true_label:
            correct_predictions += 1
            
    # --- 5. Calculate and Print Final Accuracy ---
    accuracy = (correct_predictions / total_images) * 100
    print("\n--- Evaluation Complete ---")
    print(f"Total Images: {total_images}")
    print(f"Correctly Predicted (Full Match): {correct_predictions}")
    print(f"Training Accuracy: {accuracy:.2f}%")
    print("-------------------------")
    
    if accuracy < 10:
        print("\nDIAGNOSIS: The accuracy is very low. This is almost certainly because the Haar Cascade DETECTOR is failing to find the license plates. See recommendations to improve.")
    elif 10 <= accuracy < 70:
        print("\nDIAGNOSIS: The model is learning, but performance is low. This is likely a mix of detection failures and the RECOGNITION model needing more training.")
    else:
        print("\nDIAGNOSIS: The model has learned the training data well! Poor test performance is likely due to the detector failing on the different test images.")


if __name__ == '__main__':
    run_evaluation()