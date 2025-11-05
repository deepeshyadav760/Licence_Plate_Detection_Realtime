import os
import cv2
import torch
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# --- Re-import CRNN class and decode function ---
from train import CRNN, CHARS, IMG_WIDTH, IMG_HEIGHT
from detect import decode_text

def run_prediction():
    """
    Loads trained models, processes all test images using YOLO and CRNN, 
    and generates submission.csv.
    """
    print("Starting prediction on the test dataset with YOLO detector...")
    
    # --- 1. Load Models ---
    crnn_model = CRNN(len(CHARS) + 1)
    crnn_model.load_state_dict(torch.load("models\crnn_augmented.pth", map_location=torch.device('cpu')))
    crnn_model.eval()

    YOLO_MODEL_PATH = "runs/detect/yolov8_license_plate_detector2/weights/best.pt"
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("Models loaded successfully.")

    # --- 2. Prepare Submission DataFrame ---
    submission_df = pd.read_csv("SampleSubmission.csv")
    for col in "0123456789":
        submission_df[col] = 0
    submission_df.set_index('id', inplace=True)
    
    # --- 3. Process Each Test Image ---
    test_image_dir = "data/test"
    test_images = sorted(os.listdir(test_image_dir))
    
    for img_name in tqdm(test_images, desc="Processing Test Images"):
        img_path = os.path.join(test_image_dir, img_name)
        frame = cv2.imread(img_path)
        
        # --- Use YOLO for detection ---
        results = yolo_model(frame)[0]
        
        if len(results.boxes) > 0:
            # Assume the most confident detection is the one we want
            box = results.boxes[0]
            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
            plate_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                transforms.ToTensor(),
            ])
            plate_tensor = transform(plate_roi).unsqueeze(0)
            
            with torch.no_grad():
                output = crnn_model(plate_tensor)
                plate_text = decode_text(output)
            
            # --- 4. Populate Submission ---
            img_num = img_name.split('.')[0]
            for i, char in enumerate(plate_text):
                char_position = i + 1
                row_id = f"img_{img_num}_{char_position}"
                if char.isdigit() and row_id in submission_df.index:
                    submission_df.loc[row_id, char] = 1

    # --- 5. Save Final Submission File ---
    submission_df.reset_index(inplace=True)
    submission_df.to_csv("submission_yolo.csv", index=False)
    print("\nPrediction complete!")
    print("Results saved to 'submission_yolo.csv'.")

if __name__ == '__main__':
    run_prediction()