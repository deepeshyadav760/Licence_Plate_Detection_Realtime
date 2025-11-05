import cv2
import torch
from torchvision.transforms import transforms
from PIL import Image
from ultralytics import YOLO

# --- Import CRNN model and functions ---
from train import CRNN, CHARS, IMG_WIDTH, IMG_HEIGHT

# --- Model Loading ---
# Load the pre-trained CRNN model for character recognition
try:
    crnn_model = CRNN(len(CHARS) + 1)
    crnn_model.load_state_dict(torch.load("models/crnn.pth", map_location=torch.device('cpu')))
    crnn_model.eval()
except FileNotFoundError:
    print("Error: CRNN model 'crnn.pth' not found. Please run train.py first.")
    exit()

# --- NEW: Load the trained YOLOv8 model ---
# NOTE: Update this path to where your best model is saved after training!
YOLO_MODEL_PATH = "runs/detect/yolov8_license_plate_detector2/weights/best.pt"
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print(f"Please ensure the model path is correct: {YOLO_MODEL_PATH}")
    exit()

def decode_text(preds):
    """Decodes the raw output of the CRNN model into human-readable text."""
    preds_idx = preds.argmax(2)
    preds_idx = preds_idx.transpose(1, 0).contiguous().view(-1)
    
    char_list = []
    for i in range(len(preds_idx)):
        if preds_idx[i] != len(CHARS) and (i == 0 or preds_idx[i] != preds_idx[i-1]):
            char_list.append(CHARS[preds_idx[i]])
    return "".join(char_list)


def detect_license_plate(frame):
    """
    Detects and recognizes a license plate using YOLOv8 for detection and CRNN for recognition.
    """
    # --- NEW: Use YOLO for Detection ---
    results = yolo_model(frame)[0]
    
    # Process each detected plate
    for box in results.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
        
        # Extract the license plate ROI
        # Crop from the original frame, but convert to grayscale for CRNN
        plate_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        
        # Preprocess the plate image for the CRNN model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
        ])
        plate_tensor = transform(plate_roi).unsqueeze(0)

        # Perform text recognition with the CRNN model
        with torch.no_grad():
            output = crnn_model(plate_tensor)
            plate_text = decode_text(output)

        # --- Display Results ---
        # Draw bounding box and text on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, plate_text, (x1, y1 - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return frame