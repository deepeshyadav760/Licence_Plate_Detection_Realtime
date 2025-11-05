import os
import cv2
import torch
from train import train_model
from detect import detect_license_plate

def main():
    """
    Main function to run the license plate recognition project.
    Handles model training check and launches real-time detection
    using a phone's IP camera stream.
    """
    # --- 1. MODEL TRAINING CHECK ---
    # Check if a trained model exists. If not, inform the user and exit.
    if not os.path.exists("models/crnn.pth"):
        print("---! TRAINED MODEL NOT FOUND !---")
        print("Please run 'python train.py' first to train the model.")
        # Optionally, you can automatically trigger training by uncommenting the next line:
        # print("Starting model training automatically...")
        # train_model()
        return

    print("Pre-trained model found. Initializing camera...")

    # --- 2. REAL-TIME PHONE CAMERA INTEGRATION ---
    # Replace this URL with the one provided by your IP Webcam app.
    # It must be on the same Wi-Fi network.
    url = "http://100.90.45.201:8080/video" 

    print(f"Attempting to connect to camera stream at: {url}")
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("\n---!!! CAMERA CONNECTION ERROR !!!---")
        print("1. Ensure your phone and computer are on the SAME Wi-Fi network.")
        print("2. Check that the IP Webcam app is running on your phone.")
        print("3. Verify the URL in 'main.py' matches the one on your phone's screen.")
        return

    print("\nCamera connected successfully! Starting real-time detection.")
    print("Point your phone at a license plate.")
    print("Press 'q' in the display window to exit the program.")

    # --- 3. REAL-TIME DETECTION LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the stream. Exiting.")
            break

        # Process the frame to detect and recognize license plates
        processed_frame = detect_license_plate(frame)

        # Display the output
        cv2.imshow('Real-time License Plate Recognition (Press Q to quit)', processed_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 4. CLEANUP ---
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()