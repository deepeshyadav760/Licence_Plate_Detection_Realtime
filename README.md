# SOTA License Plate Detection and Recognition

This project uses a Haar Cascade classifier for real-time license plate detection and a custom-trained Convolutional Recurrent Neural Network (CRNN) for accurate text recognition from a live phone camera feed.

## Project Structure

```
license_plate_recognition/
|
|-- data/
|   |-- train/                # All training images
|   |-- test/                 # All test images
|
|-- models/                   # Stores the trained model (crnn.pth)
|
|-- licplatesdetection_train.csv
|-- licplatesrecognition_train.csv
|
|-- main.py                 # Main script for real-time detection
|-- train.py                # Script to train the CRNN model
|-- detect.py               # Detection and recognition logic
|-- requirements.txt        # Dependencies
|-- README.md               # This file
```

## Setup and Installation

### Step 1: Clone the Repository & Set Up Folders
1.  Create the directory structure as shown above.
2.  Place your training images inside `data/train/`.
3.  Place `licplatesdetection_train.csv` and `licplatesrecognition_train.csv` in the root folder.

### Step 2: Install Dependencies
It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### Step 3: Train the Recognition Model
This step trains the CRNN model to recognize characters on license plates. Training can take a significant amount of time depending on your hardware (a GPU is recommended).

```bash
python train.py```
Upon completion, a file named `crnn.pth` will be saved in the `models/` directory. **You only need to do this once.**

### Step 4: Set Up Your Phone as an IP Camera
1.  Connect your phone and your computer to the **same Wi-Fi network**.
2.  Install an IP camera app on your phone (e.g., **"IP Webcam"** for Android).
3.  Start the app's server. It will display a URL like `http://192.168.1.5:8080`.
4.  Open the `main.py` file and update the `url` variable with the one from your phone, adding `/video` at the end:
    ```python
    url = "http://192.168.1.5:8080/video"
    ```

### Step 5: Run the Real-Time Detection
Execute the main script to start the live detection.

```bash
python main.py
```

A window will appear on your screen showing the feed from your phone's camera. Point it at a vehicle's license plate, and the system will draw a box around it and display the recognized text. Press 'q' to quit.