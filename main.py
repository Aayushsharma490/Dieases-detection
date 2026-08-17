import os
import cv2
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
# This script will automatically download a Rice Disease dataset
# (a major Indian crop) using Roboflow.
DATA_YAML_PATH = "Rice-Disease-1/data.yaml"

def download_indian_crops_gpu():
    """
    Downloads massive unified datasets (Rice, Wheat, Corn, etc.) using Roboflow.
    This is best for when you run the script on your powerful GPU server!
    """
    print("\n--- Downloading Indian Crop Diseases (Rice, Wheat, Corn, etc.) ---")
    print("To get massive, properly merged YOLOv8 datasets for Indian crops:")
    print("1. Go to https://universe.roboflow.com/")
    print("2. Search for 'Indian Crop Diseases' or 'Rice Leaf Disease'")
    print("3. Click 'Download Dataset' -> 'YOLOv8 format' -> 'Show Download Code'")
    print("4. Paste your Roboflow API key and workspace code below in main.py:")
    
    # --- UNCOMMENT AND ADD YOUR KEY HERE ON THE GPU SYSTEM ---
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="YOUR_API_KEY_HERE")
    # project = rf.workspace("workspace-name").project("project-name")
    # dataset = project.version(1).download("yolov8")
    # return dataset.location
    
    print("\n[NOTE] Once you add your key, this function will automatically download and format everything!")
    
    yaml_path = "Rice-Disease-1/data.yaml"
    if not os.path.exists(yaml_path):
        print("\n[STOP] You haven't added your Roboflow API key yet!")
        print("Please open main.py, go to line 23, uncomment the code, and paste your API Key.")
        print("You can get a free key by creating an account at https://universe.roboflow.com/")
        import sys
        sys.exit(1)
        
    return yaml_path

def setup_dataset():
    """
    Sets up the basic Plant Disease dataset we used on the laptop.
    Returns the path to the downloaded Plant Disease dataset configuration.
    """
    yaml_path = os.path.join("datasets", "crop_disease", "dataset", "data.yaml")
    
    if os.path.exists(yaml_path):
        print(f"\n--- STAGE 1: Using Local Plant Disease Dataset ---")
        return yaml_path
    else:
        print(f"\n[ERROR] Dataset not found at {yaml_path}.")
        print("Falling back to sample dataset (coco8).")
        return "coco8.yaml"

def train_and_export_yolov8(data_config):
    """
    Trains the YOLOv8 Nano model and exports it for Raspberry Pi deployment.
    """
    print("\n--- STAGE 2: Training YOLOv8 Model ---")
    
    # Load the base YOLOv8 Nano model
    model = YOLO("yolov8n.pt")
    
    # Train the model for 20 epochs on the real plant dataset
    # We force the output directory to be 'runs/detect/crop_model'
    results = model.train(
        data=data_config, 
        epochs=20, 
        imgsz=640, 
        device="cpu",
        name="crop_model",
        exist_ok=True
    )
    
    try:
        print("\n--- STAGE 3: Exporting Model for Raspberry Pi (TFLite) ---")
        # Export the best weights to TFLite format
        tflite_path = model.export(format="tflite", imgsz=640)
        print(f"Your Raspberry Pi model is saved at: {tflite_path}")
    except Exception as e:
        print(f"\n[WARNING] TFLite export failed (likely because you are on Windows).")
        print("Error:", e)
        print("You can just copy 'runs/detect/crop_model/weights/best.pt' directly to your Raspberry Pi")
        print("and run it with YOLO('best.pt') there, or export it to TFLite directly on the Pi (Linux)!")
    
    print(f"\nTraining Complete!")
    return "runs/detect/crop_model/weights/best.pt"

def run_camera_inference(model_path):
    """
    Runs real-time camera inference using the trained YOLOv8 model.
    """
    print("\n--- STAGE 4: Real-Time Camera Inference (Demo) ---")
    
    if not os.path.exists(model_path):
        print(f"Warning: Trained model not found at {model_path}. Using base yolov8n.pt")
        model_path = "yolov8n.pt"

    # Load the trained model
    model = YOLO(model_path)
    
    # Run prediction directly on the camera feed
    # source=0 accesses the default webcam
    # show=True automatically opens a window and draws bounding boxes
    model.predict(source=0, show=True, conf=0.5)

# ==========================================
# MASTER EXECUTION CONTROLLER
# ==========================================
if __name__ == "__main__":
    print("Select Mode:")
    print("1. Train Model & Export for Raspberry Pi (Basic Laptop Dataset)")
    print("2. Run Camera Inference (Laptop Demo)")
    print("3. Train Massive Unified Model (Rice, Wheat, Corn) on GPU Server")
    choice = input("Enter choice (1, 2, or 3): ")
    
    if choice == '1':
        data_yaml = setup_dataset()
        best_model_path = train_and_export_yolov8(data_yaml)
    elif choice == '2':
        # Default save path for YOLOv8 weights (bundled in repo)
        run_camera_inference("disease_model.pt")
    elif choice == '3':
        data_yaml = download_indian_crops_gpu()
        # Train for much longer on the GPU
        model = YOLO("yolov8n.pt")
        print("\n--- Training on GPU server for 100 Epochs! ---")
        model.train(data=data_yaml, epochs=100, imgsz=640, name="indian_crop_model")
        print("Done! You can download 'runs/detect/indian_crop_model/weights/best.pt' and use it on the Pi.")
    else:
        print("Invalid choice.")
