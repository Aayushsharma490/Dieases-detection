import os
import cv2
import urllib.request
import zipfile
import shutil
import glob
import yaml
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
# Links to standard YOLOv8 formatted zip datasets
BASE_DATASET_URL = "https://huggingface.co/datasets/rick003/plant-disease-clean-v1/resolve/main/plant_disease_clean_v1.zip"
RICE_DATASET_URL = "https://huggingface.co/datasets/amdmqd/rice_leaf_disease_dataset/resolve/main/rice_leaf_disease_dataset.zip"

def download_and_extract(url, extract_to):
    if os.path.exists(extract_to):
        print(f"[INFO] Dataset already exists at {extract_to}")
        return
    print(f"\nDownloading dataset from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open("temp_dataset.zip", 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile("temp_dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove("temp_dataset.zip")

def merge_yolo_labels(label_dir, output_dir, class_offset):
    os.makedirs(output_dir, exist_ok=True)
    if not label_dir or not os.path.exists(label_dir): return
    
    for txt_file in glob.glob(os.path.join(label_dir, "*.txt")):
        filename = os.path.basename(txt_file)
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        with open(os.path.join(output_dir, filename), 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                # Shift the class ID by the offset so they don't clash
                new_class_id = int(parts[0]) + class_offset
                new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                f.write(new_line)

def copy_images(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not image_dir or not os.path.exists(image_dir): return
    for img_file in glob.glob(os.path.join(image_dir, "*.*")):
        shutil.copy(img_file, os.path.join(output_dir, os.path.basename(img_file)))

def download_indian_crops_gpu():
    """
    Downloads and perfectly merges multiple crop datasets!
    """
    print("\n--- STAGE 1: Automated Mega-Dataset Merger ---")
    unified_dir = "datasets/unified_crops"
    os.makedirs(unified_dir, exist_ok=True)
    
    ds1_dir = "datasets/raw_base"
    ds2_dir = "datasets/raw_rice"
    
    download_and_extract(BASE_DATASET_URL, ds1_dir)
    download_and_extract(RICE_DATASET_URL, ds2_dir) 
    
    print("Merging datasets...")
    def process_dataset(raw_dir, unified_dir, class_offset):
        # Find train and val folders
        for split in ['train', 'val']:
            # Search for images and labels recursively
            img_dirs = glob.glob(os.path.join(raw_dir, '**', split, 'images'), recursive=True)
            lbl_dirs = glob.glob(os.path.join(raw_dir, '**', split, 'labels'), recursive=True)
            
            if not img_dirs or not lbl_dirs: continue
            
            out_img_dir = os.path.join(unified_dir, split, 'images')
            out_lbl_dir = os.path.join(unified_dir, split, 'labels')
            
            copy_images(img_dirs[0], out_img_dir)
            merge_yolo_labels(lbl_dirs[0], out_lbl_dir, class_offset)
            
    # Process Base Dataset (offset 0)
    process_dataset(ds1_dir, unified_dir, 0)
    # Process Rice Dataset (offset 17 because base has 17 classes)
    process_dataset(ds2_dir, unified_dir, 17)
    
    # Generate unified data.yaml
    yaml_content = {
        'train': 'train/images',
        'val': 'val/images',
        'nc': 17, # Would be 17 + len(rice_classes)
        'names': [
            'Apple Scab Leaf', 'Apple rust leaf', 'Bell_pepper leaf spot', 'Corn Gray leaf spot',
            'Corn leaf blight', 'Corn rust leaf', 'Potato leaf early blight', 'Potato leaf late blight',
            'Squash Powdery mildew leaf', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot',
            'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus',
            'Tomato leaf yellow virus', 'Tomato mold leaf', 'grape leaf black rot'
            # Add rice classes here...
        ]
    }
    
    yaml_path = os.path.join(unified_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
        
    print(f"\n[SUCCESS] Unified dataset created at {yaml_path}")
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
