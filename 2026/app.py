import os
import json
import csv
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import sys
import io
import time
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Paths to the DeepFish2 project
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
SEGMENTATION_DIR = os.path.join(PROJECT_DIR, "Segmentation")
IMAGE_DIR = os.path.join(SEGMENTATION_DIR, "images", "valid")
UPLOAD_FOLDER = os.path.join(APP_DIR, 'uploads')
FRAME_FOLDER = os.path.join(APP_DIR, 'video_frames')

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(SEGMENTATION_DIR, "fish_segmentation_best.h5")
YOLO_MODEL_PATH = os.path.join(APP_DIR, "training", "runs", "family_production", "weights", "best.pt")

# APP CONFIG
TAGS_FILE = os.path.join(APP_DIR, "manual_tags.csv")
SPECIES_FILE = os.path.join(APP_DIR, "species.json")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# Global model instances
seg_model = None
yolo_model = None
import threading
model_lock = threading.Lock()

def load_models():
    global seg_model, yolo_model
    with model_lock:
        # Load Segmentation Model if needed
        if seg_model is None:
            try:
                start_time = time.time()
                print("--- Loading Segmentation Model weights... ---")
                if SEGMENTATION_DIR not in sys.path:
                    sys.path.append(SEGMENTATION_DIR)
                from Segmentation import get_model, img_size, num_classes
                seg_model = get_model(img_size, num_classes)
                seg_model.load_weights(MODEL_PATH)
                print(f"--- Segmentation Model loaded in {time.time() - start_time:.2f}s ---")
            except Exception as e:
                print(f"Error loading segmentation model: {e}")
                
        # Load YOLO Model
        if yolo_model is None:
            try:
                start_time = time.time()
                print(f"--- Loading YOLO weights from: {YOLO_MODEL_PATH} ---")
                yolo_model = YOLO(YOLO_MODEL_PATH)
                # Warm up
                print("--- Warming up YOLO... ---")
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                yolo_model(dummy, verbose=False)
                print(f"--- YOLO loaded & warmed up in {time.time() - start_time:.2f}s ---")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images')
def get_images():
    # Read tagged images from CSV
    tagged_images = set()
    if os.path.exists(TAGS_FILE):
        try:
            with open(TAGS_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tagged_images.add(row['filename'])
        except Exception as e:
            print(f"Error reading tags: {e}")

    # List all images and filter out tagged ones
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    untagged_images = [f for f in all_images if f not in tagged_images]
    
    return jsonify(sorted(untagged_images))

@app.route('/api/video_frames')
def get_video_frames():
    tagged_images = set()
    if os.path.exists(TAGS_FILE):
        try:
            with open(TAGS_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tagged_images.add(row['filename'])
        except Exception as e:
            print(f"Error reading tags: {e}")

    # List all frames and filter out tagged ones
    all_frames = [f for f in os.listdir(FRAME_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    untagged_frames = [f for f in all_frames if f not in tagged_images]
    
    return jsonify(sorted(untagged_frames))

@app.route('/api/species')
def get_species():
    with open(SPECIES_FILE, 'r') as f:
        return jsonify(json.load(f))

@app.route('/api/detect/<filename>')
@app.route('/api/detect/frame/<filename>')
def detect(filename):
    start_total = time.time()
    # Check if it's a temp frame or a main image
    if '/frame/' in request.path:
        img_path = os.path.join(FRAME_FOLDER, filename)
    else:
        img_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(img_path):
        return "Not Found", 404
        
    load_models()
    
    start_yolo = time.time()
    results = yolo_model(img_path, verbose=False)[0]
    yolo_duration = time.time() - start_yolo
    
    boxes = []
    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        cls = int(box.cls[0].item())
        label = results.names[cls]
        boxes.append({
            "bbox": coords,
            "confidence": conf,
            "class": cls,
            "label": label
        })
        
    total_duration = time.time() - start_total
    print(f"--- Detection for {filename}: YOLO={yolo_duration:.2f}s, Total={total_duration:.2f}s ---")
    return jsonify(boxes)

@app.route('/api/predict/<filename>')
@app.route('/api/predict/frame/<filename>')
def predict(filename):
    start_total = time.time()
    if '/frame/' in request.path:
        img_path = os.path.join(FRAME_FOLDER, filename)
    else:
        img_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(img_path):
        return "Not Found", 404
        
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Preprocess
    load_models()
    from Segmentation import img_size
    img_input = cv2.resize(img_rgb, img_size).astype(np.float32)
    
    # Inference
    start_inf = time.time()
    pred = seg_model.predict(np.expand_dims(img_input, axis=0), verbose=0)[0]
    inf_duration = time.time() - start_inf

    mask = (pred[:, :, 1] > 0.5).astype(np.uint8) * 255
    
    # Resize mask back
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Encode as PNG
    _, buffer = cv2.imencode('.png', mask)
    
    print(f"--- Prediction for {filename}: Inference={inf_duration:.2f}s, Total={time.time() - start_total:.2f}s ---")
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/png'
    )

@app.route('/api/image/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/api/frame/<filename>')
def serve_frame(filename):
    return send_from_directory(FRAME_FOLDER, filename)

@app.route('/api/tag', methods=['POST'])
def save_tag():
    data = request.json
    filename = data.get('filename')
    common_name = data.get('common_name')
    scientific_name = data.get('scientific_name')
    bbox = data.get('bbox') # Optional: box being tagged
    
    # Append to CSV
    file_exists = os.path.isfile(TAGS_FILE)
    with open(TAGS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['filename', 'common_name', 'scientific_name', 'bbox'])
        writer.writerow([filename, common_name, scientific_name, json.dumps(bbox)])
    
    return jsonify({"status": "success"})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video part"}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extract 1 frame per second (approx)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if count % int(fps) == 0:
            frame_name = f"{filename}_frame_{count}.jpg"
            frame_path = os.path.join(FRAME_FOLDER, frame_name)
            cv2.imwrite(frame_path, frame)
            frames.append(frame_name)
        count += 1
    cap.release()

    return jsonify({"message": "Video processed", "frames": frames})

@app.route('/api/add_species', methods=['POST'])
def add_species():
    data = request.json
    family = data.get('family')
    common = data.get('common')
    scientific = data.get('scientific')
    
    with open(SPECIES_FILE, 'r+') as f:
        hierarchy = json.load(f)
        
        # If family doesn't exist, create it
        if family not in hierarchy:
            common_fam = family.replace('_', ' ')
            hierarchy[family] = {
                "common": common_fam, 
                "species": {
                    "Unknown": f"Unknown {common_fam}"
                }
            }
        
        # Add to family's species list
        hierarchy[family]["species"][scientific] = common
        
        f.seek(0)
        json.dump(hierarchy, f, indent=4)
        f.truncate()
        
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Pre-load models to avoid lag on first request
    load_models()
    app.run(debug=True, port=5000)
