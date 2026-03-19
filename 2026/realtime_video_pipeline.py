import os
import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist
from collections import OrderedDict
import matplotlib.pyplot as plt

# --- 1. Model Configuration & Loading ---
# Define custom losses for loading the model
@tf.keras.utils.register_keras_serializable()
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=num_classes)
        y_true = tf.cast(y_true, tf.float32)
        numerator = 2. * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denominator = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis=[1, 2])
        dice = (numerator + self.smooth) / (denominator + self.smooth)
        return 1 - tf.reduce_mean(dice)

@tf.keras.utils.register_keras_serializable()
class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, gamma=2.0, **kwargs):
        super(ComboLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.dice_loss = DiceLoss()
    def call(self, y_true, y_pred):
        dice = self.dice_loss(y_true, y_pred)
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.squeeze(tf.one_hot(y_true_int, tf.shape(y_pred)[-1]), axis=-2)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = y_true_one_hot * tf.pow(1 - y_pred, self.gamma)
        focal = tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))
        return self.alpha * focal + (1 - self.alpha) * dice

@tf.keras.utils.register_keras_serializable()
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None, **kwargs):
        super(SparseMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super(SparseMeanIoU, self).update_state(y_true, y_pred, sample_weight)
    def get_config(self):
        config = super(SparseMeanIoU, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config

def predict_tta(model, img):
    # 1. Original
    p1 = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    
    # 2. Horizontal Flip
    img_lr = np.fliplr(img)
    p2 = model.predict(np.expand_dims(img_lr, axis=0), verbose=0)[0]
    p2 = np.fliplr(p2) # Flip back
    
    # 3. Vertical Flip
    img_ud = np.flipud(img)
    p3 = model.predict(np.expand_dims(img_ud, axis=0), verbose=0)[0]
    p3 = np.flipud(p3) # Flip back
    
    # Average
    return (p1 + p2 + p3) / 3.0

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'DiceLoss': DiceLoss,
                'ComboLoss': ComboLoss,
                'SparseMeanIoU': SparseMeanIoU
            },
            compile=False # Only inference
        )
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- 2. Centroid Tracker ---
class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict() # ID -> Centroid
        self.disappeared = OrderedDict() # ID -> consecutive frames missed
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance # Max pixel distance to link centroids

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # Rects: list of (startX, startY, endX, endY)
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                # Distance threshold
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

# --- 3. Processing Pipeline ---
def process_video_file(video_path, model, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Tracker
    # Distance threshold depends on video resolution. 
    # For 1920x1080 at 5 FPS, fish might move significantly more pixels.
    # maxDistance to 250 prevents ID switching on large jumps.
    tracker = CentroidTracker(maxDisappeared=10, maxDistance=250) 
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    
    # For unique total count
    total_unique_ids = set()

    print(f"Processing {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 1. Preprocess
        # Resize to model input size (256, 256)
        input_img = cv2.resize(frame, (256, 256))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype("float32")
        # Input tensor creation inside predict_tta

        # 2. Predict Semantic Mask with TTA
        pred = predict_tta(model, input_img) # (256, 256, 2)
        mask_prob = pred[:, :, 1] # Probability of Fish class
        
        # 3. Instance Separation (Watershed)
        # Threshold
        binary_mask = (mask_prob > 0.6).astype(np.uint8) * 255
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area (Distance Transform)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed
        # Apply on resized 256x256 image (needs 3 channel 8-bit)
        vis_img = cv2.resize(frame, (256, 256))
        markers = cv2.watershed(vis_img, markers)
        
        # 4. Extract Bounding Boxes for Tracking
        rects = []
        
        # Loop over unique markers (skip 0=unknown, 1=background)
        unique_markers = np.unique(markers)
        for m in unique_markers:
            if m <= 1: continue # 0 is boundary, 1 is background
            
            # Create mask for this instance
            instance_mask = np.zeros_like(binary_mask)
            instance_mask[markers == m] = 255
            
            # Find contours
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # Scale boxes back to original image size
                scale_x = width / 256.0
                scale_y = height / 256.0
                
                rect_orig = (
                    int(x * scale_x), int(y * scale_y),
                    int((x + w) * scale_x), int((y + h) * scale_y)
                )
                rects.append(rect_orig)

        # 5. Update Tracker
        objects = tracker.update(rects)

        # 6. Visualization
        for (objectID, centroid) in objects.items():
            total_unique_ids.add(objectID)
            
            # Draw ID
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Draw box for debugging
        for (startX, startY, endX, endY) in rects:
             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        # Overlay Count
        info_text = f"Frame: {frame_count} | Active: {len(objects)} | Total Unique: {len(total_unique_ids)}"
        cv2.putText(frame, info_text, (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Finished. Total Unique Fish: {len(total_unique_ids)}")

def main():
    # 1. Load Model
    # Look for best model first, then latest
    model_path = "fish_segmentation_best.h5"
    if not os.path.exists(model_path):
        model_path = "fish_segmentation_latest.h5"
    
    if not os.path.exists(model_path):
        print("No model found. Please run Segmentation.py first.")
        return

    model = load_model(model_path)
    if not model:
        return

    # 2. Process Single Video
    video_file = "valid_frames_combined.avi"
    if not os.path.exists(video_file):
        print(f"Video {video_file} not found. Please run create_videos.py first.")
        return

    # Output dir for tracked videos
    out_dir = "tracked_videos"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, "tracked_combined.avi")
    process_video_file(video_file, model, out_path)

if __name__ == "__main__":
    main()