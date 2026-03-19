import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Segmentation import get_model, num_classes, img_size, SparseMeanIoU, DiceLoss, ComboLoss

# --- Configuration ---
IMG_HEIGHT, IMG_WIDTH = img_size
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fish_segmentation_best.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "science_demo_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_full_model(model_path):
    print(f"Building model architecture...")
    model = get_model(img_size, num_classes)
    
    print(f"Loading weights from {model_path}...")
    try:
        model.load_weights(model_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Fallback: Try loading full model if load_weights fails (e.g. if it's H5)
        try:
            custom_objects = {
                "SparseMeanIoU": SparseMeanIoU,
                "DiceLoss": DiceLoss,
                "ComboLoss": ComboLoss
            }
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Full model loaded successfully.")
        except Exception as e2:
             raise e # Raise original error if both fail
             
    return model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Prepare the model for Grad-CAM
    try:
        bottleneck_output = model.get_layer('sequential').input
        print("Using 'sequential' layer input as bottleneck source.")
    except Exception as e:
        print(f"Could not find 'sequential' layer input: {e}")
        try:
            encoder = model.get_layer('model')
            bottleneck_output = encoder.output[-1] 
            print("Using encoder 'model' output as bottleneck source.")
        except ValueError:
            # Fallback
            try:
                bottleneck_output = model.get_layer(last_conv_layer_name).output
            except ValueError:
                 print(f"Could not find layer for GradCAM. Available layers: {[l.name for l in model.layers]}")
                 return np.zeros((img_array.shape[1], img_array.shape[2])) # Return empty

    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[bottleneck_output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, :, :, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        print("DEBUG: Gradient is None!")
        print(f"  Target (class_channel) shape: {class_channel.shape}")
        print(f"  Source (last_conv_layer_output) shape: {last_conv_layer_output.shape}")
        # print(f"  Grad model inputs: {grad_model.inputs}")
        # print(f"  Grad model outputs: {grad_model.outputs}")
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def visualize_layers(model, img_tensor, original_img):
    # 1. Identify layers to visualize
    # In Segmentation.py, these are the outputs of the encoder (nested model)
    # The encoder outputs: [block_1_expand_relu, block_3_expand_relu, block_6..., block_13..., block_16_project]
    layer_names_map = [
        'block_1_expand_relu',   # Index 0
        'block_3_expand_relu',   # Index 1
        'block_16_project',      # Index 4 (Last)
    ]
    indices = [0, 1, 4]
    
    feature_maps = []
    
    try:
        # Try getting the nested encoder
        encoder = model.get_layer('model')
        print("Found nested encoder. Extracting features...")
        # encoder.predict(img_tensor) returns a list of arrays
        all_outputs = encoder.predict(img_tensor, verbose=0)
        
        for idx in indices:
            if idx < len(all_outputs):
                feature_maps.append(all_outputs[idx])
            else:
                 print(f"Warning: Encoder output index {idx} out of range.")
                 
    except ValueError:
        print("Nested encoder 'model' not found. Attempting direct layer access...")
        # Fallback: Try accessing layers directly by name
        try:
             outputs = [model.get_layer(name).output for name in layer_names_map]
             feature_model = keras.models.Model(inputs=model.inputs, outputs=outputs)
             feature_maps = feature_model.predict(img_tensor, verbose=0)
        except Exception as e:
            print(f"Could not extract features: {e}")
            return

    # Plotting
    for layer_name, fmap in zip(layer_names_map, feature_maps):
        # fmap shape: (1, H, W, Filters)
        fmap = fmap[0] 
        
        # Calculate average activation (Heatmap of activity)
        avg_activation = np.mean(fmap, axis=-1)
        
        # Plot
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title(f"Layer: {layer_name}\n(Average Activation)")
        plt.imshow(avg_activation, cmap='viridis')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Original Image")
        plt.imshow(original_img)
        plt.axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, f"layer_{layer_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization for {layer_name} to {save_path}")

def main():
    print("-- Starting Science Demo --")
    
    # 1. Load Model
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Primary model not found at {model_path}")
        model_path = os.path.join(BASE_DIR, "fish_segmentation_latest.keras")
        if not os.path.exists(model_path):
             print(f"Fallback model not found at {model_path}")
             return
    
    print(f"Loading model from: {model_path}")
    try:
        model = load_full_model(model_path)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load an Image (Validation set)
    from glob import glob
    image_dir = os.path.join(BASE_DIR, "images", "valid")
    files = sorted(glob(os.path.join(image_dir, "*.jpg")))
    if not files:
        print("No validation images found.")
        return
        
    img_path = files[0] # Just take the first one
    print(f"Analyzing image: {os.path.basename(img_path)}")
    
    # Preprocess
    original_bgr = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize [0, 255] float32
    img_input = cv2.resize(original_rgb, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32)
    img_tensor = np.expand_dims(img_input, axis=0)
    
    # 3. Visualize Intermediate Layers
    visualize_layers(model, img_tensor, original_rgb)
    
    # 4. Prediction Confidence Heatmap
    print("\n-- Generating Prediction Visualizations --")
    
    # Get model prediction
    preds = model.predict(img_tensor, verbose=0)[0]  # (H, W, num_classes)
    fish_prob = preds[:, :, 1]  # Fish class probability
    mask = (fish_prob > 0.5).astype(np.uint8)
    
    # 4a. Confidence Heatmap
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    display_img = cv2.resize(original_rgb, (IMG_WIDTH, IMG_HEIGHT))
    plt.imshow(display_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Prediction Confidence\n(Fish Class Probability)")
    plt.imshow(fish_prob, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Probability')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Binary Segmentation\n(Threshold: 0.5)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, "prediction_confidence.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction confidence to {save_path}")
    
    # 4b. Contour Overlay
    plt.figure(figsize=(10, 10))
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw on resized original
    overlay_img = display_img.copy()
    cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)  # Green contours
    
    plt.imshow(overlay_img)
    plt.title(f"Fish Detection Contours\n({len(contours)} regions detected)")
    plt.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, "contour_overlay.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved contour overlay to {save_path}")
    
    # 4c. Ground Truth Comparison (if mask exists)
    mask_dir = os.path.join(BASE_DIR, "masks", "valid")
    mask_basename = os.path.basename(img_path).replace('.jpg', '.png')
    gt_mask_path = os.path.join(mask_dir, mask_basename)
    
    if os.path.exists(gt_mask_path):
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (IMG_WIDTH, IMG_HEIGHT))
        gt_mask = (gt_mask > 0).astype(np.uint8)  # Binary ground truth
        
        # Calculate IoU
        intersection = np.logical_and(mask, gt_mask).sum()
        union = np.logical_or(mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Ground Truth Mask")
        plt.imshow(gt_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Overlay Comparison\nIoU: {iou:.3f}")
        # Create RGB comparison: Green = TP, Red = FN, Blue = FP
        comparison = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        comparison[np.logical_and(mask, gt_mask)] = [0, 255, 0]      # True Positive (green)
        comparison[np.logical_and(~mask.astype(bool), gt_mask.astype(bool))] = [255, 0, 0]  # False Negative (red)
        comparison[np.logical_and(mask.astype(bool), ~gt_mask.astype(bool))] = [0, 0, 255]  # False Positive (blue)
        plt.imshow(comparison)
        plt.axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, "ground_truth_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ground truth comparison to {save_path} (IoU: {iou:.3f})")
    else:
        print(f"Ground truth mask not found at {gt_mask_path}, skipping comparison.")
    
    print("\n-- Science Demo Complete! --")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
