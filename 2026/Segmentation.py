import os
import random
import datetime
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision, optimizers
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps
import matplotlib
import matplotlib.pyplot as plt
import keras as standalone_keras

# Set backend to non-interactive
matplotlib.use('Agg')

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Optimize GPU utilization with Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Mixed precision policy:', policy.compute_dtype, policy.variable_dtype)

# Global constants needed for loading
base_dir = os.path.dirname(os.path.abspath(__file__))

# Dice Loss for better segmentation performance
@tf.keras.utils.register_keras_serializable()
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        # Determine number of classes from predictions
        num_classes = tf.shape(y_pred)[-1]
        
        # Convert single-channel target to one-hot
        y_true = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=num_classes)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate Dice per class (batch-wise)
        # Sum over spatial dimensions (1, 2)
        numerator = 2. * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denominator = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis=[1, 2])
        
        dice = (numerator + self.smooth) / (denominator + self.smooth)
        
        # Return 1 - mean dice over classes and batch
        return 1 - tf.reduce_mean(dice)

# Combined Loss (Dice + Focal)
@tf.keras.utils.register_keras_serializable()
class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, gamma=2.0, **kwargs):
        super(ComboLoss, self).__init__(**kwargs)
        self.alpha = alpha # Weight for Focal Loss (1-alpha for Dice)
        self.gamma = gamma
        self.dice_loss = DiceLoss()
        
    def call(self, y_true, y_pred):
        # 1. Dice Loss
        dice = self.dice_loss(y_true, y_pred)
        
        # 2. Focal Loss
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.squeeze(tf.one_hot(y_true_int, tf.shape(y_pred)[-1]), axis=-2)
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = y_true_one_hot * tf.pow(1 - y_pred, self.gamma)
        focal = tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))
        
        return self.alpha * focal + (1 - self.alpha) * dice

def load_image_mask_numpy(img_path, mask_path):
    """
    Loads image and mask using OpenCV/Numpy logic.
    Returns (img, mask) with img in [0, 255].
    """
    img_path = img_path.decode('utf-8')
    mask_path = mask_path.decode('utf-8')
    
    # Load Image
    img = load_img(img_path, target_size=(img_size[0], img_size[1]))
    # Keep as [0, 255] float32
    img = np.array(img, dtype="float32")
    
    # Load Mask
    mask = load_img(mask_path, target_size=(img_size[0], img_size[1]), color_mode="grayscale")
    mask = np.array(mask)
    
    # Label Processing
    mask[mask == 255] = 1
    if np.min(mask) >= 1:
        mask -= 1
    mask[mask >= 3] = 0
    
    mask = np.expand_dims(mask, axis=-1).astype("uint8")
    
    return img, mask

def tf_load_data(img_path, mask_path):
    """TensorFlow wrapper for the numpy loader."""
    img, mask = tf.numpy_function(load_image_mask_numpy, [img_path, mask_path], [tf.float32, tf.uint8])
    img.set_shape([img_size[0], img_size[1], 3])
    mask.set_shape([img_size[0], img_size[1], 1])
    return img, mask

def augment_data(img, mask):
    """
    Apply TF augmentations. 
    Assumes IMG is [0, 255].
    """
    # Flips
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
        
    # Brightness (Scale delta for 0-255 range. 50 is approx 20%)
    img = tf.image.random_brightness(img, max_delta=50.0)
    
    # Contrast
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    
    # Clip to valid range
    img = tf.clip_by_value(img, 0.0, 255.0)
    
    return img, mask

def create_dataset(img_paths, mask_paths, batch_size, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    
    # Parallel load and process
    ds = ds.map(tf_load_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache the dataset in memory (RAM)
    # If it's too big, use .cache(filename='cache_train.tfrecord')
    ds = ds.cache()
    
    # Augment (only for training)
    if augment:
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
    # Batch and Prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

class SparseMeanIoU(keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None, **kwargs):
        super(SparseMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes # Store for get_config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super(SparseMeanIoU, self).update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super(SparseMeanIoU, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    # --- Encoder (MobileNetV2) ---
    # Use MobileNetV2 pretrained on ImageNet as the feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs, 
        include_top=False, 
        weights="imagenet"
    )

    # Use specific activation layers from MobileNetV2 for skip connections
    # These layers correspond to different resolutions (reductions of 2x, 4x, 8x, 16x, 32x)
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4 (Bottleneck)
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers_output)
    down_stack.trainable = True # Unfreeze encoder for fine-tuning

    skips = down_stack(inputs)
    x = skips[-1]     # The bottleneck (smallest resolution, deep features)
    skips = reversed(skips[:-1])

    # --- Decoder (Upsampling) ---
    # Upsample and concatenate with skip connections (U-Net style)
    
    def upsample(filters, size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.ReLU()
        ])

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])


    last = tf.keras.layers.Conv2DTranspose(num_classes, 3, strides=2, padding='same') # 128 -> 256
    
    x = last(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Function to load the latest saved model for resuming training
def load_latest_model():
    model_path = os.path.join(base_dir, "fish_segmentation_latest.h5")
    epoch_file = os.path.join(base_dir, "latest_epoch.txt")
    initial_epoch = 0
    model = None

    if os.path.exists(model_path) and os.path.exists(epoch_file):
        try:
            with open(epoch_file, "r") as f:
                initial_epoch = int(f.read().strip())
            
            # Custom objects for loading the model
            custom_objects = {
                "SparseMeanIoU": SparseMeanIoU,
                "DiceLoss": DiceLoss,
                "ComboLoss": ComboLoss
            }

            # Try loading with compile=False to avoid quantization issues
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                print(f"Resuming training from epoch {initial_epoch}. Loaded model from {model_path}")
            except Exception as load_err:
                print(f"Direct model load failed: {load_err}")
                # Fallback: Load weights into fresh architecture
                print("Attempting fallback: loading weights into fresh model...")
                model = get_model(img_size, num_classes)
                model.load_weights(model_path)
                print(f"Resuming training from epoch {initial_epoch}. Loaded weights from {model_path}")

        except Exception as e:
            print(f"Error loading saved model or epoch file: {e}")
            print("Starting new training run.")
            model = None
            initial_epoch = 0

    return model, initial_epoch

class ResumableModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath_best, filepath_latest, epoch_file, verbose=0):
        super().__init__()
        self.filepath_best = filepath_best
        self.filepath_latest = filepath_latest
        self.epoch_file = epoch_file
        self.verbose = verbose
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_loss = logs.get('val_loss')

        # Always save latest model for resumability
        self.model.save(self.filepath_latest)
        self._save_epoch_number(epoch + 1)
        if self.verbose > 0:
            print(f"Epoch {epoch + 1}: Saving latest model for resume to {self.filepath_latest}.")

        # Save best model if val_loss improved
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: val_loss improved, saving best model to {self.filepath_best}")
            self.model.save(self.filepath_best)
    
    def _save_epoch_number(self, epoch):
        with open(self.epoch_file, "w") as f:
            f.write(str(epoch))

class PlotMetricsCallback(keras.callbacks.Callback):
    """Custom callback to plot and save training metrics after training completes."""
    def __init__(self, log_dir, history_file=None):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
        
        # History file for persistent storage
        if history_file is None:
            history_file = os.path.join(base_dir, "training_history.json")
        self.history_file = history_file
        
        # Try to load existing history
        self.full_history = self._load_history()
        
        # Metrics for current run only
        self.current_metrics = {
            'loss': [], 'val_loss': [],
            'sparse_mean_io_u': [], 'val_sparse_mean_io_u': []
        }

    def _load_history(self):
        """Load training history from JSON file if it exists."""
        if os.path.exists(self.history_file):
            try:
                import json
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load training history: {e}")
                return {'loss': [], 'val_loss': [], 'sparse_mean_io_u': [], 'val_sparse_mean_io_u': []}
        return {'loss': [], 'val_loss': [], 'sparse_mean_io_u': [], 'val_sparse_mean_io_u': []}

    def _save_history(self):
        """Save training history to JSON file."""
        try:
            import json
            with open(self.history_file, 'w') as f:
                json.dump(self.full_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Append current logs to current run metrics
        for k in self.current_metrics.keys():
            if k in logs:
                self.current_metrics[k].append(logs[k])
    
    def on_train_end(self, logs=None):
        # Append current run metrics to full history
        for k in self.current_metrics.keys():
            self.full_history[k].extend(self.current_metrics[k])
        
        # Save accumulated history
        self._save_history()
        
        # Plot complete training history
        epochs_range = range(1, len(self.full_history['loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.full_history['loss'], label='Train Loss')
        if self.full_history['val_loss']:
            plt.plot(epochs_range, self.full_history['val_loss'], label='Val Loss')
        plt.title('Loss (All Runs)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot IoU
        plt.subplot(1, 2, 2)
        if self.full_history['sparse_mean_io_u']:
             plt.plot(epochs_range, self.full_history['sparse_mean_io_u'], label='Train IoU')
        if self.full_history['val_sparse_mean_io_u']:
             plt.plot(epochs_range, self.full_history['val_sparse_mean_io_u'], label='Val IoU')
        plt.title('Sparse Mean IoU (All Runs)')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.log_dir, "training_metrics.png")
        print(f"Saving full training metrics plot to: {os.path.abspath(save_path)}")
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close()

class GPUMonitorCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            gpu_info = result.strip().split(',')
            if len(gpu_info) >= 3:
                util = gpu_info[0].strip()
                mem_used = gpu_info[1].strip()
                mem_total = gpu_info[2].strip()
                print(f" [GPU] Util: {util}%, Mem: {mem_used}/{mem_total} MB")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

# Configuration
img_size = (256, 256)
num_classes = 2
batch_size = 20 # Increased for GPU utilization

if __name__ == "__main__":
    # Free up RAM
    keras.backend.clear_session()
    
    # Configuration
    input_dir = os.path.join(base_dir, "images", "valid")
    target_dir = os.path.join(base_dir, "masks", "valid")

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    # Split our img paths into a training and a validation set
    val_samples = 100
    random.seed(100) # Use seed for reproducibility
    random.shuffle(input_img_paths)
    random.seed(100) # Ensure targets are shuffled same way!
    random.shuffle(target_img_paths)
    
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Create Datasets using tf.data
    train_ds = create_dataset(train_input_img_paths, train_target_img_paths, batch_size=batch_size, augment=True)
    val_ds = create_dataset(val_input_img_paths, val_target_img_paths, batch_size=batch_size, augment=False)

    # Load existing model or create a new one
    model, initial_epoch = load_latest_model()
    if model is None:
        print("Creating a new model.")
        model = get_model(img_size, num_classes)
        # We will compile later with the scheduler
    else:
        print(f"Resumed training from epoch {initial_epoch}.")
    
    # TensorBoard Logging
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Attempting to create log directory: {os.path.abspath(log_dir)}")
    
    # Training Configuration
    goal_epochs = 250
    
    # Learning Rate Schedule (Cosine Decay)
    initial_learning_rate = 1e-4
    steps_per_epoch = len(train_input_img_paths) // batch_size
    # Calculate remaining steps
    remaining_epochs = max(1, goal_epochs - initial_epoch)
    total_steps = steps_per_epoch * remaining_epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_steps,
        alpha=0.01 
    )

    callbacks = [
        ResumableModelCheckpoint(
            filepath_best=os.path.join(base_dir, "fish_segmentation_best.h5"),
            filepath_latest=os.path.join(base_dir, "fish_segmentation_latest.h5"),
            epoch_file=os.path.join(base_dir, "latest_epoch.txt"),
            verbose=1
        ),
        keras.callbacks.EarlyStopping(monitor="val_sparse_mean_io_u", mode="max", patience=20, restore_best_weights=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(10, 20)),
        PlotMetricsCallback(log_dir),
        GPUMonitorCallback()
    ]

    # Re-compile model with new optimizer and scheduler
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule), 
        loss=ComboLoss(alpha=0.5, gamma=2.0), 
        metrics=[SparseMeanIoU(num_classes=num_classes)]
    )

    # Train the model
    model.fit(
        train_ds, 
        epochs=goal_epochs, 
        validation_data=val_ds, 
        initial_epoch=initial_epoch, 
        callbacks=callbacks
        )
