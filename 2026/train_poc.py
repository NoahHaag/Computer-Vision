from ultralytics import YOLO
import os

# Paths
BASE_DIR = r'C:\Users\Noah\OneDrive\Desktop\DeepFish2\TaggingApp'
DATA_YAML = os.path.join(BASE_DIR, "training", "data.yaml")

def train_family_model():
    # Load the best weights from the previous v2 run
    best_weights = os.path.join(BASE_DIR, "training", "runs", "family_production_v2", "weights", "best.pt")
    model = YOLO(best_weights) 

    print("--- Starting Family-Level Fine-Tuning (v3) ---")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Starting from weights: {best_weights}")
    
    # Train the model
    # epochs=20 for a quick fine-tune on the new data
    results = model.train(
        data=DATA_YAML,
        epochs=20,
        imgsz=640,
        plots=True,
        project=os.path.join(BASE_DIR, "training", "runs"),
        name="family_production_v3"
    )

    print("\nTraining Complete!")
    print(f"Results saved to: {results.save_dir}")
    print("Look for 'best.pt' in the weights folder to use for the next version of the tagger!")

if __name__ == "__main__":
    train_family_model()
