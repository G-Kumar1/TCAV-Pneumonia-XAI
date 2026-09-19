import os
import logging

# 1. Suppress TensorFlow C++ level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import glob
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_1samp

# ---------------------------------------------------------
# 1. Image Loading & Preprocessing Utilities
# ---------------------------------------------------------
def load_and_preprocess_folder(folder_path, target_size=(224, 224)):
    """Loads all images from a folder and prepares them for Keras ResNet50."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if len(image_paths) == 0:
        raise FileNotFoundError(f"❌ Error: Found 0 images inside '{folder_path}'")
        
    images = []
    paths_loaded = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB').resize(target_size)
            img_array = np.array(img, dtype=np.float32)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            images.append(img_array)
            paths_loaded.append(os.path.basename(path))
        except Exception as e:
            print(f"⚠️ Skipping corrupted image {path}: {e}")
        
    return np.array(images), paths_loaded

# ---------------------------------------------------------
# 2. Path Configurations & Layer Selection
# ---------------------------------------------------------
BASE_CONCEPTS_DIR = "tcav_concept_dataset"  
VAL_IMAGES_DIR = os.path.join("dataset", "chest_xray_processed", "test")
MODEL_PATH = os.path.join("models", "resnet50_fine_tuned.keras")
OUTPUT_CSV_PATH = "image_tcav_analysis.csv"

def main():
    print("🧠 Loading Fine-Tuned Keras Model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model weight file not found at {MODEL_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Target an upstream residual block to preserve spatial feature variance
    target_layer_candidates = ['conv4_block6_out', 'conv5_block1_out', 'conv5_block3_out']
    bottleneck_layer = None
    bottleneck_layer_name = ""
    
    for candidate in target_layer_candidates:
        try:
            bottleneck_layer = model.get_layer(candidate)
            bottleneck_layer_name = candidate
            break
        except ValueError:
            continue
            
    if bottleneck_layer is None:
        raise ValueError("❌ Could not find a suitable bottleneck layer in the loaded model.")
        
    print(f"⚙️ Hooking upstream feature layer: '{bottleneck_layer_name}' (Shape: {bottleneck_layer.output.shape})")

    # Isolate final dense classification parameters
    final_dense_layer = model.layers[-1]
    weights, biases = final_dense_layer.get_weights()

    # Build multi-output intermediate model:
    # 1. Bottleneck feature maps (for CAV extraction)
    # 2. Input tensor entering final dense layer (for linear logit reconstruction)
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=[bottleneck_layer.output, final_dense_layer.input]
    )

    print("\n⏳ Loading CheXpert clinical concept datasets...")
    concepts = {
        "lung_opacity": load_and_preprocess_folder(os.path.join(BASE_CONCEPTS_DIR, "lung_opacity"))[0],
        "consolidation": load_and_preprocess_folder(os.path.join(BASE_CONCEPTS_DIR, "consolidation"))[0],
        "pleural_effusion": load_and_preprocess_folder(os.path.join(BASE_CONCEPTS_DIR, "pleural_effusion"))[0],
    }
    
    # Compute concept bottleneck activations and apply Global Average Pooling (GAP)
    concept_activations = {}
    for name, img_tensor in concepts.items():
        acts, _ = activation_model.predict(img_tensor, verbose=0)   
        concept_activations[name] = np.mean(acts, axis=(1, 2))
        print(f" ➔ {name:<18} pooled activation shape: {concept_activations[name].shape}")

    # Load validation test set for pneumonia sensitivity mapping
    pneumonia_val_path = os.path.join(VAL_IMAGES_DIR, "PNEUMONIA")
    test_images, test_filenames = load_and_preprocess_folder(pneumonia_val_path)
    
    NUM_RANDOM_SETS = 5
    tcav_runs_tracker = {name: [] for name in concepts.keys()}
    per_image_results = {fname: {} for fname in test_filenames}

    print(f"\n🔄 Running Statistical Validation across {NUM_RANDOM_SETS} Discrete Normal Baselines...")

    # ---------------------------------------------------------
    # 3. Statistical Analysis Loop Across Control Sets
    # ---------------------------------------------------------
    for run_idx in range(NUM_RANDOM_SETS):
        baseline_folder_name = f"random_baseline_{run_idx + 1}"
        print(f"--- Processing Run {run_idx + 1}/{NUM_RANDOM_SETS} using '{baseline_folder_name}' ---")
        
        baseline_path = os.path.join(BASE_CONCEPTS_DIR, baseline_folder_name)
        random_imgs, _ = load_and_preprocess_folder(baseline_path)
        
        acts_random, _ = activation_model.predict(random_imgs, verbose=0)
        X_random = np.mean(acts_random, axis=(1, 2))
        
        cavs = {}
        # Train linear classifiers to extract Concept Activation Vectors (CAVs)
        for concept_name, X_concept in concept_activations.items():
            X = np.vstack((X_concept, X_random))
            y = np.array([1] * len(X_concept) + [0] * len(X_random))
            
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.20, random_state=42 + run_idx, stratify=y
            )

            clf = LogisticRegression(max_iter=3000, C=0.5, random_state=42)
            clf.fit(X_train, y_train)
            
            # Normalize CAV vector
            cav = clf.coef_[0]
            norm = np.linalg.norm(cav)
            if norm != 0:
                cavs[concept_name] = cav / norm

        # Batch Sensitivity Evaluation via Automatic Differentiation
        EVAL_BATCH_SIZE = 32
        num_images = len(test_images)
        run_positive_counts = {name: 0 for name in cavs.keys()}

        for i in range(0, num_images, EVAL_BATCH_SIZE):
            batch_images = test_images[i:i + EVAL_BATCH_SIZE]
            batch_tensor = tf.convert_to_tensor(batch_images)
            batch_fnames = test_filenames[i:i + EVAL_BATCH_SIZE]
            
            with tf.GradientTape() as tape:
                acts, dense_input = activation_model(batch_tensor)
                tape.watch(acts)
                
                # Reconstruct un-saturated pre-activation logit z = W * x + b
                logits = tf.matmul(dense_input, weights) + biases
                loss = logits[:, 0]
                
            # Backpropagation of gradients to upstream bottleneck activations
            grads = tape.gradient(loss, acts).numpy()
            grads_pooled = np.mean(grads, axis=(1, 2))
            
            # Compute directional derivatives S_c,k(x) = grad . cav
            for concept_name, cav in cavs.items():
                directional_derivatives = np.dot(grads_pooled, cav)
                run_positive_counts[concept_name] += np.sum(directional_derivatives > 0)
                
                # Record continuous derivatives from final run iteration for export
                if run_idx == NUM_RANDOM_SETS - 1:
                    for idx, fname in enumerate(batch_fnames):
                        per_image_results[fname][concept_name] = directional_derivatives[idx]

        for concept_name in cavs.keys():
            score = run_positive_counts[concept_name] / num_images
            tcav_runs_tracker[concept_name].append(score)

    # ---------------------------------------------------------
    # 4. Statistical Summary Output
    # ---------------------------------------------------------
    print("\n====================================================")
    print("🎉 STATISTICAL TCAV SUMMARY RESULTS (5 MEDICAL RUNS)")
    print("====================================================")
    print(f"{'Clinical Concept':<20} | {'Mean TCAV':<10} | {'Std Dev':<8} | {'p-value':<8} | Status")
    print("-" * 65)

    for concept_name, scores in tcav_runs_tracker.items():
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        
        # 1-sample t-test against random chance baseline threshold (0.50)
        _, p_val = ttest_1samp(scores, 0.5)
        
        status = "Significant" if p_val < 0.05 and mean_score > 0.5 else "Insignificant"
        print(f"{concept_name:<20} | {mean_score:.4f}    | {std_dev:.4f}  | {p_val:.4f} | {status}")
    print("====================================================\n")

    # ---------------------------------------------------------
    # 5. Export Per-Image Continuous Matrix
    # ---------------------------------------------------------
    print(f"💾 Saving continuous per-image TCAV matrix to '{OUTPUT_CSV_PATH}'...")
    with open(OUTPUT_CSV_PATH, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        headers = ["filename"] + list(concepts.keys())
        writer.writerow(headers)
        
        for fname, concept_data in per_image_results.items():
            row = [fname] + [concept_data.get(c_name, 0.0) for c_name in concepts.keys()]
            writer.writerow(row)
            
    print("✅ Execution complete. Run 'python plot_tcav_results.py' to generate paper figures.")

if __name__ == "__main__":
    main()