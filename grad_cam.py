import os
import cv2
import random
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

os.makedirs("results/gradcam", exist_ok=True)
model = load_model("models/resnet50_fine_tuned.keras")


def generate_gradcam(img_path, true_label):

    # -------------------------------
    # Load Image
    # -------------------------------
    
    print("=" * 70)
    print("Processing :", os.path.basename(img_path))
    IMG_SIZE = (224, 224)

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )

    probability = float(prediction[0][0])

    predicted_class = (
        "PNEUMONIA"
        if probability > 0.5
        else "NORMAL"
    )

    print("=" * 60)
    print("Image :", os.path.basename(img_path))
    print("True Label :", true_label)
    print("Predicted :", predicted_class)
    confidence = probability if predicted_class == "PNEUMONIA" else (1 - probability)

    print("Probability :", probability)
    print("Confidence :", round(confidence * 100, 2), "%")

    # -------------------------------
    # GradCAM Model
    # -------------------------------

    last_conv_layer = "conv5_block3_out"

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_tensor
        )

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_heat = tf.reduce_max(heatmap)

    if max_heat.numpy() < 1e-8:
        print("Very weak Grad-CAM activation.")
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap /= max_heat

    heatmap = heatmap.numpy()

    # -------------------------------
    # Original Image
    # -------------------------------

    original = cv2.imread(img_path)

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    superimposed = cv2.addWeighted(
        original,
        0.7,
        heatmap,
        0.3,
        0
    )

    # -------------------------------
    # Decide Save Folder
    # -------------------------------

    if predicted_class == "NORMAL" and true_label == "NORMAL":

        save_folder = "results/gradcam/True_Normal"

    elif predicted_class == "PNEUMONIA" and true_label == "PNEUMONIA":

        save_folder = "results/gradcam/True_Pneumonia"

    elif predicted_class == "PNEUMONIA" and true_label == "NORMAL":

        save_folder = "results/gradcam/False_Positive"

    else:

        save_folder = "results/gradcam/False_Negative"

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    # -------------------------------
    # Plot
    # -------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title(os.path.basename(img_path))
    plt.axis("off")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("GradCAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title(
        f"True : {true_label}\n"
        f"Pred : {predicted_class}\n"
        f"Conf : {confidence*100:.2f}%"
    )
    plt.axis("off")

    plt.tight_layout()

    filename = os.path.basename(
        img_path
    )

    save_path = os.path.join(
        save_folder,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches = 'tight'
    )

    plt.close()
    csv_file = "results/gradcam_results.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image",
            "True Label",
            "Prediction",
            "Probability",
            "Confidence"
        ])
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            os.path.basename(img_path),
            true_label,
            predicted_class,
            probability,
            confidence
        ])

    print("Saved :", save_path)

TEST_DIR = "dataset/chest_xray_processed/test"

image_list = []

for cls in ["NORMAL", "PNEUMONIA"]:

    folder = os.path.join(TEST_DIR, cls)

    for file in os.listdir(folder):

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            image_list.append(
                (
                    os.path.join(folder, file),
                    cls
                )
            )


print("Total Test Images :", len(image_list))

# Shuffle images so NORMAL and PNEUMONIA are mixed
random.shuffle(image_list)

count = 0

for img_path, true_label in image_list:

    generate_gradcam(
        img_path,
        true_label
    )

    count += 1

    if count == 20:
        break

print("\nGradCAM generation completed.")