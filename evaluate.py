import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# -------------------------------
# Create Results Folder
# -------------------------------

os.makedirs("results", exist_ok=True)

# -------------------------------
# Load Model
# -------------------------------
model1 = load_model("models/resnet50_feature_extractor.keras")
model2 = load_model("models/resnet50_fine_tuned.keras")


# -------------------------------
# Test Generator
# -------------------------------

IMG_SIZE = (224,224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

test_generator = test_datagen.flow_from_directory(

    "dataset/chest_xray_processed/test",

    target_size=(224,224),

    batch_size=32,

    class_mode="binary",

    shuffle=False

)

# -------------------------------
# Predict
# -------------------------------

pred_prob1 = model1.predict(test_generator).flatten()
pred_prob2 = model2.predict(test_generator).flatten()



pred_class1 = (pred_prob1 > 0.5).astype(int).flatten()
pred_class2 = (pred_prob2 > 0.5).astype(int).flatten()



true_class = test_generator.classes

# -------------------------------
# Classification Report
# -------------------------------


model_names = [
    "Baseline ResNet50",
    "Fine-Tuned ResNet50"
]

reports = [
    classification_report(
        true_class,
        pred_class1,
        target_names=list(test_generator.class_indices.keys())
    ),

    classification_report(
        true_class,
        pred_class2,
        target_names=list(test_generator.class_indices.keys())
    )
]

for i, report in enumerate(reports):

    print("\n")
    print("="*80)
    print(model_names[i])
    print("="*80)
    print(report)

    with open(f"results/classification_report_model{i+1}.txt","w") as f:
        f.write(model_names[i] + "\n\n")
        f.write(report)

# -------------------------------
# Confusion Matrix
# -------------------------------

cms = [
    confusion_matrix(true_class, pred_class1),
    confusion_matrix(true_class, pred_class2)
]

for i, cm in enumerate(cms):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(test_generator.class_indices.keys())
    )

    plt.figure(figsize=(6,6))

    disp.plot(
        cmap="Blues",
        values_format='d'
    )

    plt.title(model_names[i])

    plt.savefig(
        f"results/confusion_matrix_model{i+1}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# -------------------------------
# ROC Curve
# -------------------------------

fpr1,tpr1,_ = roc_curve(true_class,pred_prob1)

fpr2,tpr2,_ = roc_curve(true_class,pred_prob2)


auc1 = roc_auc_score(
    true_class,
    pred_prob1
)

auc2 = roc_auc_score(
    true_class,
    pred_prob2
)


plt.figure(figsize=(7,7))

plt.plot(fpr1,tpr1,
         label=f"Feature Extractor (AUC={auc1:.3f})")

plt.plot(fpr2,tpr2,
         label=f"Fine Tuned (AUC={auc2:.3f})")


plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("results/roc_curve_comparison.png",dpi=300)
plt.close()