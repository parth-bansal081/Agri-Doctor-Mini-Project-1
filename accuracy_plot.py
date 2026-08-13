import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# 1. Load saved model and class names
model = tf.keras.models.load_model("plant_disease_model.keras")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# 2. Load dataset directly from folder (no retraining)
dataset = tf.keras.utils.image_dataset_from_directory(
    "data", shuffle=False, image_size=(256, 256), batch_size=16
)

# 3. Collect predictions and extract misclassified images
y_true = []
y_pred = []
misclassified_images = []
misclassified_labels = []

print("Evaluating dataset...")
for images, labels in dataset:
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = labels.numpy()

    y_true.extend(true_classes)
    y_pred.extend(pred_classes)

    # Grab up to 9 misclassified images for visual inspection
    for img, true_label, pred_label in zip(
        images, true_classes, pred_classes
    ):
        if true_label != pred_label and len(misclassified_images) < 9:
            misclassified_images.append(img.numpy().astype("uint8"))
            misclassified_labels.append(
                (class_names[true_label], class_names[pred_label])
            )

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# --- PLOT 1: Confusion Matrix ---
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Reds",
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# --- PLOT 2: Per-Class Error Rates ---
class_errors = []
for i, name in enumerate(class_names):
    mask = y_true == i
    if np.sum(mask) > 0:
        err_rate = 100 * (1 - np.sum(y_pred[mask] == i) / np.sum(mask))
    else:
        err_rate = 0
    class_errors.append(err_rate)

plt.figure(figsize=(10, 6))
plt.barh(class_names, class_errors, color="crimson")
plt.xlabel("Error Rate (%)")
plt.ylabel("Plant Category")
plt.title("Error Rate per Disease Class")
plt.tight_layout()
plt.show()

# --- PLOT 3: Grid of Misclassified Images ---
if misclassified_images:
    plt.figure(figsize=(10, 10))
    for idx, (img, (true_name, pred_name)) in enumerate(
        zip(misclassified_images, misclassified_labels)
    ):
        plt.subplot(3, 3, idx + 1)
        plt.imshow(img)
        # Shorten titles for readability
        t_short = true_name.split("___")[-1]
        p_short = pred_name.split("___")[-1]
        plt.title(f"True: {t_short}\nPred: {p_short}", color="red", fontsize=9)
        plt.axis("off")
    plt.suptitle("Sample Misclassified Images", fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Terminal Classification Summary ---
print("\n--- Detailed Metrics Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))