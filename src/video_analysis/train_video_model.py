import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.metrics import confusion_matrix, classification_report


DATASET_PATH = r"D:\SMART_INTERVIEW_ANALYZER\processed_faces"
MODEL_PATH = r"D:\SMART_INTERVIEW_ANALYZER\models\video_emotion_model.keras"
GRAPH_PATH = r"D:\SMART_INTERVIEW_ANALYZER\reports\video_graphs"

os.makedirs(r"D:\SMART_INTERVIEW_ANALYZER\models", exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10


# Data loading and splitting
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# MobileNetV2 base model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False


# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


# Save model
model.save(MODEL_PATH)
print("Model saved at:", MODEL_PATH)
print("Classes:", train_data.class_indices)


# -------------------------------
# Graph 1: Accuracy graph
# -------------------------------
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_PATH, "training_accuracy.png"))
plt.close()


# -------------------------------
# Graph 2: Loss graph
# -------------------------------
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_PATH, "training_loss.png"))
plt.close()


# -------------------------------
# Graph 3: Confusion matrix
# -------------------------------
val_data.reset()

predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

class_names = list(val_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Video Emotion Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_PATH, "confusion_matrix.png"))
plt.close()


# -------------------------------
# Classification report
# -------------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

report_path = os.path.join(GRAPH_PATH, "classification_report.txt")

with open(report_path, "w") as f:
    f.write(report)

print("Graphs saved at:", GRAPH_PATH)
print("Classification report saved at:", report_path)