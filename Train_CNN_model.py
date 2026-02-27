import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import os

# =========================
# PATH & CONSTANTS
# =========================
# FIXED: Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
dataset_path = os.path.join(PROJECT_ROOT, "crop_dataset_balanced")

# Ensure model directory exists
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50   # EarlyStopping will choose optimal epoch

print(f"✅ Data Directory: {dataset_path}")
print(f"✅ Model Directory: {MODEL_DIR}")

# =========================
# DATA GENERATORS (CNN STYLE)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,           # ✅ CNN needs normalization
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# =========================
# SAVE CLASS ORDER
# =========================
class_names = list(train_data.class_indices.keys())

class_names_path = os.path.join(MODEL_DIR, "class_names.json")
json.dump(class_names, open(class_names_path, "w"))

print("✅ CLASS ORDER USED FOR TRAINING:")
print(train_data.class_indices)

# =========================
# CNN MODEL (FROM SCRATCH)
# =========================
model = Sequential([

    # -------- Block 1 --------
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # -------- Block 2 --------
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # -------- Block 3 --------
    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # -------- Block 4 --------
    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # -------- Classifier --------
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes, activation="softmax")
])

model.summary()

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CALLBACKS
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# FIXED: Save to model dir with fixed name
checkpoint_path = os.path.join(MODEL_DIR, "crop_disease_model_cnn_fixed.keras")
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True
)

callbacks = [early_stop, checkpoint]

# =========================
# TRAIN
# =========================
# Note: verify that data generators are working before running long training
if train_data.samples == 0:
    print("❌ Error: No training data found. Check dataset path.")
    exit(1)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# FIND OPTIMAL EPOCH
# =========================
best_epoch = history.history["val_loss"].index(
    min(history.history["val_loss"])
) + 1

print(f"✅ Optimal Epoch Found: {best_epoch}")

# =========================
# SAVE FINAL MODEL
# =========================
# FIXED: Save to model dir
final_model_path = os.path.join(MODEL_DIR, "crop_disease_model_cnn_fixed_final.keras")
model.save(final_model_path)
print(f"✅ Model saved to {final_model_path}")

# =========================
# SAVE HISTORY
# =========================
history_path = os.path.join(MODEL_DIR, "training_history.json")
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"✅ Training history saved to {history_path}")

# =========================
# PLOT (REPORT / VIVA)
# =========================
# FIXED: Save plot to file instead of just showing
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plot_path = os.path.join(BASE_DIR, "training_plot.png")
plt.savefig(plot_path)
print(f"✅ Plot saved to {plot_path}")

# Still show it if possible
try:
    plt.show()
except:
    pass
