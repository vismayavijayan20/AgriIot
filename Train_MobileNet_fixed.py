import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import os

# =========================
# PATH & CONSTANTS
# =========================
# FIXED: Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "crop_dataset")

# Ensure model directory exists
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

print(f"✅ Data Directory: {dataset_path}")
print(f"✅ Model Directory: {MODEL_DIR}")

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
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
# MODEL (MobileNetV2)
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # ✅ Prevent overfitting

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
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
    patience=10,
    restore_best_weights=True
)

# FIXED: Save to model dir with fixed name
checkpoint_path = os.path.join(MODEL_DIR, "crop_disease_model_best.keras")
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True
)

# NEW: Learning Rate Scheduler
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]

# =========================
# TRAIN
# =========================
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
# FINE-TUNING
# =========================
print("\n✅ Starting Fine-Tuning Phase...")

base_model.trainable = True

# Freeze all the layers before the `fine_tune_at` layer
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['accuracy'])

fine_tune_epochs = 20
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_data,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_data,
    callbacks=callbacks
)

# Merge history
for k in history.history.keys():
    history.history[k].extend(history_fine.history[k])

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
final_model_path = os.path.join(MODEL_DIR, "crop_disease_model_best_final.keras")
model.save(final_model_path)
print(f"✅ Model saved to {final_model_path}")

# =========================
# SAVE HISTORY
# =========================
history_path = os.path.join(MODEL_DIR, "training_history_mobilenet.json")
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"✅ Training history saved to {history_path}")

# =========================
# PLOT
# =========================
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

plot_path = os.path.join(BASE_DIR, "training_plot_mobilenet.png")
plt.savefig(plot_path)
print(f"✅ Plot saved to {plot_path}")

try:
    plt.show()
except:
    pass
