import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess # IMP: Match app.py logic
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization

# ... (existing imports)

# ... (skipping to fine-tuning section)


from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import os

# =========================
# PATH & CONSTANTS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "crop_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40

print(f"✅ Data Directory: {dataset_path}")
print(f"✅ Model Directory: {MODEL_DIR}")

# =========================
# DATA GENERATORS
# =========================
# CRITICAL: We use 'mobile_preprocess' because app.py uses it.
# MobileNetV2 normalization is [-1, 1].
# ResNet50V2 expects [-1, 1] as well.
# Perfect match. No rescaling needed.

train_datagen = ImageDataGenerator(
    preprocessing_function=mobile_preprocess,
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
    preprocessing_function=mobile_preprocess,
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

# SAVE CLASS ORDER
class_names = list(train_data.class_indices.keys())
json.dump(class_names, open(os.path.join(MODEL_DIR, "class_names.json"), "w"))

# =========================
# MODEL ARCHITECTURE (ResNet50V2)
# =========================
base_model = ResNet50V2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False # Start frozen

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
output_layer = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

model.summary()

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CALLBACKS
# =========================
checkpoint_path = os.path.join(MODEL_DIR, "crop_disease_model_best.keras")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# =========================
# PHASE 1: TRAIN HEAD
# =========================
print("\n🚀 Starting Training (Phase 1: Frozen Base)...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# PHASE 2: FINE TUNING
# =========================
print("\n🔓 Starting Fine-Tuning (Phase 2: Unfrozen)...")
base_model.trainable = True

# Freeze bottom layers, adapt top layers
# ResNet is deep, maybe freeze first 100 layers or so.
# ResNet50V2 has ~190 layers.
# Let's unfreeze the last block.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# CRITICAL FIX: Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5), # Low LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    epochs=EPOCHS + 20,
    initial_epoch=history.epoch[-1],
    validation_data=val_data,
    callbacks=callbacks
)

# Merge history
for k in history.history.keys():
    history.history[k].extend(history_fine.history[k])

# =========================
# SAVE & PLOT
# =========================
model.save(os.path.join(MODEL_DIR, "crop_disease_model_resnet_final.keras"))

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.savefig(os.path.join(BASE_DIR, "training_plot_resnet.png"))
print("✅ Training Complete. Model saved as 'crop_disease_model_best.keras'.")
