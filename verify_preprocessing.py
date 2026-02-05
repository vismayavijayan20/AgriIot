import os
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image

#Create dummy image if not exists
dummy_path = "test_image_verify.jpg"
if not os.path.exists(dummy_path):
    # Create a random image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tf.keras.utils.save_img(dummy_path, img)

print(f"Testing preprocessing on {dummy_path}...")
processed = preprocess_image(dummy_path)

min_val = np.min(processed)
max_val = np.max(processed)
shape = processed.shape

print(f"Shape: {shape}")
print(f"Min Value: {min_val}")
print(f"Max Value: {max_val}")

if -1.1 <= min_val and max_val <= 1.1 and min_val < 0:
    print("✅ SUCCESS: Values are in range [ -1, 1 ]")
else:
    print("❌ FAILURE: Values are NOT in range [ -1, 1 ]")

if shape == (1, 224, 224, 3):
    print("✅ SUCCESS: Output shape is correct.")
else:
    print("❌ FAILURE: Output shape is incorrect.")
