from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

IMG_SIZE = 224

def preprocess_image(path):
    # Load image ensuring RGB format and resizing
    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    
    # Expand dimensions (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess input using MobileNetV2 logic (scales to -1 to 1)
    return preprocess_input(img_array)