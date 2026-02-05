import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "crop_dataset")
AUGMENT_FACTOR = 10  # Number of new images to generate per original image
IMG_SIZE = 224

# Classes to augment (listing explicit small classes found or augment all)
# If empty, it will augment ALL classes.
TARGET_CLASSES = [
    "Pepper,_Footrot",
    "Pepper,_Pollu_Disease",
    "Pepper,_Slow-Decline",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "paddy_Bacterial_leaf_blight",
    "paddy_Blast",
    "paddy_Brown spot",
    "paddy_Eye spot",
    "paddy_Healthy Rice Leaf",
    "paddy_Leaf smut",
    "paddy_Narrow brown leaf spot",
    "paddy_Rice Hispa",
    "paddy_Sheath rot",
    "paddy_Sheath spot",
    "paddy_Tungro1",
    "paddy_bacterial_leaf_streak",
    "paddy_crown sheath rot",
    "paddy_leaf scald",
    "paddy_powdery mildew",
    "paddy_sheath blight",
    "paddy_white_stem_borer",
    "paddy_yellow mottle1",
    "paddy_yellow_stem_borer",
    "papaya_Anthracnose",
    "papaya_BacterialSpot",
    "papaya_Curl",
    "papaya_Healthy",
    "papaya_Mealybug and whitefly",
    "papaya_Mite disease",
    "papaya_Mosaic",
    "papaya_Ringspot"
]

# =========================
# AUGMENTATION SETUP
# =========================
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

def augment_images():
    print(f"🚀 Starting Data Augmentation...")
    print(f"📂 Dataset Directory: {DATASET_DIR}")
    
    # Walk through dataset
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        
        # Skip files, only process directories
        if not os.path.isdir(class_path):
            continue
            
        # Filter if TARGET_CLASSES is set
        if TARGET_CLASSES and class_name not in TARGET_CLASSES:
            # check if it has very few images, maybe we should auto-detect?
            # For now, let's just log and skip if not in target
            # print(f"Skipping {class_name} (not in target list)")
            continue

        print(f"\nProcessing Class: {class_name}")
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('aug_')]
        print(f"   Found {len(images)} original images.")
        
        if len(images) == 0:
            print("   ⚠️ No images found, skipping.")
            continue
            
        # Create 'augmented' folder inside the class folder NOT to mix (optional)
        # BUT standard practice is often to correct imbalance or just dump in same folder.
        # Let's save in the same folder with prefix 'aug_' to make it easy to identify/delete.
        
        count_generated = 0
        
        for idx, img_name in enumerate(images):
            if idx % 10 == 0:
                print(f"   Processing image {idx+1}/{len(images)}...")
            img_path = os.path.join(class_path, img_name)
            
            try:
                # Load and reshape
                img = load_img(img_path)  # PIL image
                x = img_to_array(img)     # (h, w, 3)
                x = x.reshape((1,) + x.shape) # (1, h, w, 3)
                
                # Generate batches
                i = 0
                for batch in datagen.flow(x, batch_size=1, 
                                          save_to_dir=class_path, 
                                          save_prefix='aug', 
                                          save_format='jpg'):
                    i += 1
                    count_generated += 1
                    if i >= AUGMENT_FACTOR:
                        break  # Stop after generating AUGMENT_FACTOR images
                        
            except Exception as e:
                print(f"   ❌ Error processing {img_name}: {e}")
                
        print(f"   ✅ Generated {count_generated} new images for {class_name}.")

if __name__ == "__main__":
    augment_images()
