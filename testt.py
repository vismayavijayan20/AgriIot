import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model(

   "C:\\Users\\vismaya vijayan\\OneDrive\\Desktop\\PROJECT\\AgriIoT\\model\\crop_disease_model_best.keras"
   )
print("✅ Model loaded")

# training labels order
disease_classes = [
    "Pepper – Foot Rot",                              # 0
    "Pepper – Pollu Disease",                         # 1
    "Pepper – Slow Decline",                          # 2
    "Pepper Bell – Bacterial Spot",                   # 3
    "Pepper Bell – Healthy",                          # 4

    "Potato – Early Blight",                          # 5
    "Potato – Late Blight",                           # 6
    "Potato – Healthy",                               # 7

    "Tomato – Yellow Leaf Curl Virus",                # 8
    "Tomato – Bacterial Spot",                        # 9
    "Tomato – Early Blight",                          # 10
    "Tomato – Late Blight",                           # 11
    "Tomato – Leaf Mold",                             # 12
    "Tomato – Septoria Leaf Spot",                    # 13
    "Tomato – Spider Mites (Two-spotted)",            # 14
    "Tomato – Target Spot",                           # 15
    "Tomato – Mosaic Virus",                          # 16
    "Tomato – Healthy",                               # 17

    "Paddy – Bacterial Leaf Blight",                  # 18
    "Paddy – Blast",                                  # 19
    "Paddy – Brown Spot",                             # 20
    "Paddy – Eye Spot",                               # 21
    "Paddy – Healthy Rice Leaf",                      # 22
    "Paddy – Leaf Smut",                              # 23
    "Paddy – Narrow Brown Leaf Spot",                 # 24
    "Paddy – Rice Hispa",                             # 25
    "Paddy – Sheath Rot",                             # 26
    "Paddy – Sheath Spot",                            # 27
    "Paddy – Tungro",                                 # 28
    "Paddy – Crown Sheath Rot",                       # 29
    "Paddy – Leaf Scald",                             # 30
    "Paddy – Powdery Mildew",                         # 31
    "Paddy – Sheath Blight",                          # 32
    "Paddy – Yellow Mottle",                          # 33

    "Papaya – Anthracnose",                           # 34
    "Papaya – Bacterial Spot",                        # 35
    "Papaya – Curl Disease",                          # 36
    "Papaya – Healthy",                               # 37
    "Papaya – Mealybug / Whitefly",                   # 38
    "Papaya – Mite Disease",                          # 39
    "Papaya – Mosaic",                                # 40
    "Papaya – Ringspot"                               # 41
]


IMG_SIZE = 224

# Load and preprocess image
img = cv2.imread("C:\\Users\\vismaya vijayan\\Downloads\\RiceBlast1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
pred_index = np.argmax(predictions)
confidence = float(np.max(predictions))

# Output
print("🌿 Predicted Disease :", disease_classes[pred_index])
print("🎯 Confidence        :", round(confidence * 100, 2), "%")
