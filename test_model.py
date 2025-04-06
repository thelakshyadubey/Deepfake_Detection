import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array

# Check GPU Availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", tf.test.gpu_device_name())

# Load trained model
model_path = "C:/Lakshya/NMIMS/III Year/SEM VI/Biometrics/Deepfake_Detection/models/new_deepfake_detector.h5"

if not os.path.exists(model_path):
    print(f" Error: Model file not found at {model_path}")
    exit()

model = tf.keras.models.load_model(model_path)
print("Model Loaded Successfully!")

# Define test image path
# image_path = "C:/Lakshya/NMIMS/III Year/SEM VI/Biometrics/Deepfake_Detection/dataset/test/fake/00004.jpg"  # Change for testing

# # Define test image path
# image_path = "C:/Lakshya/NMIMS/III Year/SEM VI/Biometrics/Deepfake_Detection/dataset/test/real/0A266M95TD.jpg"  # Change for testing

# # Define test image path
image_path = "C:/Lakshya/NMIMS/III Year/SEM VI/Biometrics/Deepfake_Detection/Personal Dataset/Fake_pr/4.jpg"  # Change for testing

# Ensure the image exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit()

# Load and preprocess image
try:
    img = load_img(image_path, target_size=(224, 224))  # Load and resize
    img = img_to_array(img) / 255.0  # Convert to array & normalize
    img = np.expand_dims(img, axis=0).astype("float32")  # Expand dims for batch processing
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

#Predict
prediction = model.predict(img)[0][0]
confidence = prediction * 100  # Convert to percentage

#Print Result with Confidence Score
if prediction > 0.5:
    print(f"🛑 Fake Face Detected! (Confidence: {confidence:.2f}%)")
else:
    print(f"✅ Real Face Detected! (Confidence: {100 - confidence:.2f}%)")
