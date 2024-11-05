import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("fracture_detection_model_advanced_v2.h5")

# Function to preprocess images
def preprocess_image(image_path):
    # Load image
    img = imread(image_path)
    # Convert to grayscale if necessary
    if img.ndim == 3:
        img = rgb2gray(img)
    # Resize the image to the input shape of the model
    img_resized = resize(img, (128, 128), preserve_range=True, anti_aliasing=True)
    # Reshape to match the model's input shape
    img_resized = img_resized.reshape(-1, 128, 128, 1)
    return img_resized

# Function to predict images
def predict_images(folder_path):
    predictions = []
    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predictions.append(prediction)
    return predictions

# Path to the folder containing fractured images
folder_path = ".venv/archive/data/val/fractured"
predictions = predict_images(folder_path)

# Calculate the number of correctly and incorrectly identified images
correct_count = 0
incorrect_count = 0
for prediction in predictions:
    if prediction[0][0] < 0.5:  # Predicted as fractured
        correct_count += 1
    else:  # Predicted as not fractured
        incorrect_count += 1

folder_path = ".venv/archive/data/val/not fractured"
predictions = predict_images(folder_path)
for prediction in predictions:
    if prediction[0][0] > 0.5:  # Predicted as not fractured
        correct_count += 1
    else:  # Predicted as fractured
        incorrect_count += 1

print(f"Correctly identified images: {correct_count}")
print(f"Incorrectly identified images: {incorrect_count}")
