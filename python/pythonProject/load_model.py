import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


categories = ["fractured", "not fractured"]
loaded_model = pickle.load(open('.model.p', 'rb'))


def preprocess_image(img_path):
    img = imread(img_path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img_resized = resize(img, (128, 128), preserve_range=True, anti_aliasing=True)
    return img_resized.flatten()


# Predicting a single image
test_image_path = ".venv/archive/data/val/googled/fractured/2.jpg"
preprocessed_image = preprocess_image(test_image_path)
prediction = loaded_model.predict([preprocessed_image])
print("Prediction:", categories[prediction[0]])

# Predicting multiple images
# test_image_paths = ["path_to_image1.jpg", "path_to_image2.jpg", ...]
# for img_path in test_image_paths:
#     preprocessed_image = preprocess_image(img_path)
#     prediction = loaded_model.predict([preprocessed_image])
#     print("Image:", img_path, "Prediction:", prediction)
