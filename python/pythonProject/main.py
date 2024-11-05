import os
import pickle
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Prepare Data
train_import = ".venv/archive/data/train"
val_import = ".venv/archive/data/val"

categories = ["fractured", "not fractured"]

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(train_import, category)):
        img_path = os.path.join(train_import, category, file)
        img = imread(img_path)

        # Convert image to grayscale
        if img.ndim == 3:
            img = rgb2gray(img)

        try:
            # Resize the image to a consistent size while maintaining aspect ratio
            img_resized = resize(img, (128, 128), preserve_range=True, anti_aliasing=True)
            # print(f"Image size: {img_resized.shape}")
            data.append(img_resized.flatten())
            labels.append(category_idx)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# for category_idx, category in enumerate(categories):
#     for file in os.listdir(os.path.join(val_import, category)):
#         img_path = os.path.join(val_import, category, file)
#         img = imread(img_path)
#
#         # Convert image to grayscale
#         if img.ndim == 3:
#             img = rgb2gray(img)
#
#         try:
#             # Resize the image to a consistent size while maintaining aspect ratio
#             img_resized = resize(img, (128, 128), preserve_range=True, anti_aliasing=True)
#             # print(f"Image size: {img_resized.shape}")
#             data.append(img_resized.flatten())
#             labels.append(category_idx)
#         except Exception as e:
#             print(f"Error processing {img_path}: {e}")

data = np.asarray(data)
labels = np.asarray(labels)


# Train / Test Model
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train Classifier
classifier = SVC()

parameters = [{'gamma' : [0.01, 0.001, 0.0001], 'C' : [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# Test
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print("{}% of samples were correctly classified".format(str(score * 100)))

pickle.dump(best_estimator, open('.model.p', 'wb'))