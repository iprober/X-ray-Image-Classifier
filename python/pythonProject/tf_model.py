import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# Prepare Data
train_import = ".venv/archive/data/train"
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
            data.append(img_resized)
            labels.append(category_idx)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Shuffle the data
data, labels = shuffle(data, labels, random_state=42)

# Train-Validation Split
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Model Architecture
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),  # Specify input shape explicitly
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Early Stopping Callback
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Model Compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Model Evaluation
test_loss, test_accuracy = model.evaluate(x_val, y_val)
print(f'Test Accuracy: {test_accuracy}')

# Save the model
model.save("fracture_detection_model_advanced_v2.h5")

# Visualize training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
