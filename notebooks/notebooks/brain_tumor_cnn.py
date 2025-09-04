# brain_tumor_cnn.py

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ---------------- Data Loading & Preprocessing ----------------
X_data = []
Y_data = []
image_size = 150
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Load training data
for label in labels:
    folder_path = os.path.join('../input/brain-tumor-classification-mri/Training', label)
    for img_name in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_name))
        img = cv2.resize(img, (image_size, image_size))
        X_data.append(img)
        Y_data.append(label)

# Load testing data
for label in labels:
    folder_path = os.path.join('../input/brain-tumor-classification-mri/Testing', label)
    for img_name in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_name))
        img = cv2.resize(img, (image_size, image_size))
        X_data.append(img)
        Y_data.append(label)

# Convert to numpy arrays and shuffle
X_data = np.array(X_data)
Y_data = np.array(Y_data)
X_data, Y_data = shuffle(X_data, Y_data, random_state=101)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=101)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical([labels.index(i) for i in y_train])
y_test = tf.keras.utils.to_categorical([labels.index(i) for i in y_test])

# ---------------- CNN Model ----------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# ---------------- Plot Training Results ----------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# ---------------- Save Model ----------------
model.save('brain_tumor_detection_model.h5')
