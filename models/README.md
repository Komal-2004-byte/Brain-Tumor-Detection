from keras.models import load_model
import cv2
import numpy as np

# Load pre-trained model
model = load_model('models/brain_tumor_detection_model.h5')

# Example: predict on a sample image
img = cv2.imread('sample_image.jpg')  # users can provide their own image
img = cv2.resize(img, (150, 150))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print(prediction)




