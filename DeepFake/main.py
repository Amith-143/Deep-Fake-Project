import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("deepfake_model.h5")

img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

print("Confidence:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("REAL IMAGE")
else:
    print("FAKE IMAGE")