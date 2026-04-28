import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Dummy data (simulation)
X = np.random.rand(100, 224, 224, 3)
y = np.random.randint(0, 2, 100)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=2)

model.save("deepfake_model.h5")

print("Model ready")