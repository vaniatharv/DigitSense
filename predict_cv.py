# predict_cv.py - Script for testing the trained model on external images.
# It loads a saved model and predicts the digit in a provided image file.

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Computer vision library

# Load the trained model
try:
    model = tf.keras.models.load_model("saved_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Ask user which digit sample to predict
print("\n================================\n")
print("Available digit samples: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
while True:
    try:
        user_input = input("Enter the digit number (0-9) to predict: ").strip()
        digit_num = int(user_input)
        if 0 <= digit_num <= 9:
            break
        else:
            print("Please enter a number between 0 and 9!")
    except ValueError:
        print("Invalid input! Please enter a number between 0 and 9.")
print("\n================================\n")

# Load the selected digit image
img_path = os.path.join("sample_digits", f"digit_{digit_num}.png")
if not os.path.exists(img_path):
    print(f"Error: Image file '{img_path}' not found!")
    print("Please run 'extract_sample_digits.py' first to generate sample images.")
    exit(1)

try:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to load image")
    print(f"Loaded image: {img_path}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Preprocess image to match model input
image = cv2.resize(image, (28, 28))     # Resize to 28x28
image = image / 255.0                   # Normalize (0–1)
image = np.expand_dims(image, axis=-1)  # Add channel dimension
image = np.expand_dims(image, axis=0)   # Add batch dimension

# Predict using model
y_prob = model.predict(image)
pred_class = np.argmax(y_prob)
confidence = y_prob[0][pred_class]

# Display result
plt.imshow(image[0].squeeze(), cmap='gray')
plt.title(f"Predicted: {pred_class} | Confidence: {confidence:.2f}")
plt.axis('off')
plt.show()
print("\n================================\n")

print(f"Prediction probabilities:\n{np.round(y_prob)}")

print("\n================================\n")
