# extract_sample_digits.py - Extract and save one random sample image for each digit (0-9)
# This program loads the MNIST dataset and saves a random image for each digit class

import os
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

# Create output directory for storing digit images
output_dir = "sample_digits"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Load MNIST test set
print("Loading MNIST dataset...")
(_, _), (x_test, y_test) = mnist.load_data()

# For each digit (0-9), find a random sample and save it
print("Extracting random samples for each digit...")
for digit in range(10):
    # Find all indices where the label matches the current digit
    indices = np.where(y_test == digit)[0]
    
    if len(indices) == 0:
        print(f"Warning: No samples found for digit {digit}")
        continue
    
    # Pick a random index for this digit
    random_idx = np.random.choice(indices)
    image = x_test[random_idx]
    
    # Save the image as PNG
    filename = os.path.join(output_dir, f"digit_{digit}.png")
    cv2.imwrite(filename, image)
    print(f"Saved: {filename} (actual label: {y_test[random_idx]})")

print(f"\nAll digit samples saved to '{output_dir}' folder!")
print(f"Total files created: 10")
