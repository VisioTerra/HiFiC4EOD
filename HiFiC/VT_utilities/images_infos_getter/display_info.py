import numpy as np
from PIL import Image
import sys

def display_image_info(image_path):
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            img_type = img.mode
            img_shape = img.size
            img_max_val = np.max(img_array)
            img_min_val = np.min(img_array)

            print(f"Image Type: {img_type}")
            print(f"Image Shape: {img_shape}")
            print(f"Max Value: {img_max_val}")
            print(f"Min Value: {img_min_val}")
            print("Array Preview:")
            print(img_array[:3, :3])
    except Exception as e:
        print(f"Error: {e}")

if len(sys.argv) != 3:
    print("Usage: python script.py <image1_path> <image2_path>")
    sys.exit(1)

image1_path = sys.argv[1]
image2_path = sys.argv[2]

print("Info for Image 1:", image1_path)
display_image_info(image1_path)

print("\nInfo for Image 2:", image1_path)
display_image_info(image2_path)
