import argparse
import math
import os.path

import numpy as np
import torch
from PIL import Image
import sys

from matplotlib import pyplot as plt
from torchvision import transforms, datasets

SCALE_MIN = 0.75
SCALE_MAX = 0.95

# Function to cast image array to int32
def cast_to_int32(x):
    return x.astype(np.int32)
def display_image_modif_8bit(image_path):
    print("8 BITS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    with Image.open(image_path) as img:
        img_array = np.array(img)
        img_shape = img.size
        img_max_val = np.max(img_array)
        img_min_val = np.min(img_array)

        print(f"Original Image:")
        print(f"Image Type: {img.mode}")
        print(f"Image datatype: {img_array.dtype}")
        print(f"Image Shape: {img_shape}")
        print(f"Max Value: {img_max_val}")
        print(f"Min Value: {img_min_val}")
        print("Array Preview:")
        print(img_array[:3, :3])
        crop_size = 256
        W, H = img.size  # slightly confusing

        shortest_side_length = min(H, W)

        minimum_scale_factor = float(crop_size) / float(shortest_side_length)
        scale_low = max(minimum_scale_factor, SCALE_MIN)
        scale_high = max(scale_low, SCALE_MAX)
        scale = np.random.uniform(scale_low, scale_high)
        # Apply the transformation Normalize
        transforms_list = [
            #transforms.RandomHorizontalFlip(),
            #transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
            #transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ]

        dynamic_transform = transforms.Compose(transforms_list)
        transformed = dynamic_transform(img)

        print("\nTransformed Image:")
        # Calcul du minimum et du maximum
        min_value = torch.min(transformed)
        max_value = torch.max(transformed)

        # Affichage des résultats
        print("Minimum:", min_value.item())
        print("Maximum:", max_value.item())
        print(f"Type: {transformed.dtype}")
        print(f"shape: {transformed.size()}")
        print(f"content: {transformed}")

def display_image_modif_16bit(image_path):
    print("16 BITS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    with Image.open(image_path) as img:

        img_array = np.array(img)
        img_shape = img.size
        img_max_val = np.max(img_array)
        img_min_val = np.min(img_array)

        print(f"Original Image:")
        print(f"Image Type: {img.mode}")
        print(f"Image datatype: {img_array.dtype}")
        print(f"Image Shape: {img_shape}")
        print(f"Max Value: {img_max_val}")
        print(f"Min Value: {img_min_val}")
        print("Array Preview:")
        print(img_array[:3, :3])
        crop_size = 256
        W, H = img.size  # slightly confusing

        shortest_side_length = min(H, W)
        minimum_scale_factor = float(crop_size) / float(shortest_side_length)
        scale_low = max(minimum_scale_factor, 0.75)
        scale_high = max(scale_low, 0.95)
        scale = np.random.uniform(scale_low, scale_high)

        # Apply the transformation
        transforms_list = [
            #transforms.RandomHorizontalFlip(),
            #transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
            #transforms.RandomCrop(crop_size)
        ]

        dynamic_transform = transforms.Compose(transforms_list)
        transformed = dynamic_transform(img)
        transformed_array = np.array(transformed)
        print("array shape = ", transformed_array.shape)
        print("array size = ", transformed_array.size)
        print("array dtype = ", transformed_array.dtype)
        # Normalisation entre 0 et 1

        # Conversion du tableau NumPy en tensor PyTorch
        transformed = torch.from_numpy(transformed_array).float().unsqueeze(0) / 65535.0

        #transformed = torch.tensor(transformed_array, dtype=torch.float32) / 65535.0

        # Normalisation entre -1 et 1
        #transformed = 2 * transformed - 1
        print("\ntransformed  :")
        print(f"Type: {transformed.dtype}")
        print(f"shape: {transformed.size()}")
        # Calcul du minimum et du maximum
        min_value = torch.min(transformed)
        max_value = torch.max(transformed)
        # Affichage des résultats
        print("Minimum:", min_value.item())
        print("Maximum:", max_value.item())
        print(f"content: {transformed}")






        #VISUALISER L IMAGE
        # Convertir l'image en tensor en utilisant les transformations torchvision
        image_from_tensor = transforms.ToPILImage()(transformed.squeeze())

        # Afficher l'image à l'aide de matplotlib
        plt.imshow(image_from_tensor, cmap='gray')
        plt.show()

if __name__ == '__main__':

    input_image_name_1 = "8bit_t2m.png"
    input_image_name_2 = "16bit_t2m.png"

    description = "do modifications and look at the results"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-ip1", "--input_path_1", default=os.path.join("inputs",input_image_name_1), help="Input 1 to modify and analyse")
    general.add_argument("-ip2", "--input_path_2", default=os.path.join("inputs",input_image_name_2), help="Input 2 to modify and analyse")

    cmd_args = parser.parse_args()
    input_path_1 = cmd_args.input_path_1
    input_path_2 = cmd_args.input_path_2

    print("modif for Image :", input_image_name_1)
    display_image_modif_8bit(input_path_1)
    display_image_modif_16bit(input_path_2)