import numpy as np
from PIL import Image
import os

from tqdm import tqdm


def convertir_images(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    print("Conversion de RGB8 vers L8 de ", input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtenez la liste des fichiers JPG dans le dossier d'entrée
    files = [filename for filename in os.listdir(input_folder)]

    # Parcourez toutes les images dans le dossier d'entrée
    for filename in tqdm(files, desc="Conversion en cours", unit="image"):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(input_folder, filename))
            # if img is only L, then it become LLL
            img = img.convert('RGB')
            r, g, b = img.split()
            # Apply the L formula to each channel
            new_r = np.array(r) * (299 / 1000)
            new_g = np.array(g) * (587 / 1000)
            new_b = np.array(b) * (114 / 1000)
            l_array = new_r + new_b + new_g
            # Convert the arrays back to Image objects
            l = Image.fromarray(l_array.astype('uint8'))
            # Merge the channels back into a new image
            output_path = os.path.join(output_folder, filename.split(".")[0] + ".png")
            l.save(output_path, bitmap_format="png")

    print("Conversion terminée avec succès!")


# Utilisation de la fonction avec les dossiers d'entrée et de sortie spécifiés
datasets = {"OID7_RGB8_100", "OID7_RGB8_1000", "OID7_RGB8_10000"}
folders = ["train", "test", "validation"]
for dataset in datasets:
    for folder in folders:
        input_folder = os.path.join("input", dataset, folder)
        output_folder = os.path.join("output", dataset, folder)
        convertir_images(input_folder, output_folder)
