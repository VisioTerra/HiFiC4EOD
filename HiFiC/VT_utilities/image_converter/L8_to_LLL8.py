import numpy as np
from PIL import  Image
import os

from tqdm import tqdm


def convertir_images(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    print("Conversion de 16bit 1 canal vers 8bit 1 canal")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtenez la liste des fichiers JPG dans le dossier d'entrée
    files = [filename for filename in os.listdir(input_folder)]

    # Parcourez toutes les images dans le dossier d'entrée
    for filename in tqdm(files, desc="Conversion en cours", unit="image"):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(input_folder, filename))
            # Merge the channels back into a new image
            new_img = Image.merge('RGB', (img,img,img))
            """
            print("img = ", img)
            print("new img =", new_img)"""

            output_path = os.path.join(output_folder, filename.split(".")[0]+".png")
            #print("output path = ",output_path)
            new_img.save(output_path, bitmap_format="png")


    print("Conversion terminée avec succès!")

# Utilisation de la fonction avec les dossiers d'entrée et de sortie spécifiés
folder = "/train"
input_folder = "input/test/"
output_folder = "output/test/"
convertir_images(input_folder, output_folder)
