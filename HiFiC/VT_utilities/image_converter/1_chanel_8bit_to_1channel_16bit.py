import cv2
import os

import numpy as np
from tqdm import tqdm


def convertir_images(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    print("Conversion de 8bit 1 canal vers 16bit 1 canal")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtenez la liste des fichiers JPG dans le dossier d'entrée
    files = [filename for filename in os.listdir(input_folder)]

    # Parcourez toutes les images dans le dossier d'entrée
    for filename in tqdm(files, desc="Conversion en cours", unit="image"):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Charger l'image en niveaux de gris (1 canal, 16 bits)
            img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_UNCHANGED)
            print(img[-2:])
            img = np.uint16(img) * 256
            print(img[-2:])
            # Enregistrer l'image dans le dossier de sortie
            output_filename=os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, img)
        break

    print("Conversion terminée avec succès!")

# Utilisation de la fonction avec les dossiers d'entrée et de sortie spécifiés
folder = "/train"
input_folder = "input"
output_folder = "output"
convertir_images(input_folder, output_folder)
