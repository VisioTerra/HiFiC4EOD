import cv2
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def convertir_images(input_folder, output_folder,clamp_min, clamp_max):
    # Vérifier si le dossier de sortie existe, sinon le créer
    print("Conversion de TIF 16bit 1 canal vers PNG 16bit 1 canal et clamp entre ",clamp_min," et ",clamp_max)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtenez la liste des fichiers JPG dans le dossier d'entrée
    files = [filename for filename in os.listdir(input_folder)]
    original_range = [clamp_min, clamp_max]
    target_range = [0, 255*255]
    # Parcourez toutes les images dans le dossier d'entrée
    for filename in tqdm(files, desc="Conversion en cours", unit="image"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_name_without_extension = os.path.splitext(filename)[0]
            # Charger le fichier TIFF
            tif_image = Image.open(os.path.join(input_folder,filename))

            # Convertir en array numpy
            tif_array = np.array(tif_image)

            # Appliquer le clamp entre 0 et 500
            stretched_array = np.interp(tif_array, original_range, target_range)

            # Créer une nouvelle image avec le tableau modifié
            new_image = Image.fromarray(stretched_array.astype(np.uint16))

            # Sauvegarder en PNG
            new_image.save(os.path.join(output_folder,file_name_without_extension+'.png'),bitmap_format='png')


    print("Conversion terminée avec succès!")

# Utilisation de la fonction avec les dossiers d'entrée et de sortie spécifiés
input_folder = "input/S1A_VV_tif_L16"
output_folder = "output/S1A_VV_rgb_L16_clamp_0_500"
min =  0 #min value for clamp
max = 650 #max value for clamp
convertir_images(input_folder, output_folder,min,max)
