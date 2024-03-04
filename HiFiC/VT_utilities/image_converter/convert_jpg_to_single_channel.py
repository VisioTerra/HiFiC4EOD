#convertis tous les jpg du dossier input et les ajoute dans le dossier output par defaut ( peut etre spécifié )
import argparse
import os
from PIL import Image
import sys
from tqdm import tqdm  # Importez la classe tqdm pour la barre de progression

def convert_to_single_channel(input_folder, output_folder):
    # Vérifiez si le dossier de sortie existe, sinon créez-le
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtenez la liste des fichiers JPG dans le dossier d'entrée
    jpg_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".jpg")]

    # Parcourez toutes les images dans le dossier d'entrée
    for filename in tqdm(jpg_files, desc="Conversion en cours", unit="image"):
        if filename.endswith(".jpg"):
            # Chargez l'image
            input_path = os.path.join(input_folder, filename)
            image = Image.open(input_path)

            # Convertissez l'image en mode L (un seul canal)
            image_single_channel = image.convert("L")

            # Enregistrez l'image résultante dans le dossier de sortie
            output_path = os.path.join(output_folder, filename)
            image_single_channel.save(output_path)

    print("Conversion terminée avec succès.")

if __name__ == "__main__":
    # Vérifiez si les arguments de ligne de commande sont fournis
    description = "convert jpg to L format (1 channel for luminance)"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-i", "--input", default=os.path.join("image_converter","input"), help="path of input folder.")
    general.add_argument("-o", "--output",default=os.path.join("image_converter","output"), help="path of output folder.")
    # Parse the command line arguments
    args = parser.parse_args()
    # Access the values of input_folder and output_folder
    input_folder = args.input
    output_folder = args.output
    print("input = ",input_folder)
    print("output folder = ",output_folder)


    # Appelez la fonction de conversion
    convert_to_single_channel(input_folder, output_folder)
