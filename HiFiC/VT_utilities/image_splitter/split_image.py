import cv2
import os

from PIL import Image
from tqdm import tqdm


def create_folder_if_not_exists(folder_path):
    """
    Crée un dossier s'il n'existe pas déjà.

    Args:
        folder_path (str): Le chemin du dossier à créer.

    Returns:
        str: Le chemin du dossier créé.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def reconstruct_image(input_folder, output_path, factor):
    # Liste pour stocker les sous-images dans l'ordre de reconstruction
    tiles = []

    # Boucler à travers toutes les tuiles et les ajouter à la liste
    for i in range(factor):
        for j in range(factor):
            tile_path = os.path.join(input_folder, f"image_{i}_{j}.png")
            tiles.append(Image.open(tile_path))

    # Récupérer la taille de la première tuile pour déterminer les dimensions de l'image
    tile_width, tile_height = tiles[0].size

    # Calculer les dimensions de l'image finale
    width = tile_width * factor
    height = tile_height * factor

    # Créer une nouvelle image vide
    reconstructed_image = Image.new('L', (width, height))

    # Reconstruire l'image en assemblant les tuiles
    for i in range(factor):
        for j in range(factor):
            # Coordonnées de la région à coller
            left = j * tile_width
            upper = i * tile_height
            right = (j + 1) * tile_width
            lower = (i + 1) * tile_height

            # Coller la tuile dans l'image finale
            reconstructed_image.paste(tiles[i * factor + j], (left, upper, right, lower))

    # Enregistrer l'image reconstruite
    reconstructed_image.save(output_path)


def split_image(input_path, output_folder, factor):
    print("splitting ", input_path, " to ", output_folder)
    # Ouvrir l'image d'entrée
    img = Image.open(input_path)

    # Récupérer les dimensions de l'image
    width, height = img.size

    # Calculer les dimensions des sous-images
    sub_width = width // factor
    sub_height = height // factor

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Diviser l'image en sous-images
    for i in range(factor):
        for j in range(factor):
            # Coordonnées de la région à extraire
            left = j * sub_width
            upper = i * sub_height
            right = (j + 1) * sub_width
            lower = (i + 1) * sub_height

            # Extraire la région de l'image
            sub_img = img.crop((left, upper, right, lower))

            # Enregistrer la sous-image
            sub_img.save(os.path.join(output_folder, f"image_{i}_{j}.png"))

    print("splitting done with succes")


# Utilisation de la fonction avec les dossiers d'entrée et de sortie spécifiés
facteur_de_division = 8
image_name = "S1A_IW_GRDH_1SDV_20170915T043511_20170915T043536_018381_01EF06_1A44_12.png"
input_image = os.path.join("input", image_name)
output_folder = os.path.join("output", str(image_name), str(facteur_de_division))
create_folder_if_not_exists(output_folder)
split_image(input_image, output_folder, facteur_de_division)
