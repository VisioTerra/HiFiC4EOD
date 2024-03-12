import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def afficher_histogramme(image_path):
    # Charger l'image en niveaux de gris

    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Obtenir le minimum et le maximum des valeurs de l'image
    img_min = img.min()
    img_max = img.max()
    print("img min = [",img_min,"], max = [",img_max,"]")
    print("image dtype = ",img.dtype)
    if img.dtype == np.uint8 :
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])

    if img.dtype == np.uint16:
        # Calculer l'histogramme
        hist = cv2.calcHist([img], [0], None, [256*256], [0, 255*255])

    # Afficher l'histogramme
    plt.plot(hist)
    plt.title('Histogramme de l\'image')
    plt.xlabel('Niveaux de gris (16 bits)')
    plt.ylabel('Fréquence')
    plt.show()

# Appel de la fonction avec le chemin de l'image en entrée
input = os.path.join("inputs","ad249bba099568403dc6b97bc37f8d74_L16_RECON_0.267bpp.png")
afficher_histogramme(input)
