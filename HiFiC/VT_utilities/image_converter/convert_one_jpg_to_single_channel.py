from PIL import Image
import numpy as np

# Chargez votre image JPEG
image_path = "input/ad249bba099568403dc6b97bc37f8d74.png"
image = Image.open(image_path)
# Convertissez l'image en mode L (un seul canal)
image_single_channel = image.convert("L")

# Affichez la forme (shape) de l'image originale
print("Shape de l'image originale:", np.array(image).shape)

# Affichez la forme (shape) de l'image à un seul canal
print("Shape de l'image à un seul canal:", np.array(image_single_channel).shape)



# Enregistrez l'image résultante
image_single_channel.save("output/ad249bba099568403dc6b97bc37f8d74.jpg")

# Affichez un message pour confirmer la conversion
print("Image convertie avec succès en un seul canal.")
