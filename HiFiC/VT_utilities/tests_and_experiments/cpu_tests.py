import torch

#from tensorflow.keras.callbacks import tensorboard
print("Torch version:",torch.__version__)
# Vérifier si CUDA est disponible
if torch.cuda.is_available():
    # Afficher la version de CUDA
    print(f"Version de CUDA disponible: {torch.version.cuda}")
else:
    print("CUDA n'est pas disponible sur ce système.")

"""print(np.random.uniform(0.75,0.95))
print(np.random.uniform(0.75,0.95))
print(np.random.uniform(0.75,0.95))
print(np.random.uniform(0.75,0.95))

transforms.Normalize((0.5), (0.5))"""