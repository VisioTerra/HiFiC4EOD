version de cuda installée : https://developer.nvidia.com/cuda-12-1-0-download-archive

j'ai eu un probleme pour installer le package absl-py==0.9.0
(Python version 2.7 or 3.4+ is required.) mais j'etais en python 3.11
j'ai donc juste installer la derniere version de absl-py
```python
pip install absl-py
Successfully installed absl-py-2.1.0
```
ERROR: Could not build wheels for grpcio, which is required to install pyproject.toml-based projects
j'ai juste
```python
pip install grpcio
Successfully installed grpcio-1.60.1
```

pip install matplotlib==3.3.0 ne fonctionnait pas )
donc 
```python
pip install matplotlib
Successfully installed contourpy-1.2.0 fonttools-4.48.1 kiwisolver-1.4.5 matplotlib-3.8.2 packaging-23.2
```
pip install numpy==1.19.1 ne fonctionne pas
```python
pip install numpy
Requirement already satisfied: numpy in d:\visioterra\technique\p382_esrin_copex-dcc\engineering\hific\pytorch_nightly\lib\site-packages (1.26.2)
```
pip install Pillow==8.3.2 ne fonctionne pas
```python
pip install Pillow
Requirement already satisfied: Pillow in d:\visioterra\technique\p382_esrin_copex-dcc\engineering\hific\pytorch_nightly\lib\site-packages (9.3.0)
```
pip install PyWavelets==1.1.1 not working
```python
pip install PyWavelets
Successfully installed PyWavelets-1.5.0
```
pip install scikit-image==0.17.2 not working
```python
pip install scikit-image
Successfully installed imageio-2.34.0 lazy_loader-0.3 networkx-3.2.1 scikit-image-0.22.0 scipy-1.12.0 tifffile-2024.2.12
```
pip install scipy==1.5.2
```python
pip install scipy
Requirement already satisfied: scipy in d:\visioterra\technique\p382_esrin_copex-dcc\engineering\hific\pytorch_nightly\lib\site-packages (1.12.0)
```
pip install torch==1.6.0 ne fonctionne pas
```python
pip install torch
already satisfied
```
pip install torchvision==0.7.0 ne fonctionne pas
```python
pip install torchvision
already satisfied
```
for tensorboard to be recognized, needed tensorboard 2.16.1
LISTE TOTAL DES IMPORTS EN PLUS A FAIRE A LA MAIN (voir requirement.txt pour les numéro de version):
```python
pip install absl-py
pip install grpcio
pip install matplotlib
pip install numpy
pip install Pillow
pip install PyWavelets
pip install scikit-image
pip install scipy
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install torchvision
pip install tdqm
pip install autograd
pip install tensorboard
pip install pandas
pip install tables
```
Une fois importés, vous pouvez vous rendre dans le dossier et vérifier que le code marche, refaire cette commande dès que vous aurez corriger tout le code :
```python
cd high-fidelity-generative-compression-master
python -m src.model
```
_ImportError: cannot import name 'compare_ssim' from 'skimage.measure' (D:\VisioTerra\technique\P382_ESRIN_COPEX-DCC\engineering\HiFiC.git\HiFiC\pytorch_nightly_venv\lib\site-packages\skimage\measure\__init__.py)_
**devrait apparaitre dans le terminal, rendez vous dans perceptual loss , et supprimez :**
`from skimage.measure import compare_ssim
` & `from skimage.measure import compare_ssim`

Il faut maintenant
des changements ont été apportés a l'appel des différents paquets (les imports ne fonctionnent pas encore)