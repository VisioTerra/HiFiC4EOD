This document allows you to use Justin Tan's fork code with the following cuda/python versions:   
installed python version: <span style="color:red;">3.10</span>  
cuda version installed: https://developer.nvidia.com/cuda-12-1-0-download-archive
  
It traces the errors encountered and the modifications made to get the environment working.  
  
  
Problem installing package absl-py==0.9.0
(Python version 2.7 or 3.4+ is required.) I was in python 3.10
Installed the latest version of absl-py
```python
pip install absl-py
Successfully installed absl-py-2.1.0
```
ERROR: Could not build wheels for grpcio, which is required to install pyproject.toml-based projects
```python
pip install grpcio
Successfully installed grpcio-1.60.1
```

pip install matplotlib==3.3.0 not working
```python
pip install matplotlib
Successfully installed contourpy-1.2.0 fonttools-4.48.1 kiwisolver-1.4.5 matplotlib-3.8.2 packaging-23.2
```
pip install numpy==1.19.1 not working
```python
pip install numpy
Requirement already satisfied: numpy in d:\visioterra\technique\p382_esrin_copex-dcc\engineering\hific\pytorch_nightly\lib\site-packages (1.26.2)
```
pip install Pillow==8.3.2 not working
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
pip install scipy==1.5.2 not working
```python
pip install scipy
Requirement already satisfied: scipy in d:\visioterra\technique\p382_esrin_copex-dcc\engineering\hific\pytorch_nightly\lib\site-packages (1.12.0)
```
pip install torch==1.6.0 not working -> torch version made for pytorch nightly + GPU
python
```python
pip --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
already satisfied
```
pip install torchvision==0.7.0 not working
```python
pip install torchvision
already satisfied
```
for tensorboard to be recognized, needed tensorboard 2.16.1  

TOTAL LIST OF ADDITIONAL IMPORTS TO BE DONE BY HAND (see requirement.txt for version numbers):
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
pip install panel
```
Once imported, you can go check that the code works, use this command to check if env is working:
```python
cd high-fidelity-generative-compression-master
python -m high-fidelity-generative-compression-master/src.model
```
_ImportError: cannot import name 'compare_ssim' from 'skimage.measure' (D:\VisioTerra\technique\P382_ESRIN_COPEX-DCC\engineering\HiFiC.git\HiFiC\pytorch_nightly_venv\lib\site-packages\skimage\measure\__init__.py)_  
**should appear in the terminal, go to perceptual loss , and modify :**.  
`from skimage.measure import compare_ssim
` to `from skimage.metrics import structural_similarity`