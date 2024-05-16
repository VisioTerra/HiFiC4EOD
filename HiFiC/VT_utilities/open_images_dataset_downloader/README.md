follow procedure at https://storage.googleapis.com/openimages/web/download_v7.html#download-manually

put the images you want to download in input and then put this in terminal :
```python
python ./open_images_dataset_downloader\open_images_dataset_downloader.py .\open_images_dataset_downloader\input\open_images_dataset_files_to_download.txt --download_folder=.\open_images_dataset_downloader\output\100_img --num_processes=5
```

format type d'un .txt :

> <p>train/4fa8054781a4c382</p><p>train/b37f763ae67d0888</p>
