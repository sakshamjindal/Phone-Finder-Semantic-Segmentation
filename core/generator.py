import torch
import os
import glob
from PIL import Image
import scipy
import numpy as np
import zipfile
import gdown
from sklearn.model_selection import train_test_split

from core.utils import random_paste

datagen_configs = {
    "gdrive_bg_url" : "https://drive.google.com/uc?id=13LBYN6eTfV9G9xdgZtdpNHrXSA8mpv-2",
    "gdrive_bg_output_path" : "/tmp/Splited.zip",
    "bg_images_output_dir" : "/tmp/Textures",
    "gdrive_stock_url": "https://drive.google.com/uc?id=18WckfN5c-tsZ0Qy6slEmpGOA-XOcgHlF",
    "gdrive_stock_output_path" : "/tmp/iphone.png"
}

class SyntheticDataGenerator():

    def __init__():

        self._download_bg_images()
        self._download_stock_image()

    def _download_bg_images(self):

        # Download the dataset of texture images
        # Dataset taken from here : https://github.com/abin24/Textures-Dataset
        gdown.download(datagen_configs["gdrive_bg_url"], datagen_configs["gdrive_bg_output_path"], quiet=False)
        os.makedirs(datagen_configs["bg_images_output_dir"], exist_ok = True)

        with zipfile.ZipFile(datagen_configs["gdrive_bg_output_path"], 'r') as zip_ref:
            zip_ref.extractall(datagen_configs["bg_images_output_dir"])

    def _download_stock_image(self):

        # Download the stock iphone photo
        gdown.download(datagen_configs["gdrive_stock_url"], datagen_configs["gdrive_stock_output_path"], quiet=False)

    def generate_synthetic_dataset(self, folder = "train", split = 0.2):

        # load the background images in the memory
        background_image_root = os.path.join(datagen_configs["bg_images_output_dir"], "train")
        bg_img_paths = glob.glob(os.path.join(background_image_root, "*/*.jpg"))
        bg_imgs = [np.array(Image.open(x)) for x in bg_img_paths]

        # load the stock image in memory
        iphone_img_path = datagen_configs["gdrive_stock_output_path"]
        img = (Image.open(iphone_img_path))

        # iterate over bg images and paste the iphone onto it
        training_set = []  # image, segmentation mask
        for bg_image in bg_imgs:
            aug_image, aug_mask = random_paste((Image.fromarray(bg_image)).copy(), img.copy())
            # convert PIL images to pytorch tensors
            training_pair = [
                np.array(aug_image)[:,:, :3],  # keep the rgb only
                # For the mask, we only need the last (4th) channel,
                # and we will encode the mask as boolean
                np.array(aug_mask)[:,:,-1] > 0,
            ]
            training_set.append(training_pair)

        # do train val split
        trainset, valset = train_test_split(training_set, test_size = 0.2, random_state = 42)

        return trainset, valset


