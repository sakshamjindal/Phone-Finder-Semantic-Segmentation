'''
    Phone finder application modules to train and infer a phone finder model
'''

import random

import os
import glob
from PIL import Image
import scipy
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union

from core.dataset import IphoneDataset, train_augmentations, test_augmentations
from core.generator import SyntheticDataGenerator
from core.model import create_model
from core.datamodule import DataModule
from segmenter import SemanticSegmenter

from core.postprocessing import 

class PhoneFinder():
	
	def __init__(self):
		pass

	def train_phone_finder(self, params):

		# generate the synthetic dataset
		data_generator = SyntheticDataGenerator()
		trainset, valset = data_generator.generate_synthetic_dataset()


		## load the datasets
		train_dataset = IphoneDataset(trainset, transforms = train_augmentations(params["image_size"]))
		val_dataset = IphoneDataset(valset, transforms = test_augmentations(params["image_size"]))

		## initialise the model
		model = create_model(output_channels = params["num_classes"])

		## initialise the datamodule
		datamodule = DataModule(train_dataset,
								val_dataset,
								batch_size = params["batch_size"],
								num_workers = params["num_workers"]
								)
		## initialise the segmenter module
		segmenter = SemanticSegmenter(model, datamodule, params)


		## start the training
		# params["num_epochs"]
		segmenter.train_and_test(params["num_epochs"])


	def test_phone_finder(self, params, test_img_path):

		## initialise the model
		model = create_model(output_channels = params["num_classes"])

		## initialise the segmenter module
		segmenter = SemanticSegmenter(model, None, params, mode = "inference", model_path = params["model_path"])

		## do prediction from the deep learning model here
		test_img_tensor, pred_tensor = segmenter.predict_single_image(test_img_path)

		## check if mask is detected
		## if yes - find a tight convex hull around the mask
		if torch.sum(pred_tensor).item()==0:
			phone_found = False
			x_norm = 0.0
			y_norm = 0.0
		else:
			phone_found = True
			polygon_points = get_tight_polygon_from_mask(pred_tensor)
			x_norm, y_norm  = (phone_location[1]/params["image_size"][1]).item(), (phone_location[0]/params["image_size"][0]).item()
	
		print(x_norm, y_norm)
		



		
		