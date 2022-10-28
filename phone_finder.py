'''
  Train phone finder on a generated synthetic datset
'''
import os
import math
import torch
import glob
from tqdm import tqdm

from core.dataset import IphoneDataset, train_augmentations, test_augmentations
from core.generator import SyntheticDataGenerator
from core.model import create_model
from core.datamodule import DataModule
from core.segmenter import SemanticSegmenter
from core.utils import load_labels
from core.postprocessing import get_tight_polygon_from_mask


class PhoneFinder():
	
	def __init__(self):
		pass

	def train_phone_finder(self, params):

		## generate the synthetic dataset
		data_generator = SyntheticDataGenerator()
		trainset, valset = data_generator.generate_synthetic_dataset()

		## load the datasets
		train_dataset = IphoneDataset(trainset, transforms=train_augmentations(params["image_size"]))
		val_dataset = IphoneDataset(valset, transforms=test_augmentations(params["image_size"]))

		## initialise the model
		model = create_model(output_channels=params["num_classes"])

		## initialise the datamodule
		datamodule = DataModule(
						train_dataset,
						val_dataset,
						batch_size=params["batch_size"],
						num_workers=params["num_workers"]
					)
		## initialise the segmenter module
		segmenter = SemanticSegmenter(model, datamodule, params)


		## start the training
		segmenter.train_and_test(params["num_epochs"])


	def find_phone(self, params, test_img_path):

		## initialise the model
		model = create_model(output_channels=params["num_classes"])

		## initialise the segmenter module
		segmenter = SemanticSegmenter(model, None, params, mode="inference", model_path=params["model_path"])

		## do prediction from the deep learning model here
		test_img_tensor, pred_tensor = segmenter.predict_single_image(test_img_path)

		## check if mask is detected
		## if yes - find a tight convex hull around the mask
		if torch.sum(pred_tensor).item() == 0:
			phone_found = False
			x_norm = 0.0
			y_norm = 0.0
		else:
			phone_found = True
			polygon_points = get_tight_polygon_from_mask(pred_tensor)
			phone_location = torch.mean(polygon_points.float(), axis=0)
			x_norm, y_norm = (phone_location[1]/params["image_size"][1]).item(), (phone_location[0]/params["image_size"][0]).item()
			x_norm, y_norm = round(x_norm, 4), round(y_norm, 4)
	
		print(x_norm, y_norm)

	def test_phone_finder(self, params, test_image_root):

		## get test image path
		image_paths = glob.glob(os.path.join(test_image_root, "*.jpg"))

		## get labels path
		labels_path = os.path.join(test_image_root, "labels.txt")
		label_data = load_labels(labels_path)

		## initialise the model
		model = create_model(output_channels=params["num_classes"])

		## initialise the segmenter module
		segmenter = SemanticSegmenter(model, None, params, mode="inference", model_path=params["model_path"])

		phone_detected = 0
		phone_detected_correctly = 0

		# Iterate over the set of Images
		for i in tqdm(range(len(image_paths))):

			test_image_path = image_paths[i]
			# print("Image Path : {}".format(os.path.basename(test_image_path)))

			# Load the ground truth of the phone location
			x_gt, y_gt = label_data[os.path.basename(test_image_path)]

			# do prediction from the deep learning model here
			test_img_tensor, pred_tensor = segmenter.predict_single_image(test_image_path)

			# extract out convex hull polygon points using the Jarvis match algorithm
			## Do only if atleast one pixel of prediction detected (can be improved)
			if torch.sum(pred_tensor).item()==0:
				phone_found = False
			else:
				phone_found = True
				phone_detected += 1
				polygon_points = get_tight_polygon_from_mask(pred_tensor)

			# phone_location is centroid of convex hull
			phone_location = torch.mean(polygon_points.float(), axis=0)
			x_norm, y_norm = (phone_location[1]/params["image_size"][1]).item(), (phone_location[0]/params["image_size"][0]).item()

			# if found is found, find the normalised distance between
			# ground truth and predicted locations
			if phone_found:
				norm_distance = math.sqrt((x_norm - x_gt)**2 + (y_norm - y_gt)**2)
				if norm_distance < 0.05:
					phone_found_correctly = True
					phone_detected_correctly += 1
				else:
					phone_found_correctly = False
			else:
				phone_found_correctly = False

		percentage_correct = (phone_detected_correctly/len(image_paths))*100
		print("Number of Images provided by Brain Corp: {}".format(len(image_paths)))
		print("Number of Images in which phone detected correctly: {}".format(phone_detected_correctly))
		print("Phone found correctly in {} % of provided labeled dataset".format(round(percentage_correct, 4)))