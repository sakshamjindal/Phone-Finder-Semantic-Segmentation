'''
	Phone detection system
'''
import numpy as np
import cv2
import joblib
import os.path
from sklearn.kernel_ridge import KernelRidge

class PhoneFinder():
	def __init__(self):
		if os.path.isfile('clf.model'):
			self.clf = joblib.load('clf.model')
		else:
			self.clf = KernelRidge(alpha = 1.0)
	def train_segment(self, X, y):
		'''
			Train a color classifier to find the area of phone
			Inputs:
				X - n*3 numpy array, pixels for training
				y - label of each pixel. 1: phone, 0: other 
		'''
		print("training ...")
		self.clf.fit(X, y)
		joblib.dump(self.clf, 'clf.model')
		print("model saved in clf.model")
		return

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			Inputs:
				img - input image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is black and 0 otherwise
		'''
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		img = np.array(img)
		img = img.astype(np.float64)/255.0
		h,w = img.shape[:2]
		img_vector = img.reshape(w*h, 3)
		mask_vactor = self.clf.predict(img_vector)
		mask_img = mask_vactor.reshape([h, w])
		mask_img = np.array(mask_img*255, dtype = np.uint8)
		mask_img = cv2.GaussianBlur(mask_img,(5,5),0)
		mask_img =  np.where(mask_img < 110, 255, 0)	
		return mask_img.astype(np.uint8)

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the phone
			call other functions in this class if needed
			
			Inputs:
				img - input image
			Outputs:
				center - [x, y], center of the phone in image, [-1, -1] if no phone is found
		'''
		mask_img = self.segment_image(img) 
		contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		centers = []
		for c in contours:
			rect = cv2.minAreaRect(c)
			w = rect[1][0]
			h = rect[1][1]
			if(w*h < 150):
				continue
			if (w*h>1000):
				continue
			if (w/h > 2.5):
				continue
			if (h/w > 2.5):
				continue
			centers.append([rect[0][0]*1.0/img.shape[1], rect[0][1]*1.0/img.shape[0]])
		if not centers:
			return [-1, -1]
		return centers[0]
		
		