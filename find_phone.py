'''
  Test phone finder on given dataset, calculate the accuracy
'''

import os
import cv2
from tqdm import tqdm

from phone_finder import PhoneFinder

if __name__ == '__main__':

  try:
      test_img_path = sys.argv[1]
  except Exception as e:
      print(sys.argv)
      print(e)

  phone_finder = PhoneFinder()
  phone_finder.test_phone_finder(params, test_img_path)
