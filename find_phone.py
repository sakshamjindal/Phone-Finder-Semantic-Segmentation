'''
  Test phone finder on given image and ouput the normalised x and y coordinates
'''

import sys
from phone_finder import PhoneFinder

params = {
    "image_size" : (256, 256),
    "num_classes": 2,
    "device" : "cuda",
    "model_path": "best_model-3.pth"
}

if __name__ == '__main__':

  try:
      test_img_path = sys.argv[1]
  except Exception as e:
      print(sys.argv)
      print(e)

  phone_finder = PhoneFinder()
  phone_finder.test_phone_finder(params, test_img_path)
