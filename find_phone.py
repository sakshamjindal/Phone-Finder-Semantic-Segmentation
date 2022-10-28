'''
  Test phone finder on given image and ouput the normalised x and y coordinates
'''

import sys
from phone_finder import PhoneFinder

if __name__ == '__main__':

  try:
      test_img_path = sys.argv[1]
  except Exception as e:
      print(sys.argv)
      print(e)

  phone_finder = PhoneFinder()
  phone_finder.test_phone_finder(params, test_img_path)
