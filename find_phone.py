'''
    Detect and Print the normalized coordinates of the phone detected on this test image
'''
import sys
import cv2

from phone_finder import PhoneFinder


if __name__ == '__main__':
    try:
        img_path = sys.argv[1]
    except Exception as e:
        print(sys.argv)
        print(e)

    phone_finder = PhoneFinder()

    org = cv2.imread(img_path)
    img = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)
    
    result = phone_finder.get_bounding_boxes(img)

    print(result[0], result[1])