'''
    Visualize segmented images for debugging
'''
import os
import cv2

from phone_finder import PhoneFinder

if __name__ == '__main__':
    phone_finder = PhoneFinder()
    folder = "find_phone/"
    f_labels = open(folder + "labels.txt")
    
    for line in f_labels.readlines():
        s = line.split()
        file = s[0]
        org = cv2.imread(os.path.join(folder, file))    
        img = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)
        segment_img = phone_finder.segment_image(img)
        
        cv2.imshow("org", org)
        cv2.waitKey(0)
        cv2.imshow("segment", segment_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()