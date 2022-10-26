'''
  Test phone finder on given dataset, calculate the accuracy
'''

import os
import cv2
from tqdm import tqdm

from phone_finder import PhoneFinder


if __name__ == '__main__':
    phone_finder = PhoneFinder()
    folder = "find_phone/"
    X = []
    Y = []
    f_labels = open(folder + "labels.txt")
    correct_cnt = 0
    total_cnt = 0
    for line in tqdm(f_labels.readlines()):
        total_cnt += 1
        s = line.split()
        file = s[0]
        gt_x = float(s[1])
        gt_y = float(s[2])
        org = cv2.imread(os.path.join(folder, file))
        img = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)
        result = phone_finder.get_bounding_boxes(img)
        print(total_cnt)
        if not result:
          continue
        dis = (gt_x - result[0])**2 + (gt_y - result[1])**2
        if(dis <= 0.0025):
          correct_cnt += 1
    print("detection accuracy: ", correct_cnt*1.0/total_cnt)