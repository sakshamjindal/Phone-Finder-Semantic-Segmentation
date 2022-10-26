'''
    Train phone finder using given dataset
'''
import os
import cv2
import sys
import numpy as np

from phone_finder import PhoneFinder

if __name__ == '__main__':
    try:
        folder = sys.argv[1]
    except Exception as e:
        print(sys.argv)
        print(e)

    phone_finder = PhoneFinder()
    X = []
    Y = []
    
    f_labels = open(folder + "/labels.txt")
    for line in f_labels.readlines():
        s = line.split()
        file = s[0]
        img = cv2.imread(os.path.join(folder, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(-2, 3):
            x = int(float(s[2])*img.shape[0]) + i
            for j in range(-2, 3):
                y = int(float(s[1])*img.shape[1]) + j
                X.append(img[x,y].astype(np.float64)/255)
                Y.append(1)
        #random_X = np.random.random_sample((75*121,))
        #random_Y = np.random.random_sample((75*121,))
        #for i in range(75*121):
        #    x = int(float(random_X[i])*img.shape[0])
        #    y = int(float(random_Y[i])*img.shape[1])
        #    X.append(img[x, y].astype(np.float64)/255)
        #    Y.append(0)
    X = np.array(X)
    Y = np.array(Y)
    
    phone_finder.train_segment(X, Y)