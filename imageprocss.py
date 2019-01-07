import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import math
from sklearn.externals import joblib
from skimage.feature import hog
random_seed=np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import cv2
from keras.models import load_model

# loading pre trained model
model = load_model('cnnmodel.h5')
def a(path):
 # Read the input image 
 im = cv2.imread(path)
 cv2.imshow("1",im)
 # Convert to grayscale and apply Gaussian filtering
 im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
 im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
 cv2.imshow("2",im_gray)
 # Threshold the image
 ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
 cv2.imshow("3",im_th)
 # Find contours in the image
 _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 # Get rectangles contains each contour
 rects = [cv2.boundingRect(ctr) for ctr in ctrs]

 # For each rectangular region, calculate HOG features and predict

 for rect in rects:
     # Draw the rectangles
     cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
     # Make the rectangular region around the digit
     leng = int(rect[3] * 1.6)
     pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
     pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
     roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
     # Resize the image
     roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
     roi = cv2.dilate(roi, (3, 3))
     roi=roi.reshape(-1,28,28,1)
    
     nbr = np.argmax(model.predict(roi))
     cv2.putText(im, str(int(nbr)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

 cv2.imshow("Resulting Image with Rectangular ROIs", im)
 cv2.waitKey()
 return 0
s="C:/Users/ajju/Desktop/ml/ml projects/2digit recogniser/final/exp5.jpg"
a(s)
