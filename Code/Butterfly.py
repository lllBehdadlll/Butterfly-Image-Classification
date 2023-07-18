import cv2
import numpy as np
import pandas as pd
import matplotlib as plt


df_train = pd.read_csv('Dataset/Training_set.csv')
img_bgr = cv2.imread('Dataset/train/'+df_train.iloc[0]['filename'])
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow('orginal gray' , img_gray)
print(img_gray)




cv2.waitKey(0)
cv2.destroyAllWindows()