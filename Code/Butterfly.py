import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv('Dataset/Training_set Copy.csv')
df_final = []
i=0
while(i<499):


    img_bgr = cv2.imread('Dataset/train/'+df_train.iloc[i]['filename'])
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray'+str(i) , img_gray)
    #print(img_gray)

    binary = np.where(img_gray > np.mean(img_gray), 1.0, 0.0)

    #cv2.imshow('binary'+str(i) , binary)
    #print(binary)
    #print(df_train.head())

    df_final.append(binary)





    i = i+1
    print(i)


print("done")

le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['label'])




y = df_train['filename']

x = df_train['label']


#print(df_final)

df_final = pd.DataFrame()
#df_final['filename'] = binary
#df_final['label'] = x
df_final.to_csv('submission.csv')



cv2.waitKey(0)
cv2.destroyAllWindows()