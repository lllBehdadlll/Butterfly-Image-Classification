import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
o=500
df_train = pd.read_csv('Dataset/Training_set Copy.csv')
IMG_SIZE=80
T_Data = []
R_DATA = []
i=0


import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output



while(i<o):


    img_bgr = cv2.imread('Dataset/train/'+df_train.iloc[i]['filename'])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    new_img = cv2.resize(img_rgb,(IMG_SIZE,IMG_SIZE))
    T_Data.append(new_img)
    img_flip = cv2.flip(new_img,1)
    #R_DATA.append(sp_noise(img_flip,0.05))
    R_DATA.append(img_flip)
    #cv2.imshow('bgr'+str(i) , img_bgr)
    #cv2.imshow('rgb'+str(i) , img_rgb)
    #print(img_gray)
    i = i+1
    #print(i)


print("done")

le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['label'])
h=df_train['label']
X=[]
y=[]

T_Data = np.concatenate((T_Data, R_DATA))





X = T_Data
y = h[0:o]
y= np.concatenate((y, y))
X = np.array(X).reshape(2*o,-1)

print(X.shape)
#X = np.where(X > np.mean(X), 1.0, 0.0)
X = X/255.0
y = np.array(y)
print(y.shape)

#print(X)
#print(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)


from sklearn.svm import SVC
svc = SVC(kernel='linear',gamma=0.5,probability=True)
svc.fit(X_train, y_train)


y2 = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy on unknown data is",accuracy_score(y_test,y2))


from sklearn.metrics import classification_report
#print("Accuracy on unknown data is",classification_report(y_test,y2))


result = pd.DataFrame({'original' : y_test,'predicted' : y2})

#print(result)

#print(df_final)
#print(y)
#print(binary)
#df_final = pd.DataFrame()
#df_final['filename'] = binary
#df_final['label'] = x
#df_final.to_csv('submission.csv')



cv2.waitKey(0)
cv2.destroyAllWindows()