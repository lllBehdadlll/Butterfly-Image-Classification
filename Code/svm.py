import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import random

DataSetPath = 'Dataset/train/'
NumOfImg=3000
df_train = pd.read_csv('Dataset/Training_set Copy.csv')
IMG_SIZE=45
Img_Data = []
Label = []
Ext_DATA = []
Ext_LABEL = []
i=0
en = 0

le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['label'])
Lb = df_train['label']
Label = Lb[0:NumOfImg]

print(Label)

def flip(img):
    img_flip = cv2.flip(img,1)
    return img_flip


def noise(image,prob):
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


def img_preprocess(path,length,size,gray = False):
    global i

    while(i<length):

        img_bgr = cv2.imread(path + df_train.iloc[i]['filename'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if gray == True:
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            new_img = cv2.resize(img_gray,(size,size))
        else:
            new_img = cv2.resize(img_rgb,(size,size))
        
        Img_Data.append(new_img)

        i = i+1


    print("done")

img_preprocess(DataSetPath,NumOfImg,IMG_SIZE,False)


def extra_data(path,length,prop,lab,size,gray = False):
    ii=0
    global en
    
    while(ii<length):
        P = np.random.choice([0, 1], p=[1-prop, prop])
        if P == 1:
            img_bgr = cv2.imread(path + df_train.iloc[ii]['filename'])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            if gray == True:
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                new_img = cv2.resize(img_gray,(size,size))
            else:
                new_img = cv2.resize(img_rgb,(size,size))

            PFlipNoise = np.random.choice([0, 1, 2], p=[0.6, 0.2,0.2])
            
            
            if PFlipNoise==0:
                Ext_DATA.append(flip(new_img))
                Ext_LABEL.append(lab[ii])
                en = en+1
            elif PFlipNoise==1:
                Ext_DATA.append(noise(new_img,0.2))
                Ext_LABEL.append(lab[ii])
                en = en+1
            elif PFlipNoise==2:
                T = flip(new_img)
                Ext_DATA.append(noise(T,0.2))
                Ext_LABEL.append(lab[ii])
                en = en+1
            
            
            
            
            
            
            
        ii = ii+1
    print("done")



extra_data(DataSetPath,NumOfImg,.9,Label,IMG_SIZE,False)





X=[]
y=[]


Img_Data_Data = np.concatenate((Img_Data, Ext_DATA))





X = Img_Data_Data
y = Label
y= np.concatenate((y, Ext_LABEL))
X = np.array(X).reshape(en + NumOfImg,-1)

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