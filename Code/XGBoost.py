import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
o=500
df_train = pd.read_csv('Dataset/Training_set Copy.csv')
IMG_SIZE=50
T_Data = []
i=0
while(i<o):


    img_bgr = cv2.imread('Dataset/train/'+df_train.iloc[i]['filename'])
    #img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    new_img = cv2.resize(img_bgr,(IMG_SIZE,IMG_SIZE))
    T_Data.append(new_img)
    #T_Data.append(img_bgr)
    #cv2.imshow('gray'+str(i) , img_gray)
    #print(img_gray)

    #binary = np.where(img_gray > np.mean(img_gray), 1.0, 0.0)

    #cv2.imshow('binary'+str(i) , new_img)
    
    

    





    i = i+1
    #print(i)


print("done")

le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['label'])
h=df_train['label']
X=[]
y=[]

X = T_Data
y = h[0:o]
X = np.array(X).reshape(o,-1)

print(X.shape)
#X = np.where(X > np.mean(X), 1.0, 0.0)
X = X/255.0
y = np.array(y)
print(y.shape)

#print(X)
#print(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)




from xgboost import XGBClassifier


xgb = XGBClassifier(tree_method = 'gpu_hist',n_jobs=-1, n_estimators=45,max_depth=3)
xgb.fit(X_train,y_train)
y2 = xgb.predict(X_test)






from sklearn.metrics import accuracy_score
print("Accuracy on unknown data is",accuracy_score(y_test,y2))


from sklearn.metrics import classification_report
#print("Accuracy on unknown data is",classification_report(y_test,y2))


result = pd.DataFrame({'original' : y_test,'predicted' : y2})

print(result)

#print(df_final)
#print(y)
#print(binary)
#df_final = pd.DataFrame()
#df_final['filename'] = binary
#df_final['label'] = x
#df_final.to_csv('submission.csv')



cv2.waitKey(0)
cv2.destroyAllWindows()