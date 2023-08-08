#کد تا حد متوسط کامنت گذاری شده است توضیحات دقیقتر و تکمیل تر در فایل گزارش وجود دارد
import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import random

DataSetPath = 'Dataset/train/'                           #آدرس تصاویر
df_train = pd.read_csv('Dataset/Training_set.csv')       #خواندن اطلاعات train
NumOfImg=6499     #تعداد تصاویر که در اموزش شرکت میکنند
IMG_SIZE=40      #سایز تصاویر شرکت کننده
Img_Data_Train = []
Img_Data_Test = []
Label = []
Label_Train =[]
Label_Test = []
Ext_DATA_Train = []
Ext_LABEL = []
i=0
NumOfextImg = 0
NumOftrainImg = 0
NumOfXTest = 0



#قطعه کد تبدیل داده های رشته ای لیبل ها به عدد
le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['label'])
Lb = df_train['label']
Label = Lb[0:NumOfImg]


#قطعه کد فلیب کردن تصاویر برای ساخت و اضافه کردن تصاویر جدید از تصاویر موجود
def flip(img):
    img_flip = cv2.flip(img,1)
    return img_flip

#قطعه کد اضافه کردن نویز به تصاویر
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


#قطعه کد خواندن تصویر اصلاح رنگ و سایز
def img_preprocess(img,size,gray = False):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    if gray == True:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        new_img = cv2.resize(img_gray,(size,size))
    else:
        new_img = cv2.resize(img_rgb,(size,size))
    
    return new_img




#قطعه کد ساخت تصاویر جدید از تصاویر موجود برای افزایش مقدار دیتا های قابل تحلیل
def extra_data(img,prop,size,gray = False):
 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    if gray == True:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        new_img = cv2.resize(img_gray,(size,size))
    else:
        new_img = cv2.resize(img_rgb,(size,size))

    PFlipNoise = np.random.choice([0, 1, 2], p=[0.7, 0.2,0.1])   #انتخاب عمل که بروی تصویر اعمال میشود با توجه به احتمالی که تعیین شده است
            
    if PFlipNoise==0:       #تنها فلیب شدن عکس
        return flip(new_img)
    elif PFlipNoise==1:     #تنها اضافه شدن نویز به تصویر
        return noise(new_img,0.2)

    elif PFlipNoise==2:     #هم فلیب و هم اضافه شدن نویز به تصویر
        T = flip(new_img)
        return noise(T,0.2)
              
            
        

def img_process(path,length,propTT,propED):
    global i
    global NumOfextImg
    global NumOftrainImg
    global NumOfXTest
    while(i<length):
        img_bgr = cv2.imread(path + df_train.iloc[i]['filename'])

        Prop_train_test = np.random.choice([0, 1], p=[1-propTT, propTT]) #احتمال حضور عکس در ترین یا در تست بودن
        if Prop_train_test==1:
            Img_Data_Train.append(img_preprocess(img_bgr,IMG_SIZE,False))
            Label_Train.append(Label[i])
            NumOftrainImg = NumOftrainImg+1
            Prop_extraData = np.random.choice([0, 1], p=[1-propED, propED]) #احتمال ساخته شدن عکس جدید
            if Prop_extraData==1:
                Ext_DATA_Train.append(extra_data(img_bgr,.9,IMG_SIZE,False))
                Ext_LABEL.append(Label[i])
                NumOfextImg = NumOfextImg+1
        else:
            Img_Data_Test.append(img_preprocess(img_bgr,IMG_SIZE,False))
            Label_Test.append(Label[i])
            NumOfXTest = NumOfXTest+1
        i =i+1
    
    print("Size of image:",IMG_SIZE )
    print("Orginal image in train:",NumOftrainImg )  
    print("Extra image in train:",NumOfextImg ) 
    print("Orginal image split for test:",NumOfXTest )  
    print("img_process done")
                

            

#شروع پروسس برروی داده ها
img_process(DataSetPath,NumOfImg,0.9,0.9)



X=[]
y=[]

Img_Data_Data = np.concatenate((Img_Data_Train, Ext_DATA_Train)) #ادغام داده های تصاویر اصلی و تصاویر ساخته شده توسط ما
X = Img_Data_Data
y = Label_Train
y= np.concatenate((y, Ext_LABEL))   #ادغام لیبل تصاویر اصلی و تصاویر ساخته شده توسط ما
X = np.array(X).reshape(NumOfextImg + NumOftrainImg,-1)   #دو بعدی کردن آرایه های دریافتی برای اینکه مدل اس وی ام تنها با داده های دو بعدی کار میکند
#print(X.shape)
#X = np.where(X > np.mean(X), 1.0, 0.0)
X = X/255.0     #تقسیم مقادیر آرایه ها بر 255 تا بازه اعداد در بین 0 تا 1 باشد
y = np.array(y)
#print(y.shape)

X_test = Img_Data_Test  #تصاویر تست
X_test = np.array(X_test).reshape(NumOfXTest,-1)
X_test = X_test/255
y_test = Label_Test #لیبل های تست
y_test = np.array(y_test)


print("Train data is ready!")




#کتابخانه های مورد نیاز برای کار با مدل ها
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)  #تقسیم داده با نسبت مشخص شده برای ترین و تست

svc = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
 tol=0.001, cache_size=200, class_weight='balanced', verbose=False,
 max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=None)   #فراخانی مدل و مشخص کردن پارامتر های آن

print("Training... !")
svc.fit(X, y)       #اعمال مدل بر داده ها

y2 = svc.predict(X_test)        #پیشبینی کردن توسط مدل بر روی داده های تست

print("Accuracy: ",accuracy_score(y_test,y2))  #تعیین میزان دقت


result = pd.DataFrame({'original' : y_test,'predicted' : y2})   #نمایش پیشبینی مدل و مقدار واقعی
print(result)
