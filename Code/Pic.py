import cv2
import numpy as np
from matplotlib import pyplot as plt



#نمایش ساده عکس بصورت سیاه و سفید
img_bgr = cv2.imread('Dataset/train/Image_4.jpg')
cv2.imshow('orginal' , img_bgr)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
cv2.imshow('orginal gray' , img_gray)







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

n8 = sp_noise(img_gray,0.05)

cv2.imshow('b8' , n8)
#نمایش عکس بصورت نموداری و کشیدن خط بروی عکس و ذخیره کردن
#plt.imshow(img_bgr, cmap='gray', interpolation='bicubic')
#plt.plot([120,20], [100,50], 'r', linewidth=5)
#plt.show()
#cv2.imwrite('imgo2ut.jpg',img_bgr)

#مشخص کردن لبه های اجسام روش 1
#laplacian = cv2.Laplacian(img_gray, cv2.CV_8U)
#cv2.imshow('laplacian' , laplacian)


#مشخص کردن لبه های اجسام روش 2
#sobelx = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=5)
#sobely = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize=5)
#cv2.imshow('soblex', sobelx)
#cv2.imshow('sobley', sobely)

#مشخص کردن لبه های اجسام روش 3
#canny = cv2.Canny(img_gray, 100, 200)
#cv2.imshow('canny', canny)



#تشخیص الگو در تصویر
#img_template = cv2.imread('Image_7.jpg',0)
#w, h = img_template.shape[::-1]

#res = cv2.matchTemplate(img_gray, img_template, cv2.TM_CCOEFF_NORMED)
#threshhold = 0.3

#loc = np.where(res >= threshhold)

#for pt in zip(*loc[::-1]):
#    cv2.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0,0,255),1)

#cv2.imshow('tmp', img_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()