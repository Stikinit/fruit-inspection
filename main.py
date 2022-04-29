import cv2
import numpy as np
from matplotlib import pyplot as plt


image=cv2.imread("./asset/first_task/C0_000003.png",cv2.IMREAD_GRAYSCALE)

t_image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
#,t_image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel_opcl=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
open=cv2.morphologyEx(t_image,cv2.MORPH_OPEN,kernel_opcl)
#closed=cv2.morphologyEx(t_image,cv2.MORPH_CLOSE,kernel)
#kernel_dil=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#dilated=cv2.dilate(open,kernel_dil,iterations=1)

contours,hier=cv2.findContours(open,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#open_rgb=cv2.cvtColor(open,cv2.COLOR_BGR2RGB)
contours.sort(key=cv2.contourArea,reverse=True)
cv2.drawContours(open,contours,0,(255,255,255),-1)

plt.imshow(open,cmap='gray',vmin=0, vmax=255)
plt.show()
kernel_opcl=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
mask=cv2.morphologyEx(open,cv2.MORPH_OPEN,kernel_opcl)

plt.imshow(mask, cmap='gray',vmin=0, vmax=255)
plt.show()

masked=cv2.bitwise_and(image,image,mask=mask)
cv2.imwrite("C:/Users/danie/Desktop/cvip prog/Images/maskededd.png",masked)
