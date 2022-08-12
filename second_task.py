import cv2
from cv2 import pointPolygonTest
from cv2 import COVAR_NORMAL
import numpy as np
from matplotlib import pyplot as plt


image=cv2.imread("./asset/second_task/C0_000004.png",cv2.IMREAD_GRAYSCALE)
img = cv2.imread("./asset/second_task/C1_000004.png",)
final_img=img.copy()

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
blank=np.zeros(image.shape,np.uint8)
cv2.drawContours(blank,contours,1,(255,255,255),-1)

plt.imshow(blank,cmap='gray',vmin=0, vmax=255)
plt.show()

kernel_opcl=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
mask=cv2.morphologyEx(blank,cv2.MORPH_OPEN,kernel_opcl)

plt.imshow(mask, cmap='gray',vmin=0, vmax=255)
plt.show()

masked=cv2.bitwise_and(img,img,mask=mask)
#cv2.imwrite("C:/Users/danie/Desktop/cvip prog/Images/maskededd.png",masked)
#plt.imshow(masked,vmin=0, vmax=255)
#plt.show()
cv2.imshow("finestra", masked)
#cv2.waitKey(0)


red=0
blue=0
green=0
counter=0

for i in range(masked.shape[0]):
    for j in range(masked.shape[1]):
        if(masked[i,j].all()!=np.zeros(3).all()):
            counter=counter+1
            blue=blue+masked[i,j][0]
            green=green+masked[i,j][1]
            red=red+masked[i,j][2]
            

mean_blue=blue/counter
mean_green=green/counter
mean_red=red/counter



print(mean_blue,mean_green,mean_red)
mymean=np.array([mean_blue,mean_green,mean_red])

b,g,r=cv2.split(masked)
b_flat=b.flatten()
g_flat=g.flatten()
r_flat=r.flatten()

flattened=np.array([b_flat,g_flat,r_flat])
print(flattened.shape)
covar,mean_out=cv2.calcCovarMatrix(flattened, mean=mymean, flags=cv2.COVAR_ROWS)
print(covar)
print(masked.shape)
cv2.imshow("finestra", masked)
cv2.waitKey(0)
