import cv2
from cv2 import pointPolygonTest
import numpy as np
from matplotlib import pyplot as plt


image=cv2.imread("./asset/final_challenge/C0_000006.png",cv2.IMREAD_GRAYSCALE)
img = cv2.imread("./asset/final_challenge/C1_000006.png")
final_img=img.copy()

ret,t_image=cv2.threshold(image,40,255,cv2.THRESH_BINARY)
#plt.imshow(t_image,cmap='gray',vmin=0, vmax=255)
#plt.show()

contours,hier=cv2.findContours(t_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
blank=np.zeros(image.shape,np.uint8)
contours=list(contours)
contours.sort(key=cv2.contourArea,reverse=True)
cv2.drawContours(blank,contours,0,(255,255,255),-1)

#plt.imshow(blank,cmap='gray',vmin=0, vmax=255)
#plt.show()
kernel_opcl=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
mask=cv2.morphologyEx(blank,cv2.MORPH_OPEN,kernel_opcl)

#plt.imshow(mask, cmap='gray',vmin=0, vmax=255)
#plt.show()

masked=cv2.bitwise_and(img,img,mask=mask)

masked = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
#plt.imshow(masked)
#plt.show()

# lower mask (0-10)
lower_kiwi = np.array([4,0,0])
upper_kiwi = np.array([15,255,255])
masked = cv2.inRange(masked, lower_kiwi, upper_kiwi)
cv2.imshow('masked', masked)
cv2.waitKey(0)

contours,hier=cv2.findContours(masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
blank=np.zeros(image.shape,np.uint8)
contours=list(contours)
contours.sort(key=cv2.contourArea,reverse=True)
cv2.drawContours(blank,contours,0,(255,255,255),-1)

cv2.imshow('blank', blank)
cv2.waitKey(0)

masked=cv2.bitwise_and(image,image,mask=blank)

img_blur = cv2.bilateralFilter(masked, 20, 40, 40)


# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
cl = clahe.apply(img_blur)
# Stacking the original image with the enhanced image
result = np.hstack((img_blur, cl))
cv2.imshow('Result', result)
cv2.waitKey(0)
img_blur=cl

plt.imshow(img_blur)
plt.show()

lower_kiwi = 70
upper_kiwi = 255
masked = cv2.inRange(img_blur, lower_kiwi, upper_kiwi)
cv2.imshow('Result', masked)
cv2.waitKey(0)


kernel_dil=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dilated=cv2.dilate(masked,kernel_dil,iterations=1)

cv2.imshow('Dilated Canny Edge Detection', dilated)
cv2.waitKey(0)

contours_edge,hier_edge=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_edge=list(contours_edge)
contours_edge.sort(key=cv2.contourArea,reverse=True)
cv2.drawContours(img,contours_edge,-1,(0,0,255),1)

cv2.imshow('Canny Edge Detection', img)
cv2.waitKey(0)


# Contour filtering based on color inside contour
filtered_contours = []
for idx, contour in enumerate(contours_edge):
    blank = np.zeros(image.shape, dtype='uint8')
    cv2.drawContours(blank,contours_edge,idx,(255,255,255),-1)

    locs = np.where(blank == 255)
    pixels = dilated[locs]

    if (np.mean(pixels) < 240 and (cv2.contourArea(contour)>=126.0) and (cv2.contourArea(contour)<50000.0)):
        filtered_contours.append(contour)


inside=np.zeros(len(filtered_contours), np.uint8)
for i, c in enumerate(filtered_contours):
    for j, k in enumerate(filtered_contours):
        if (i!=j):
            if(pointPolygonTest(k,(int(c[0][0][0]),int(c[0][0][1])), False)>=0):
                inside[i]=1

defects=[]
for i,v in enumerate(inside):
    if(v==0):
        defects.append(filtered_contours[i])



cv2.drawContours(final_img,defects,-1,(0,0,255),2)