from enum import Enum
import cv2
from cv2 import pointPolygonTest
from cv2 import COVAR_NORMAL
import numpy as np
from matplotlib import pyplot as plt
from colorthief import ColorThief

class ColorSpace(Enum):
    RGB=0
    LUV=1
    HSL=2

class DistanceType(Enum):
    EUCLIDIAN=0
    MAHALANOBIS=1
    DOUBLE=2

imgpath_ir="./asset/second_task/C0_000005.png"
imgpath_col="./asset/second_task/C1_000005.png"
COLOR_SPACE=ColorSpace.LUV
DISTANCE_TYPE=DistanceType.EUCLIDIAN

image=cv2.imread(imgpath_ir,cv2.IMREAD_GRAYSCALE)
img = cv2.imread(imgpath_col)
final_img=img.copy()

t_image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
kernel_opcl=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
open=cv2.morphologyEx(t_image,cv2.MORPH_OPEN,kernel_opcl)

contours,hier=cv2.findContours(open,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours=list(contours)
contours.sort(key=cv2.contourArea,reverse=True)
blank=np.zeros(image.shape,np.uint8)
cv2.drawContours(blank,contours,1,(255,255,255),-1)

#plt.imshow(blank,cmap='gray',vmin=0, vmax=255)
#plt.show()

kernel_opcl=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
mask=cv2.morphologyEx(blank,cv2.MORPH_OPEN,kernel_opcl)

#plt.imshow(mask, cmap='gray',vmin=0, vmax=255)
#plt.show()

masked=cv2.bitwise_and(img,img,mask=mask)
#cv2.imshow("finestra", masked)

# masked = cv2.cvtColor(masked, cv2.COLOR_BGR2HLS)
#for i, c in enumerate(masked):
#        for j, k in enumerate(masked[i]):
#            if(k[1]<10):
#                masked[i][j][1]=255
#masked = cv2.cvtColor(masked, cv2.COLOR_HLS2BGR)

#plt.imshow(masked, cmap='gray',vmin=0, vmax=255)
#plt.show()


red=0
blue=0
green=0
counter=0
for i in range(masked.shape[0]):
    for j in range(masked.shape[1]):
        #if(masked[i,j].all()!=np.zeros(3).all()):
            counter=counter+1
            blue=blue+masked[i,j][0]
            green=green+masked[i,j][1]
            red=red+masked[i,j][2]
mean_blue=blue/counter
mean_green=green/counter
mean_red=red/counter     
mymean=np.array([mean_blue,mean_green,mean_red])
color_pixel = np.zeros((1,1,3), dtype="uint8")
color_pixel[0][0][0] = mymean[0]
color_pixel[0][0][1] = mymean[1]
color_pixel[0][0][2] = mymean[2]

if (COLOR_SPACE==ColorSpace.RGB):
    mymean=cv2.cvtColor(color_pixel, cv2.COLOR_BGR2RGB)[0][0]
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
elif (COLOR_SPACE==ColorSpace.LUV):
    mymean=cv2.cvtColor(color_pixel, cv2.COLOR_BGR2Luv)[0][0]
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2Luv)
elif (COLOR_SPACE==ColorSpace.HSL):
    mymean=cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HLS)[0][0]
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2HLS)

color_thief = ColorThief(imgpath_col)

#GET BRIGHTEST COLOR FROM PALETTE
palette = color_thief.get_palette(color_count=3)
cv2.waitKey(0)
max=0
max_i=0
for i,c in enumerate(palette):
    Vmax = np.max(palette[i])
    Vmin = np.min(palette[i])
    sum=Vmax+Vmin
    if sum>max:
        max=sum
        max_i=i

# FIND DOMINANT COLOR
color_pixel = np.zeros((1,1,3), dtype="uint8")
dominant=np.array(palette[max_i])
color_pixel[0][0][0] = dominant[0]
color_pixel[0][0][1] = dominant[1]
color_pixel[0][0][2] = dominant[2]
if (COLOR_SPACE==ColorSpace.RGB):
    dominant=color_pixel[0][0]
elif (COLOR_SPACE==ColorSpace.LUV):
    dominant=cv2.cvtColor(color_pixel, cv2.COLOR_RGB2Luv)[0][0]
elif (COLOR_SPACE==ColorSpace.HSL):
    dominant=cv2.cvtColor(color_pixel, cv2.COLOR_RGB2HLS)[0][0]
    

if (DISTANCE_TYPE==DistanceType.MAHALANOBIS):
    # CREATE COVAR MATRIX
    fc,sc,tc=cv2.split(masked)
    fc_flat=fc.flatten()
    sc_flat=sc.flatten()
    tc_flat=tc.flatten()

    flattened=np.array([fc_flat,sc_flat,tc_flat])
    print(flattened.shape)
    covar,mean_out=cv2.calcCovarMatrix(flattened, mean=mymean, flags=cv2.COVAR_ROWS)
    print(covar)
    cv2.imshow("finestra", masked)
    cv2.waitKey(0)

    # INVERT COVAR MATRIX
    #covar2=np.diag(np.diag(covar))
    u,s,v=np.linalg.svd(covar)
    inv_cov=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
    print(inv_cov)
    #inv_cov=np.linalg.inv(covar2)
    pos=lambda x: abs(x)
    inv_cov=pos(inv_cov)

    # CALC MAHALANOBIS DISTANCE FOR EACH PIXEL
    dominant=dominant.astype(np.float32)
    mymean=mymean.astype(np.float32)
    masked=masked.astype(np.float32)
    inv_cov=inv_cov.astype(np.float32)
    dist=np.zeros((masked.shape[0],masked.shape[1]),np.double)
    for i, c in enumerate(masked):
        for j, k in enumerate(masked[i]):
            dist[i,j]=cv2.Mahalanobis(masked[i][j],mymean,inv_cov)

elif (DISTANCE_TYPE==DistanceType.EUCLIDIAN):
    dominant=dominant.astype(np.float32)
    masked=masked.astype(np.float32)
    dist=np.zeros((masked.shape[0],masked.shape[1]),np.double)
    for i, c in enumerate(masked):
        for j, k in enumerate(masked[i]):
            dist[i,j] = cv2.norm(masked[i][j] - dominant, cv2.NORM_L2)


max_dist=np.max(dist)
min_dist=np.min(dist)
#plt.imshow(dist, cmap='viridis',vmin=min_dist, vmax=max_dist)
#plt.show()

OldRange = (max_dist - min_dist)

map_fcn=lambda x:255-np.uint8(((x - min_dist) * 255) / OldRange)
#NewValue = (((OldValue - min_dist) * 255) / OldRange)
mapped_dist=map_fcn(dist)
#plt.imshow(mapped_dist, cmap='gray',vmin=0, vmax=255)
#plt.show()

#for i, c in enumerate(mapped_dist):
#        for j, k in enumerate(mapped_dist[i]):
#            if(k<10):
#                mapped_dist[i][j]=255
#plt.imshow(mapped_dist, cmap='gray',vmin=0, vmax=255)
#plt.show()

if (DISTANCE_TYPE==DistanceType.DOUBLE):
    mapped_dist=mapped_dist.astype(np.float32)
    dist=np.zeros((mapped_dist.shape[0],mapped_dist.shape[1]),np.double)
    for i, c in enumerate(mapped_dist):
        for j, k in enumerate(mapped_dist[i]):
            dist[i,j] = cv2.norm(mapped_dist[i][j] - 40, cv2.NORM_L2)

    max_dist=np.max(dist)
    min_dist=np.min(dist)
    OldRange = (max_dist - min_dist)
    mapped_dist=map_fcn(dist)

    plt.imshow(mapped_dist, cmap='Greys',vmin=0, vmax=255)
    plt.show()


#mapped_dist=cv2.equalizeHist(mapped_dist)
#plt.imshow(mapped_dist, cmap='gray',vmin=0, vmax=255)
#plt.show()

mapped_dist=cv2.bilateralFilter(mapped_dist, 20, 20, 20)
plt.imshow(mapped_dist,cmap='gray',vmin=0, vmax=255)
plt.show()

kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
mapped_dist = cv2.filter2D(mapped_dist, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
plt.imshow(mapped_dist,cmap='gray',vmin=0, vmax=255)
plt.show() 

mapped_dist=cv2.medianBlur(mapped_dist, 9)
plt.imshow(mapped_dist,cmap='gray',vmin=0, vmax=255)
plt.show()

#mapped_dist=cv2.adaptiveThreshold(mapped_dist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,5)
#plt.imshow(mapped_dist,cmap='gray',vmin=0, vmax=255)
#plt.show()

#kernel_opcl=cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
#mapped_dist=cv2.morphologyEx(mapped_dist,cv2.MORPH_OPEN,kernel_opcl)
#plt.imshow(mapped_dist,cmap='gray',vmin=0, vmax=255)
#plt.show()

edges = cv2.Canny(image=mapped_dist, threshold1=30, threshold2=150)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)



