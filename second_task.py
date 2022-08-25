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

class Image_num(Enum):
    IMG4=0
    IMG5=1

IMAGE=Image_num.IMG4

if(IMAGE==Image_num.IMG4):
    imgpath_ir="./asset/second_task/C0_000004.png"
    imgpath_col="./asset/second_task/C1_000004.png"
    X_MIN = 124
    X_MAX = 135
    Y_MIN = 94
    Y_MAX = 105

elif(IMAGE==Image_num.IMG5):
    imgpath_ir="./asset/second_task/C0_000005.png"
    imgpath_col="./asset/second_task/C1_000005.png"
    X_MIN = 139
    X_MAX = 144
    Y_MIN = 152
    Y_MAX = 157


COLOR_SPACE=ColorSpace.RGB
DISTANCE_TYPE=DistanceType.MAHALANOBIS



image=cv2.imread(imgpath_ir,cv2.IMREAD_GRAYSCALE)
img = cv2.imread(imgpath_col)
final_img=img.copy()

t_image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
kernel_opcl=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
opened=cv2.morphologyEx(t_image,cv2.MORPH_OPEN,kernel_opcl)

contours,hier=cv2.findContours(opened,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours=list(contours)
contours.sort(key=cv2.contourArea,reverse=True)
blank=np.zeros(image.shape,np.uint8)
blank2=np.zeros(image.shape,np.uint8)
cv2.drawContours(blank,contours,1,(255,255,255),-1)
cv2.drawContours(blank2,contours,1,(255,255,255),28)
#plt.imshow(blank,cmap='gray',vmin=0, vmax=255)
#plt.show()

#contours,hier=cv2.findContours(blank,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours=list(contours)
#contours.sort(key=cv2.contourArea,reverse=True)
#blank=np.zeros(image.shape,np.uint8)
#cv2.drawContours(blank,contours,1,(255,255,255),-1)


#plt.imshow(blank,cmap='gray',vmin=0, vmax=255)
#plt.show()


kernel_opcl=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
mask=cv2.morphologyEx(blank,cv2.MORPH_OPEN,kernel_opcl)

#plt.imshow(mask, cmap='',vmin=0, vmax=255)
#plt.show()

masked=cv2.bitwise_and(img,img,mask=mask)

#plt.imshow(masked,vmin=0, vmax=255)
#plt.show()

#masked=cv2.medianBlur(masked, 3)
#plt.imshow(masked,cmap='gray',vmin=0, vmax=255)
#plt.show()

#masked=cv2.bilateralFilter(masked, 5, 10, 10)
#plt.imshow(masked,cmap='gray',vmin=0, vmax=255)
#plt.show()

# converting to LAB color space
lab= cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)
# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(9,9))
cl = clahe.apply(l_channel)
# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))
# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# Stacking the original image with the enhanced image
result = np.hstack((masked, enhanced_img))
#cv2.imshow('Result', result)
#cv2.waitKey(0)
masked=enhanced_img.copy()


#masked=cv2.bilateralFilter(masked, 20, 20, 20)
#plt.imshow(masked,cmap='gray',vmin=0, vmax=255)
#plt.show()


ch1=0
ch2=0
ch3=0
counter=0
for i in range(Y_MIN, Y_MAX):
    for j in range(X_MIN, X_MAX):
        #if(masked[i,j].all()!=np.zeros(3).all()):
            counter=counter+1
            ch1=ch1+masked[i,j][0]
            ch2=ch2+masked[i,j][1]
            ch3=ch3+masked[i,j][2]
            final_img[i,j] = [255,255,255]
mean_ch1=ch1/counter
mean_ch2=ch2/counter
mean_ch3=ch3/counter     
mymean=np.array([mean_ch1,mean_ch2,mean_ch3])
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
#plt.imshow(final_img,vmin=0, vmax=255)
#plt.show()

#with open('my_mean_05.csv', 'w') as my_file:
    #for i in range(len(mymean)):
        #np.savetxt(my_file, mymean)
#print('Array exported to file')
#mymean = np.loadtxt('my_mean.csv')
#print(mymean)



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
    cropped_selection = masked[Y_MIN:Y_MAX,X_MIN:X_MAX]

    cropped_selection=cv2.medianBlur(cropped_selection, 9)


    fc,sc,tc=cv2.split(cropped_selection)
    fc_flat=fc.flatten()
    sc_flat=sc.flatten()
    tc_flat=tc.flatten()

    flattened=np.array([fc_flat,sc_flat,tc_flat])
    print(flattened.shape)
    covar,mean_out=cv2.calcCovarMatrix(flattened, mean=mymean, flags=cv2.COVAR_ROWS)
    print(covar)
    print(np.linalg.det(covar))
    #cv2.imshow("finestra", masked)
    #cv2.waitKey(0)

    # INVERT COVAR MATRIX
    #covar2=np.diag(np.diag(covar))
    u,s,v=np.linalg.svd(covar, full_matrices=True)
    inv_cov=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
    #print(inv_cov)
    #inv_cov=np.linalg.pinv(covar2)
    pos=lambda x: abs(x)
    #inv_cov=pos(inv_cov)
    #with open('my_inv_cov_04.csv', 'w') as my_file:
        #for i in inv_cov:
            #np.savetxt(my_file, i)
    #print('Array exported to file')

    #inv_cov = np.loadtxt('my_inv_cov.csv')
    #inv_cov = inv_cov.reshape(3,3)
    print(inv_cov)

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
            dist[i,j] = cv2.norm(masked[i][j] - mymean, cv2.NORM_L2)


max_dist=np.max(dist)
min_dist=np.min(dist)
#plt.imshow(dist, cmap='twilight',vmin=min_dist, vmax=max_dist)
#plt.show()

OldRange = (max_dist - min_dist)

map_fcn=lambda x:np.uint8(((x - min_dist) * 255) / OldRange)
mapped_dist=map_fcn(dist)
mapped_dist=cv2.applyColorMap(mapped_dist,cv2.COLORMAP_TWILIGHT)
cv2.imshow("We fratm", mapped_dist)
cv2.waitKey(0)

# From here inrange function to capture the russet

# inRange function works only on HSV images
mapped_dist = cv2.cvtColor(mapped_dist, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(mapped_dist, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([140,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(mapped_dist, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = mapped_dist.copy()
output_img[np.where(mask==0)] = 0
cv2.imshow("Ranged",output_img)
cv2.waitKey(0)
# or your HSV image, which I *believe* is what you want
#output_hsv = mapped_dist.copy()
#output_hsv[np.where(mask==0)] = 0

output_img[np.where(blank2==255)]=0
cv2.imshow("Masked",output_img)
cv2.waitKey(0)

output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

retval,output_img=cv2.threshold(output_img,1,255,cv2.THRESH_BINARY)
cv2.imshow("aaaaaaaaaaaaa",output_img)
cv2.waitKey(0)

kernel_dil=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
output_img=cv2.dilate(output_img,kernel_dil,iterations=1)
cv2.imshow("aaaaaaaaaaaaa",output_img)
cv2.waitKey(0)


kernel = np.ones((2,2),np.uint8)
output_img = cv2.erode(output_img,kernel,iterations = 2)
cv2.imshow("aaaaaaaaaaaaa",output_img)
cv2.waitKey(0)

contours,hier=cv2.findContours(output_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours=list(contours)
contours.sort(key=cv2.contourArea,reverse=True)
blank=np.zeros(image.shape,np.uint8)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
good_contours=[]

for i,c in enumerate(contours):
    if(cv2.contourArea(c)<350):
        continue

    #print(cv2.contourArea(c))
    blank=np.zeros(image.shape,np.uint8)
    cv2.drawContours(blank,contours,i,(255,255,255),-1)
    locs = np.where(blank == 255)
    h,s,v=cv2.split(img_hsv)
    pixels = h[locs]
    mean_h=np.mean(pixels)
    print(mean_h)
    if(mean_h<23):
        good_contours.append(c)

blank=np.zeros(image.shape,np.uint8)
print(len(good_contours))
for i,c in enumerate(good_contours):
    cv2.drawContours(blank,good_contours,i,(255,255,255),-1)


contours,hier=cv2.findContours(blank,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3)
cv2.imshow("final image",img)
cv2.waitKey(0)