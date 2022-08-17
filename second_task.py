import cv2
from cv2 import pointPolygonTest
from cv2 import COVAR_NORMAL
import numpy as np
from matplotlib import pyplot as plt
from colorthief import ColorThief

imgpath_ir="./asset/second_task/C0_000005.png"
imgpath_col="./asset/second_task/C1_000005.png"

image=cv2.imread(imgpath_ir,cv2.IMREAD_GRAYSCALE)
img = cv2.imread(imgpath_col,)
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
        #if(masked[i,j].all()!=np.zeros(3).all()):
            counter=counter+1
            blue=blue+masked[i,j][0]
            green=green+masked[i,j][1]
            red=red+masked[i,j][2]
            

mean_blue=blue/counter
mean_green=green/counter
mean_red=red/counter


print(mean_blue,mean_green,mean_red)


#mymean=masked[120][110]
#mymean=np.array([mean_blue,mean_green,mean_red])

color_thief = ColorThief(imgpath_col)
# get the dominant color
dominant_color = color_thief.get_color(quality=1)

palette = color_thief.get_palette(color_count=3)
print(palette[0][0])
cv2.waitKey(0)
max=0
max_i=0
for i,c in enumerate(palette):
    sum=palette[i][0]+palette[i][1]+palette[i][2]
    if sum>max:
        max=sum
        max_i=i

mymean=np.array(palette[max_i])


cv2.waitKey(0)

b,g,r=cv2.split(masked)
b_flat=b.flatten()
g_flat=g.flatten()
r_flat=r.flatten()

flattened=np.array([b_flat,g_flat,r_flat])
print(flattened.shape)
covar,mean_out=cv2.calcCovarMatrix(flattened, mean=mymean, flags=cv2.COVAR_ROWS)
cv2.imshow("finestra", masked)
cv2.waitKey(0)



#mymean=[mean_blue,mean_green,mean_red]

#inv_cov=covar.inv(cv2.DECOMP_SVD)

covar2=np.diag(np.diag(covar))

u,s,v=np.linalg.svd(covar)
inv_cov=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
#inv_cov=np.linalg.inv(covar2)
pos=lambda x: abs(x)
inv_cov=pos(inv_cov)

print(inv_cov)
mymean=mymean.astype(np.float32)
masked=masked.astype(np.float32)
inv_cov=inv_cov.astype(np.float32)
dist=np.zeros((masked.shape[0],masked.shape[1]),np.double)
for i, c in enumerate(masked):
    for j, k in enumerate(masked[i]):

        #print(mymean.shape)
        dist[i,j]=cv2.Mahalanobis(masked[i][j],mymean,inv_cov)

max_dist=np.max(dist)
min_dist=np.min(dist)

print(max_dist)
print(min_dist)

OldRange = (max_dist - min_dist)
NewRange = 255
myfcn=lambda x: (((x - min_dist) * NewRange) / OldRange)
colour_dist=myfcn(dist)
#print(colour_dist)
colour_dist=colour_dist.astype(int)

plt.imshow(dist, cmap='Greys',vmin=min_dist, vmax=max_dist)
plt.show()

plt.imshow(dist, cmap='viridis',vmin=min_dist, vmax=max_dist)
plt.show()

plt.imshow(image, cmap='gray',vmin=0, vmax=255)
plt.show()