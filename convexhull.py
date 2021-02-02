from cv2 import cv2
import numpy as np

#input image
src=cv2.imread('frame.jpg',1)
src=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
src1=cv2.imread('background.jpg',0)
# src1=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
sub=cv2.subtract(src1,src)
sub=sub*3
# cv2.imshow('sub',sub)
#binarize the input 
ret,thresh=cv2.threshold(sub,0,255,cv2.THRESH_OTSU)
# cv2.imshow('thresh',thresh)
kernel=np.ones((5,5),np.uint8)
dilate=cv2.dilate(thresh,kernel,iterations=1)
# cv2.imshow('dilated',dilate)
#finding the contours on the thresholded image
imgcontour=dilate.copy()
contours,hierarchy=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgcontour,contours,-1,(0,255,0),3)
# cv2.imshow('contours',imgcontour)


#create hull of convex points
hull=[]
#calculate points for each contour
for i in range (len(contours)):
    hull.append(cv2.convexHull(contours[i],False))

imghull=imgcontour.copy()
imghull=cv2.cvtColor(imghull,cv2.COLOR_GRAY2BGR)
for i in range(len(contours)):
    cv2.drawContours(imghull,contours,i,(0,255,0),3)
    cv2.drawContours(imghull,hull,i,(255,0,0),3)


cv2.imshow('hull',imghull)
cv2.waitKey(0)