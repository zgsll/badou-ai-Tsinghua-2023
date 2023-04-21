# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:16:20 2023

@author: YYX
"""

#nearest interp
import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.unint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            emptyImage[i,j]=img[x,y]
            return emptyImage
        
img=cv2.imread("G:/test/lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv.imshow("nearest interp",zoom)
cv.imshow("image",img)
cv2.waitKey(0)

#bilinear interpolation
import mumpy as np
import cv2

def bilinear_interpolation(img,out_dim):
    src_h,src_w,channel+img.shape
    dst_h,dst_w=out_dim[1],out_dim[0]
    print("src_h,src_w=",src_h,src_w)
    print("dst_h,dst_w=",dst_h,dst_w)
    if src_h==dst_h and src_w==dst_w:
        return img.copy()
    dst_img=np.zeros((dst_h,dst_w,3),dtype=np.unit8)
    scale_x,scale_y=float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x=(dst_x+0.5)*scale_x-0.5
                src_y=(dst_y+0.5)*scale_y-0.5
                src_x0=int(np.floor(src_x))
                src_x1=min(src_x0+1,src_w-1)
                src_y0=int(np.floor(src_y))
                src_y1=min(src_y0+1,src_h-1)
                temp0=(src_x1-src_x)*img[src_y0,src_x0,i]+(src_x-src_x0)*img[src_y0,src_x1,i]
                temp1=(src_x1-src_x)*img[src_y1,src_x0,i]+(src_x-src_x0)*img[src_y1,src-x1,i]
                dst_img[dst_y,dst_x,i]=int((src_y1-src_y)*temp+(src_y-src_y0)*temp1)
                
    return dst_img
    
if __name__=='__main__':
    img=cv2.imread('lenna.png')
    dst=bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()
    
#histogram equalization
import cv2
import numpy as np
from matlotlib import pyplot as plt

img=cv2.imread("lenna.png",1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst=cv2.equalizehist(gray)
hist=cv2.calchist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()
cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv.waitKey(0)

img=cv2.imread("lenna.png",1)
cv2.imshow("src",img)
(b,g,r)=cv2.split(img)
bH=cv2.equalizeHist(b)
gH=cv2.equalizeHist(g)
rH=cv2.equalizeHist(r)
result=cv2.merge((bH,gH,rH))
cv2.imshow("dst_rgb",result)

cv2.waitKey(0)