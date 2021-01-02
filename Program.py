#Import the necessary libraries
import cv2 
import numpy as np
from matplotlib import pyplot as plt

#Read the input image
img=cv2.imread('picture.png')

#Histogrm for the input image
hist,bins=np.histogram(img.flatten(),256,[0,256])
cdf=hist.cumsum()
cdf_normalized=cdf*float(hist.max())/cdf.max()
plt.plot(cdf_normalized,color='b')
plt.hist(img.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc='upper left')
plt.show()

#Histogram Equalization
R,G,B=cv2.split(img)
output1_R=cv2.equalizeHist(R)
output1_G=cv2.equalizeHist(G)
output1_B=cv2.equalizeHist(B)
result=cv2.merge((output1_R,output1_G,output1_B))

#Output Image
result=cv2.resize(result,(960,540))
cv2.imshow('result',result)

#Histogram for the output image
hist,bins=np.histogram(img.flatten(),256,[0,256])
cdf=hist.cumsum()
cdf_normalized=cdf*float(hist.max())/cdf.max()
plt.plot(cdf_normalized,color='b')
plt.hist(img.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc='upper left')
plt.show()

