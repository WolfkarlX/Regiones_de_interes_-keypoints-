# Desde la versión 3.4.2 de OpenCV, el algoritmo SURF ya no se incluye de forma predeterminada debido a restricciones de patentes.
#Actualmente no está disponible libremente :(

import cv2 as cv 
# pip install opencv-contrib-python==3.4.2.16
# Use python 3.6 wtih virtual env named 'pyenv36'
import numpy as np 
import matplotlib.pyplot as plt 
import os 

def SURF(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'car.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
    # bigger - fewer features
    # smaller - more features 
    hessianThreshold = 3000 
    surf = cv.xfeatures2d.SURF_create(hessianThreshold)
    keypoints = surf.detect(imgGray,None)
    imgGray = cv.drawKeypoints(imgGray,keypoints,imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure() 
    plt.imshow(imgGray)
    plt.show() 
"""Actualmente se encuentra patentada =(  """
if __name__ == '__main__': 
    SURF()