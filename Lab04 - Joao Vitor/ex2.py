import cv2
import numpy as np

def applyGaussMask(sizeMask, deviation_standard):
    lambdaValue = (1/ (2 * np.pi * deviation_standard**2))
    gaussMask = np.fromfunction(
        lambda x, y: lambdaValue * 
        np.exp(-((x - (sizeMask-1)/2)**2 + (y - (sizeMask-1)/2)**2) / (2*deviation_standard**2)),
        (sizeMask, sizeMask)
    )
    gaussMask = gaussMask / np.sum(gaussMask)    
    return gaussMask

img = cv2.imread("Lab04 - Joao Vitor/ben2.png", cv2.IMREAD_GRAYSCALE)

imgFilterGaussMask_1_5 = cv2.filter2D(img, -1, applyGaussMask(1, 5))
imgFilterGaussMask_2_9 = cv2.filter2D(img, -1, applyGaussMask(2, 9))
imgFilterGaussMask_4_15 = cv2.filter2D(img, -1, applyGaussMask(4, 15))

cv2.imwrite("gaus_1_5.png", imgFilterGaussMask_1_5)
cv2.imwrite("gaus_2_9.png", imgFilterGaussMask_2_9)
cv2.imwrite("gaus_4_15.png", imgFilterGaussMask_4_15)