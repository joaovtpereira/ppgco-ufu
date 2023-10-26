import cv2
import matplotlib.pyplot as plt
import numpy as np

line_inches_size = 2.96
column_inches_size = 2.25

linhas = 0
colunas = 0

original_image = cv2.imread("Lab01 - Joao Vitor/relogio.tif", cv2.IMREAD_GRAYSCALE)

def calculateJumpBitsResize(dpiOriginal, newDpi):
    return round(dpiOriginal/newDpi)

def createMatrixAndResizeImage(image_base, oldDpi, newDpi):
    global line_inches_size
    global column_inches_size
    global linhas
    global colunas

    jumpBitsQuantify = calculateJumpBitsResize(oldDpi, newDpi)
    newImageMatrix = np.zeros((round(line_inches_size * newDpi), round(column_inches_size * newDpi)))
    newImageMatrix = np.matrix(image_base[0:linhas:jumpBitsQuantify, 0:colunas:jumpBitsQuantify])
  
    return newImageMatrix

def getLinesAndColumnsLength(image_base):
    linhas = image_base.shape[0]
    colunas = image_base.shape[1]
    return {linhas, colunas}

plt.imshow(original_image, cmap="gray", vmin=0, vmax=255)
# linhas = original_image.shape[0]
# colunas = original_image.shape[1]
[linhas, colunas] = getLinesAndColumnsLength(original_image)

img_300dpi_matrix = createMatrixAndResizeImage(original_image, 1250, 300)
plt.imshow(img_300dpi_matrix, cmap="gray", vmin=0, vmax=255)

[linhas, colunas] = getLinesAndColumnsLength(img_300dpi_matrix)
img_150dpi_matrix = createMatrixAndResizeImage(img_300dpi_matrix,300, 150)
plt.imshow(img_150dpi_matrix, cmap="gray", vmin=0, vmax=255)

img_75dpi_matrix = createMatrixAndResizeImage(img_150dpi_matrix,150, 75)
plt.imshow(img_75dpi_matrix, cmap="gray", vmin=0, vmax=255)