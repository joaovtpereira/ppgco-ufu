import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalization(hist):
    return ((hist - hist.min()) / (hist.max() - hist.min()) * 255).astype('uint8')

def acumulation_histogram(hist):
    acumulation = np.zeros(len(hist), dtype=int)
    acumulation[0] = hist[0]
    for i in range(1, len(hist)):
        acumulation[i] = acumulation[i - 1] + hist[i]
    return acumulation

def equalizationImage(img):
    img_matrix = np.asarray(img)
    imgHist, _ = np.histogram(img_matrix.flatten(), bins=256, range=(0, 256))
    acumulation = acumulation_histogram(imgHist)
    acumulation = normalization(acumulation)
    img_equalizate = acumulation[img_matrix]

    return img_equalizate

img = cv2.imread("Lab03 - Joao Vitor/pout.tif")

histImg = cv2.calcHist([img], [0], None, [256], [0, 256])
cdfImg = np.cumsum(histImg)
img_equalized = equalizationImage(img)
histImgEqualized = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
cdfImgEqualized = np.cumsum(histImgEqualized)

plt.figure(figsize=(8, 6))
plt.title('Histograma da Imagem')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Frequência')
plt.plot(histImg)
plt.xlim([0, 256])
plt.show()

plt.figure(figsize=(8, 6))
plt.title('Curva de transformacao Imagem')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Frequência')
plt.plot(cdfImg)
plt.xlim([0, 256])
plt.show()

plt.figure(figsize=(8, 6))
plt.title('Histograma da Imagem Equalizada')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Frequência')
plt.plot(histImgEqualized)
plt.xlim([0, 256])
plt.show()

plt.figure(figsize=(8, 6))
plt.title('Curva de transformacao Imagem Equalizada')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Frequência')
plt.plot(cdfImgEqualized)
plt.xlim([0, 256])
plt.show()