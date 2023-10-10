import cv2
import numpy as np

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

img_camera = cv2.imread("Lab03 - Joao Vitor/cameraman.tif")

img_cells = cv2.imread("Lab03 - Joao Vitor/im_cells.png")

imgCameraEqualization1 = equalizationImage(img_camera)
imgCameraEqualization2 = equalizationImage(imgCameraEqualization1)

imgCellsEqualization1 = equalizationImage(img_cells)
imgCellsEqualization2 = equalizationImage(imgCellsEqualization1)

cv2.imwrite('imgCameraEqualized1.png', imgCameraEqualization1)
cv2.imwrite('imgCameraEqualized2.png', imgCameraEqualization2)
cv2.imwrite('imgCellsEqualized1.png', imgCellsEqualization1)
cv2.imwrite('imgCellsEqualized2.png', imgCellsEqualization2)
