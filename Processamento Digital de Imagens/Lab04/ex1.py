import cv2
import numpy as np

def applyFilter(img, image_filter, size):
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            window = img[max(0, y - size // 2):min(height, y + size // 2 + 1),
                            max(0, x - size // 2):min(width, x + size // 2 + 1)]
            media = np.mean(window)
            image_filter[y, x] = media
    return image_filter

def makeFilter(img, size):
    image_filter = np.zeros_like(img)
    image_filter = applyFilter(img, image_filter, size)

    return image_filter

img = cv2.imread("Lab04 - Joao Vitor/sta2.png", cv2.IMREAD_GRAYSCALE)

img_3x3 = makeFilter(img, 3)
img_7x7 = makeFilter(img_3x3, 7)
img_2_3x3 = makeFilter(img_7x7, 3)

cv2.imwrite("3x3_1.png", img_3x3)

cv2.imwrite("7x7.png", img_7x7)

cv2.imwrite("3x3_2.png", img_2_3x3)

