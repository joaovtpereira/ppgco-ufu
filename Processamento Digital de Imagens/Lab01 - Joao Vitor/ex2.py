import cv2
import matplotlib.pyplot as plt
import numpy as np

image_bits = 256
downgrade_level = 2

def upgradePixelValue(pixel):
    pixel = round((pixel / (image_bits-1)) * (round(image_bits/downgrade_level) - 1))
    return pixel

def updateImageBitValue(current_bit_max_value):
    global downgrade_level
    return current_bit_max_value/downgrade_level

updateValues = np.vectorize(upgradePixelValue)
    
img = cv2.imread("Lab01 - Joao Vitor/ctskull-256.tif", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)

img_128 = updateValues(img)
plt.imshow(img_128, cmap="gray", vmin=0, vmax=127)
image_bits = updateImageBitValue(image_bits)

img_64 = updateValues(img_128)
plt.imshow(img_64, cmap="gray", vmin=0, vmax=63)
image_bits = updateImageBitValue(image_bits)

img_32 = updateValues(img_64)
plt.imshow(img_32, cmap="gray", vmin=0, vmax=31)
image_bits = updateImageBitValue(image_bits)

img_16 = updateValues(img_32)
plt.imshow(img_16, cmap="gray", vmin=0, vmax=15)
image_bits = updateImageBitValue(image_bits)

img_8 = updateValues(img_16)
plt.imshow(img_8, cmap="gray", vmin=0, vmax=7)
image_bits = updateImageBitValue(image_bits)

img_4 = updateValues(img_8)
plt.imshow(img_4, cmap="gray", vmin=0, vmax=3)
image_bits = updateImageBitValue(image_bits)


img_2 = updateValues(img_4)
plt.imshow(img_2, cmap="gray", vmin=0, vmax=1)

print("teste")