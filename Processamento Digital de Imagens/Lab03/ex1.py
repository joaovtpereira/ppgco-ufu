import numpy as np
import cv2

def negativeValue(pixel):
    return 255 - pixel

def logImg(img):
    image_matrix = np.matrix(img)
    max_value = np.max(image_matrix)
    constant = round(255/np.log(1+max_value))
    log_img = np.log(1 + image_matrix)
    log_img *= constant
    log_img = np.round(log_img)
    return log_img

updateValueToNegative = np.vectorize(negativeValue)

imgPath = "/Users/joaovitorpereira/Documents/mestrado/Repositorios/pgc111-labs-2023-02/Lab03 - Joao Vitor/im_cells.png"

img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

imgNegative = updateValueToNegative(img) 

revertImg = updateValueToNegative(imgNegative) 

logaritImage = logImg(img)

cv2.imwrite("negative.png", imgNegative)

cv2.imwrite("revert.png", revertImg)

cv2.imwrite("log_image.png", logaritImage)




