import cv2
import matplotlib.pyplot as plt

# acao de ler a msg
img = cv2.imread("pgc111-labs-2023-02/Lab00 - Joao Vitor/foz.jpg", cv2.COLOR_BGR2RGB)
plt.imshow(img)
print("finalizou")