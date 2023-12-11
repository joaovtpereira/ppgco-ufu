import cv2
import numpy as np

def compute_lbp(image, P=8, R=1):
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Obter as dimensões da imagem
    height, width = gray.shape
    
    # Inicializar a matriz de resultados LBP
    lbp_result = np.zeros_like(gray, dtype=np.uint8)
    
    # Calcular o LBP para cada pixel
    for y in range(R, height - R):
        for x in range(R, width - R):
            # Obter os valores dos pixels vizinhos ao redor do pixel central
            values = [gray[y - R, x],
                      gray[y - R, x + R],
                      gray[y, x + R],
                      gray[y + R, x + R],
                      gray[y + R, x],
                      gray[y + R, x - R],
                      gray[y, x - R],
                      gray[y - R, x - R]]
            
            # Calcular o padrão binário LBP
            binary_pattern = 0
            for i in range(P):
                binary_pattern |= (values[i] >= values[(i + 1) % P]) << i
            
            # Atribuir o valor LBP ao pixel correspondente
            lbp_result[y, x] = binary_pattern
    
    return lbp_result

# Função para aplicar ajuste de contraste
def adjust_contrast(image, alpha=1.2, beta=10):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# Função para aplicar filtro de mediana
def apply_median_filter(image, kernel_size=5):
    median_filtered = cv2.medianBlur(image, kernel_size)
    return median_filtered

# Função para aplicar equalização do histograma
def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def HasAlgorithm(image):
    # Carregue o classificador Haar para detecção de faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detecte faces na imagem
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhe retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

# Capturar vídeo
cap = cv2.VideoCapture(1)  # 0 para a câmera padrão, você pode alterar para o nome do arquivo de vídeo

while True:
    ret, frame = cap.read()

    # Aplicar ajuste de contraste
    frame_contrast = adjust_contrast(frame)

    # # Aplicar filtro de mediana
    frame_median = apply_median_filter(frame_contrast)

    # # Aplicar equalização do histograma
    frame_equalized = apply_histogram_equalization(frame_median)

    imageDetectFace = HasAlgorithm(frame_equalized)

    # # Calcular o LBP na imagem
    # lbp_result = compute_lbp(frame)

    cv2.imshow('Face detectada', imageDetectFace)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura quando tudo estiver feito
cap.release()
cv2.destroyAllWindows()
