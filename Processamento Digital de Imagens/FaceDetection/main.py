import cv2
import numpy as np
import os

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

def HasAlgorithmAndSaveImage(image, person_name, count):
    # Carregue o classificador Haar para detecção de faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detecte faces na imagem
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    # Desenhe retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Store the captured face images in the Faces folder
        cv2.imwrite(f'/Users/joaovitorpereira/Documents/mestrado/Repositorios/ppgco-ufu/Processamento Digital de Imagens/FaceDetection/Faces/{person_name}_{count}.jpg', image[y:y + h, x:x + w])
    return image


# Create a face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def captureImage(name_of_person): 
    # Capturar vídeo
    cap = cv2.VideoCapture(1)  # 0 para a câmera padrão, você pode alterar para o nome do arquivo de vídeo

    # Set the image counter as 0
    count = 0

    while True:
        ret, frame = cap.read()

        # Aplicar ajuste de contraste
        frame_contrast = adjust_contrast(frame)

        # # Aplicar filtro de mediana
        frame_median = apply_median_filter(frame_contrast)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        imageDetectFace = HasAlgorithmAndSaveImage(gray, name_of_person, count)
        count+=1
        print(count)

        cv2.imshow('Face detectada', imageDetectFace)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop after capturing a certain number of images
        if count >= 1000:
            break
    
    # Liberar a captura quando tudo estiver feito
    cap.release()
    cv2.destroyAllWindows()

def train_model(label):
	# Create lists to store the face samples and their corresponding labels
	faces = []
	labels = []
	
	# Load the images from the 'Faces' folder
	for file_name in os.listdir('Processamento Digital de Imagens/FaceDetection/Faces'):
		if file_name.endswith('.jpg'):
			# Extract the label (person's name) from the file name
			name = file_name.split('_')[0]
			
			# Read the image and convert it to grayscale
			image = cv2.imread(os.path.join('Processamento Digital de Imagens/FaceDetection/Faces', file_name))
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# Detect faces in the grayscale image
			detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

			# Check if a face is detected
			if len(detected_faces) > 0:
				# Crop the detected face region
				face_crop = gray[detected_faces[0][1]:detected_faces[0][1] + detected_faces[0][3],
								detected_faces[0][0]:detected_faces[0][0] + detected_faces[0][2]]

				# Append the face sample and label to the lists
				faces.append(face_crop)
				labels.append(label[name])

	# Train the face recognition model using the faces and labels
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.train(faces, np.array(labels))

	# Save the trained model to a file
	recognizer.save('trained_model.xml')
	return recognizer

# Function to recognize faces
def recognize_faces(recognizer, label):
    # Open the camera
    cap = cv2.VideoCapture(0)
     
    # Reverse keys and values in the dictionary
    label_name = {value: key for key, value in label.items()}
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
 
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
         
        # Recognize and label the faces
        for (x, y, w, h) in faces:
            # Recognize the face using the trained model
            label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            #print(confidence)
            if confidence > 20:
                # Display the recognized label and confidence level
                cv2.putText(frame, label_name[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
     
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print('Unrecognized')
 
        # Display the frame with face recognition
        cv2.imshow('Recognize Faces', frame)
 
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Primeiro capture as images, informar o nome da pessoa
# captureImage("name")
# captureImage("name2")

# Depois adiciona o label com nomes deles
#label = {'name':0,'name2':1}

# # Train the model
#recognizer = train_model(label)
## recognizer
#recognize_faces(recognizer, label)

