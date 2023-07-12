import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("C:/Users/Virat/Downloads/Projects/Automatic_attendence_system_using_facial_recognition_python_openCV-main/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])

            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)

        except Exception as e:
            print(f"Error processing image: {imagePath}")
            print(f"Error message: {str(e)}")

    return faceSamples, Ids

dataset_path = ('C:/Users/Virat/Downloads/Projects/Automatic_attendence_system_using_facial_recognition_python_openCV-main/dataset/')
faces, Ids = getImagesAndLabels(dataset_path)

if len(faces) > 0 and len(Ids) > 0:
    recognizer.train(faces, np.array(Ids))
    recognizer.write('C:/Users/Virat/Downloads/Projects/Automatic_attendence_system_using_facial_recognition_python_openCV-main/trainer/trainer.yml')
    print("Successfully trained")
else:
    print("No training data found. Please ensure the dataset contains images.")
