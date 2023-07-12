import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

face_id = input('Enter your ID: ')
dataset_path = "C:/Users/Virat/Downloads/Projects/Data Set/"

# Start capturing video
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier("C:/Users/Virat/Downloads/Projects/Automatic_attendence_system_using_facial_recognition_python_openCV-main/haarcascade_frontalface_default.xml")

# Initialize sample face image
count = 0

assure_path_exists(dataset_path)

# Start looping
while True:
    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loop over each face
    for (x, y, w, h) in faces:
        # Crop the image frame into a rectangle
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Increment the sample face image count
        count += 1

        # Save the captured image into the datasets folder
        image_path = os.path.join(dataset_path, f"User.{face_id}.{count}.jpg")
        cv2.imwrite(image_path, gray[y:y + h, x:x + w])

        # Display the video frame with a bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If 50 images are captured, stop taking video
    elif count >= 50:
        print("Successfully Captured")
        break

# Stop video
vid_cam.release()

# Close all windows
cv2.destroyAllWindows()
