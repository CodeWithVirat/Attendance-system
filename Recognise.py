import cv2
import numpy as np
import xlwt
from datetime import datetime
from xlrd import open_workbook
from xlwt import Workbook
from xlutils.copy import copy
from pathlib import Path
import time
from playsound import playsound
import os

save_directory = "C:/Users/Virat/Downloads/Projects/Firebase/"

def output(filename, sheet, num, name, present):
    my_file = Path(save_directory + filename + str(datetime.now().date()) + '.xls')
    if my_file.is_file():
        rb = open_workbook(save_directory + filename + str(datetime.now().date()) + '.xls')
        book = copy(rb)
        sh = book.get_sheet(0)
    else:
        book = Workbook()
        sh = book.add_sheet(sheet)

    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on',
                         num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')

    sh.write(0, 0, datetime.now().date(), style1)

    col1_name = 'Name'
    col2_name = 'Present'

    sh.write(1, 0, col1_name, style0)
    sh.write(1, 1, col2_name, style0)

    sh.write(num + 1, 0, name)
    sh.write(num + 1, 1, present)

    fullname = filename + str(datetime.now().date()) + '.xls'
    book.save(save_directory + fullname)
    return fullname

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

start = time.time()
period = 8
face_cas = cv2.CascadeClassifier('C:/Users/Virat/Downloads/Projects/Automatic_attendence_system_using_facial_recognition_python_openCV-main/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/Virat/Downloads/Projects/Automatic_attendence_system_using_facial_recognition_python_openCV-main/trainer/trainer.yml')
flag = 0
filename = 'attendance'
attendance_dict = {}
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.3, 7)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = recognizer.predict(roi_gray)
        
        if conf < 50:
            if id == 1:
                id = 'Virat'
                if str(id) not in attendance_dict:
                    filename = output('attendance', 'class1', 1, id, 'yes')
                    attendance_dict[str(id)] = str(id)
            elif id == 2:
                id = 'Bipodtaran'
                if str(id) not in attendance_dict:
                    filename = output('attendance', 'class1', 2, id, 'yes')
                    attendance_dict[str(id)] = str(id)
            elif id == 3:
                id = 'Chandana'
                if str(id) not in attendance_dict:
                    filename = output('attendance', 'class1', 3, id, 'yes')
                    attendance_dict[str(id)] = str(id)
        else:
            id = 'Unknown, cannot recognize'
            flag = flag + 1
            break
        
        cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
    
    cv2.imshow('frame', img)
    
    if flag == 10:
        playsound('transactionSound.mp3')
        print("Transaction Blocked")
        break
    
    if time.time() > start + period:
        break
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
