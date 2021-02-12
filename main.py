import cv2
from random import randrange
# load some pre-trained data on face frontals from open cv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture the webcam
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    successful_frame_read, frame = webcam.read()
    # Converting image in gray_scale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # Draw rectangle on face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)
    cv2.imshow('Real Time Face Detector', frame)
    cv2.waitKey(1)

