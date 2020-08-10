import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#img = cv2.imread('RDJ.jpg')
img = cv2.imread('group.jpg')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('JP Scriven Face Detector', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()
# Convert img to grayscale
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect Faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# cv2.imshow('JP Scriven Face Detector', img)
# # Wait to exit till key pressed
# cv2.waitKey()



print("Code Completed...")