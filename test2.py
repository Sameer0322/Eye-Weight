# Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
# face_utils for basic operations of conversion
from imutils import face_utils

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    x1=x2=y1=y2 = 0
    status=""
    color = (0, 0, 0)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Create a bounding box frame for landmarks
        face_frame_landmarks = frame.copy()
        cv2.rectangle(face_frame_landmarks, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Initialize status for each face
        status = ""
        color = (0, 0, 0)

        # Update status based on blink detection
        if left_blink == 0 or right_blink == 0:
            status = "SLEEPING"
            color = (255, 0, 0)
        elif left_blink == 1 or right_blink == 1:
            status = "Drowsy"
            color = (0, 0, 255)
        else:
            status = "Active"
            color = (0, 255, 0)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame_landmarks, (x, y), 1, (255, 255, 255), -1)

        # Display the frame with landmarks and bounding box
        cv2.imshow("Landmarks Frame", face_frame_landmarks)

    # Create a frame for the status text and bounding box
    frame_with_status = frame.copy()
    cv2.rectangle(frame_with_status, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_with_status, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Display the frame with the status text and bounding box
    cv2.imshow("Status Frame", frame_with_status)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


