import cv2
import numpy as np
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

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
    all_landmarks = []
    all_statuses = []

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        all_landmarks.extend(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        status = ""
        color = (0, 0, 0)

        if left_blink == 0 or right_blink == 0:
            status = "SLEEPING"
            color = (255, 0, 0)
        elif left_blink == 1 or right_blink == 1:
            status = "Drowsy"
            color = (0, 0, 255)
        else:
            status = "Active"
            color = (0, 255, 0)

        all_statuses.append((status, color, (x1, y1, x2, y2)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print(f"Face Status: {status}")
        
    landmarks_frame = frame.copy()
    for (x, y) in all_landmarks:
        cv2.circle(landmarks_frame, (x, y), 1, (255, 255, 255), -1)
    cv2.imshow("Facial Landmarks", landmarks_frame)

    status_frame = frame.copy()
    for status, color, (x1, y1, x2, y2) in all_statuses:
        cv2.rectangle(status_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(status_frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Status of Faces", status_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
