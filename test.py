import cv2

# Load the pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect the state of the eyes
def detect_eyes_state(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            return "Open"
        elif len(eyes) == 1:
            return "Semi-Closed"
        else:
            return "Closed"

# Load the image or start the video capture
# image = cv2.imread("path_to_image.jpg")
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    eyes_state = detect_eyes_state(frame)

    # Display the state of the eyes in red color
    cv2.putText(frame, f"Eyes State: {eyes_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eyes Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()