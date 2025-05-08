import cv2
import pathlib
from deepface import DeepFace
import numpy as np

# Haar cascade path
cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'

if not pathlib.Path(cascade_path).exists():
    print(f"Error: Cascade file not found at {cascade_path}")
    exit()

clf = cv2.CascadeClassifier(cascade_path)
if clf.empty():
    print(f"Error: Failed to load cascade classifier from {cascade_path}")
    exit()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        try:
            # Resize to 224x224 for best model performance
            resized_face = cv2.resize(face_roi, (224, 224))

            # Analyze with DeepFace (only emotion detection)
            analysis = DeepFace.analyze(
                resized_face,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            emotion = "Unknown"

        label = f"{emotion}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

