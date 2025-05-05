import cv2
import pathlib

# Set path for Haar Cascade
cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'

# Check if the cascade file exists
if not pathlib.Path(cascade_path).exists():
    print(f"Error: Cascade file not found at {cascade_path}")
    exit()

# Load the CascadeClassifier
clf = cv2.CascadeClassifier(cascade_path)
if clf.empty():
    print(f"Error: Failed to load cascade classifier from {cascade_path}")
    exit()

# Open the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,    # Increased to improve detection
        minSize=(30, 30),  # Slightly adjusted for better face detection
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Display the result
    cv2.imshow("Faces", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
