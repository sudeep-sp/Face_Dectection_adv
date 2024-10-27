import cv2 as cv
import os
import uuid
import mediapipe as mp

# Ask for the name to create the folder
name = input("Enter the name for the folder: ")

# Define the output directory
output_dir = f"data/faces/train/{name}"

# Check if the directory already exists
if os.path.exists(output_dir):
    print(f"Folder '{
          output_dir}' already exists. Images will be saved in the existing folder.")
else:
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")

cap = cv.VideoCapture(0)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5)

count = 0
while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Process the image and detect faces
    results = face_detection.process(img_rgb)

    # If faces are detected, save the image
    if results.detections:
        img_name = os.path.join(output_dir, f"{name}_{str(uuid.uuid4())}.jpg")
        cv.imwrite(img_name, img)
        count += 1

        # Draw face detections on the image
        for detection in results.detections:
            mp_drawing.draw_detection(img, detection)

    cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

cap.release()
cv.destroyAllWindows()
