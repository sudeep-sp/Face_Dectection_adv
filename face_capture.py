import cv2 as cv
import os
import uuid
import mediapipe as mp
import random

# Ask for the name to create the folder
name = input("Enter the name for the folder: ")

# Define the output directory
training_output_dir = f"data/faces/train/{name}"
testing_output_dir = f'data/faces/test/{name}'

# Check if the directory already exists
if os.path.exists(training_output_dir):
    print(
        f"name '{name}' already exists. Images will be saved in the existing folder.")
else:
    os.makedirs(training_output_dir)
    os.makedirs(testing_output_dir)
    print(f"name '{name}' is added to database")

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
        img_name = os.path.join(training_output_dir, f"{
                                name}_{str(uuid.uuid4())}.jpg")
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


training_images = os.listdir(training_output_dir)

test_images = random.sample(training_images, 20)

for img_name in test_images:
    src_path = os.path.join(training_output_dir, img_name)
    dst_path = os.path.join(testing_output_dir, img_name)
    os.rename(src_path, dst_path)
