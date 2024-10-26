import cv2 as cv
import os
import uuid

name = input('please enter you name : ')

training_data_folder = f'data/faces/train/{name}'

if os.path.exists(training_data_folder):
    print(f'the folder {name} is already exists at {
          training_data_folder},so data will be saved there..')
else:
    os.makedirs(training_data_folder)
    print(f'folder {name} is created')


cap = cv.VideoCapture(0)

count = 0
while True:

    success, img = cap.read()
    if not success:
        break

    img_name = os.path.join(training_data_folder, f'{
                            name}_{str(uuid.uuid1())}.jpg')
    cv.imwrite(img_name, img)
    count += 1

    cv.imshow('frame', img)
    if cv.waitKey(1) & count >= 100:
        break

cap.release()
cv.destroyAllWindows()
