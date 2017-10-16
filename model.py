import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D

def process_image(img):
    return img

rows = []
with open('data/internet/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        rows.append(line)
        
car_images = []
steering_angles = []

first = True

steering_correction = 0.2

for j, row in enumerate(rows):
    print("(", j+1, "/", len(rows), ")", round((j+1)/len(rows)*100,2), "% completed...", end='\r')
    
    # Skip header row
    if first:
        first = False
        continue
        
    for i in range(3):
        steering_center = float(row[3])

        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction

        path = "data/internet/IMG/"
        img_loc = row[i].split('/')[-1]
        img = process_image(cv2.imread(path + img_loc))

        car_images.append(img)
        car_images.append(cv2.flip(img, 1))
        
        angle = 0
        
        if i == 0: 
            angle = steering_center
        elif i == 1: 
            angle = steering_left
        elif i == 2:
            angle = steering_right
        else:
            raise Exception('Uhhhm?')
        
        steering_angles.append(angle)
        steering_angles.append(-angle)

X_train = np.array(car_images)
y_train = np.array(steering_angles)

print()
print("Done!")

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')