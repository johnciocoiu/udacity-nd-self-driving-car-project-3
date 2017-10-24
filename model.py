import csv
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D, Dropout

# Save location of data files, so this data does not have to be generated over and over again when chaging the network.
X_file = 'data/X_file.npy'
y_file = 'data/y_file.npy'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 10, "The number of epochs.")

# The folders to read the data from, multiple drivings around the track
# internet = from udacity
# track1 = driving mouse on track 1
# track1_2 = driving mouse and recovering from outside the track
# track1_reverse = driving mouse track 1 the other way around
# training = driving with keyboard
# track2 = single lap on track 2 with mouse
data_folders = ['data/track2/', 'data/internet/', 'data/track1/', 'data/track1_2/', 'data/track1_reverse/', 'data/training/']

car_images = []
steering_angles = []

# For every folder, add the data to the arrays
for fnum, folder in enumerate(data_folders):
    print ""
    print "Doing folder ", fnum+1, "/", len(data_folders),
    print ""
    rows = []
    with open(folder+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            rows.append(line)

    first = True

    # Add a steering correction
    steering_correction = 0.2
    for j, row in enumerate(rows):
        # Print the current finishing of the current folder
        print "\r", j+1, "/", len(rows), "completed...",
    
        # Skip header row
        if first:
            first = False
            continue
            
        # For every of the 3 generated images, do the data generate thing
        for i in range(3):
            steering_center = float(row[3])

            steering_left = steering_center + steering_correction
            steering_right = steering_center - steering_correction

            path = folder+"/IMG/"
            img_loc = row[i].split('/')[-1]
            img = cv2.imread(path + img_loc)

            # Add images to array
            car_images.append(img)

            # Add flipped images to array
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
            
            # Add steering angle to array
            steering_angles.append(angle)

            # Add flipped steering angle to array
            steering_angles.append(angle*-1.0)

# Generate np arrays
X_train = np.array(car_images)
y_train = np.array(steering_angles)

print ""
print "Done making training set!" 

# The model, based on https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf given in the lecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20), (0,0))))
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
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=FLAGS.epochs)

# Save the model to a file
model.save('model.h5')