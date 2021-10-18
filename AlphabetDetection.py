# Import all the necessary libraries to the file.
from sklearn.datasets import fetch_openml
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 
import os, ssl, time

# Load the image.npz file and read the labels.csv file.
x = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv.txt")["labels"]
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

# Split the data to train and test the model
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 9, train_size = 7500, test_size = 2500)
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0
# Fit the data into the model.
# Code to make a prediction and print the accuracy.
classifier = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_train_scale, y_train)
y_predict = classifier.predict(x_test_scale)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

# Code to open the camera and start reading the frames.
cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        # Draw a box at the center of the screen. And consider only the area inside the box to detect the images.
        height, width = gray.shape()
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right= (int(width/2 + 56), int(height/2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0,255,0))

        # Convert the image to pil format.
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        pil_format = Image.fromarray(roi)
        
        # Convert to grayscale image. - 'L' format mans each pixel is rpresented by a single value from 0 to 255
        image_bw = pil_format.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
        # invert the image
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        # converting to scalar quantitiy
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        # using clip to limit the values between 0,255 
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0,255)
        max_pixel = np.max(image_bw_resized_inverted)
        # converting into an array
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        # creating a test sample and making a prediction
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_predict = classifier.predict(test_sample)
        print("Predicted class is: ", test_predict)

        cv2.imshow("frame", gray)
        
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
    
    except Exception as e:
        pass


cap.release()
cv2.destroyAllWindows()