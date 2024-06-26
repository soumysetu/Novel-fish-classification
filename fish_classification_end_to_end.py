import requests 
import cv2 
import numpy as np 
import imutils 
import RPi.GPIO as GPIO
#from cv2 import *
import asyncio
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last. 
url = "http://192.168.122.36:8080/shot.jpg"

# While loop to continuously fetching data from the Url 
 
img_resp = requests.get(url) 
img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
img = cv2.imdecode(img_arr, -1) 
img = imutils.resize(img, width=1000, height=1800) 

cv2.imwrite("image.png", img) 

cv2.destroyAllWindows() 




print("hello ji")



import tensorflow as tf
import matplotlib.pyplot as plt


test_set= tf.keras.utils.image_dataset_from_directory(
    "./test",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)
cnn=tf.keras.models.load_model("./trained_model.h5")
import cv2
image_path="./image.png"
img=cv2.imread(image_path)
#plt.imshow(img)
#plt.title("Test Image")
#plt.xticks([])
#plt.yticks([])
#plt.show()
image=tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
input_arr=tf.keras.preprocessing.image.img_to_array(image)
input_arr=np.array([input_arr]) #convert single image to batch
predictions=cnn.predict(input_arr)
#print(predictions[0])
#print(max(predictions[0]))
test_set.class_names
result_index=np.where(predictions[0]==max(predictions[0]))
#print(result_index)
#display image
#plt.imshow(img)
#plt.title("Test Image")
#plt.xticks([])
#plt.yticks([])
#plt.show()
#single prediction
print("It's a {}".format(test_set.class_names[result_index[0][0]]))
a = format(test_set.class_names[result_index[0][0]])
print(a)


import time

# Set GPIO numbering mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
IN1 = 23
IN2 = 24
IN3 = 25
IN4 = 27


# Setup GPIO pins
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Function to drive motors forward
def forward1(duration, speed):
    GPIO.output(IN1, GPIO.HIGH)

    time.sleep(duration)
    stop1()
# Function to drive motors forward
def forward2(duration, speed):
    GPIO.output(IN2, GPIO.HIGH)

    time.sleep(duration)
    stop2()
# Function to drive motors forward
def forward3(duration, speed):
    GPIO.output(IN3, GPIO.HIGH)

    time.sleep(duration)
    stop3()
# Function to drive motors forward
def forward4(duration, speed):
    GPIO.output(IN4, GPIO.HIGH)

    time.sleep(duration)
    stop4()


def stop1():
    GPIO.output(IN1, GPIO.LOW)
def stop2():
    GPIO.output(IN2, GPIO.LOW)
def stop3():
    GPIO.output(IN3, GPIO.LOW)
def stop4():
    GPIO.output(IN4, GPIO.LOW)


if a=="Sederine":
    try:
        forward1(0.5, 100)
        stop1()  # Move forward for 2 seconds at 50% speed
    except KeyboardInterrupt:
        stop1()  # Stop motors when program is interrupted
        GPIO.cleanup()  # Clean up GPIO
if a=="Sole":
    try:
        forward3(0.5, 100)
        stop3()  # Move forward for 2 seconds at 50% speedaaaaa
    except KeyboardInterrupt:
        stop3()  # Stop motors when program is interrupted
        GPIO.cleanup()  # Clean up GPIO
else:
	print("no motor is assign")