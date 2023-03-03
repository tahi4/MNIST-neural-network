import os
import numpy as np
import cv2 #computer vision
import tensorflow as tf
import matplotlib.pyplot as plt

# PERSONAL DATA TESTING

model = tf.keras.models.load_model('handwritten.model')


image_number = 1

# reading images

while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:, 0] #looks at the shape of img, not colors
        img= np.invert(np.array([img])) #bcs by default its black on white, so we need to invert it to make it white on black
        prediction = model.predict(img)
        print(f"digit is probably {np.argmax(prediction)}") #np.argmax gives us the highest likely index
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show() #shows img
    except:
        print("Error!")
    
    finally:
        image_number +=1

