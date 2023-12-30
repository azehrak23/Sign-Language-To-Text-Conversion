
# Sign-Language-to-Text-Conversion

This project focuses on bridging the communication gap between sign language users and non-signers by developing a real-time sign language to text conversion system. Leveraging computer vision and deep learning techniques, it recognizes and translates sign language gestures captured through a webcam into textual representations. Additionally, the project facilitates the collection of diverse gesture datasets for training robust recognition models, enabling further advancements in gesture recognition technology.
## Project Description

The project aims to develop a system capable of real-time translation of sign language gestures into text. Using computer vision and deep learning techniques, the system will interpret live gestures captured via a webcam and instantly convert them into understandable text, facilitating communication between sign language users and those unfamiliar with signing.

Sign language is a visual language and consists of 3 major components 

![components](components.jpg)

In this project I basically focus on producing a model which can recognize Fingerspelling based hand gestures in order to form a complete word by combining each gesture. 

The gestures I  trained are as given in the image below.

![Signs](signs.jpg)

## Features

### Sign Language to Text Conversion

* Real-time sign language translation through a webcam.
* Uses image processing, neural networks (specifically Keras), and language processing libraries.
* Generates text output from recognized sign gestures.

### Data Collection for Gesture Recognition:

* Helps collect data for training a model to recognize sign language gestures.
* Utilizes OpenCV for image processing and directory management for organizing collected data.

## Software Dependencies

1. **Libraries**

* OpenCV for video capture and image processing.
* Keras for deep learning models.
* Hunspell and enchant for language processing.
* tkinter and PIL for the graphical user interface.
* NumPy for numerical operations.

2. Neural Network

* Uses pre-trained models in Keras for recognizing sign language gestures.

## Steps to Build

**Step 1**: Setting up Dependencies and Libraries

### Install necessary libraries
_!pip install opencv-python keras numpy pillow_

**Step 2**: Loading Pre-Trained Model for Sign Recognition

_from keras.models import model_from_json_

### Load pre-trained model architecture from JSON file
_json_file = open('sign_language_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)_

### Load pre-trained model weights
_loaded_model.load_weights('sign_language_model.h5')_

**Step 3**: Implementing Image Processing and Gesture Recognition

_import cv2_

_import numpy as np_

_cap = cv2.VideoCapture(0)_

_while True:
ret, frame = cap.read()
processed_frame = preprocess(frame)_
    
_prediction = loaded_model.predict(processed_frame)_
    
_cv2.imshow('Sign Language Recognition', frame)_
    
_if cv2.waitKey(1) & 0xFF == ord('q'):
break_

_cap.release()
cv2.destroyAllWindows()_

**Step 4**: Setting up OpenCV for Image Capture

import cv2

### Access the webcam
_cap = cv2.VideoCapture(0)_

### Define keys to save images of gestures
_key_mapping = {
    ord('a'): 'A', 
    ord('b'): 'B',
    # Add more mappings for other gestures
}_

_while True:
ret, frame = cap.read()
cv2.imshow('Collecting Gestures', frame)_
    
_key = cv2.waitKey(1)
if key in key_mapping:
gesture = key_mapping[key]
cv2.imwrite(f'dataset/{gesture}/gesture_{len(os.listdir(f"dataset/{gesture}"))}.jpg', frame)_
    
_if cv2.waitKey(1) & 0xFF == ord('q'):
break_

_cap.release()
cv2.destroyAllWindows()_

**Step 5**: Saving Images into Respective Directories

Ensure you have a directory structure for each gesture, like 'A', 'B', etc., to organize the collected images.
