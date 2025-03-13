# Sign-language-translator
This repository contains a Python-based application that detects sign language gestures using a webcam feed. The project utilizes deep learning models, OpenCV for video processing, and TensorFlow for gesture classification. The main objective is to recognize hand gestures in real time and display their corresponding labels.

# Features
Real-Time Hand Gesture Detection: Detects hand gestures through a webcam feed.

Deep Learning Model: Utilizes a TensorFlow-based model to classify gestures.

Dynamic Cropping: Automatically identifies the hand in the webcam feed and adjusts the region of interest.

Interactive Interface: Displays the webcam feed and gesture classification results directly on the screen.

# Prerequisites
Make sure you have the following installed:

Python 3.7 or higher

TensorFlow 2.x

OpenCV 4.x

NumPy

cvzone (for hand detection)

# How it Works
Webcam Feed: The application accesses the webcam and displays a live feed.

# Hand Gesture

OpenCV and the cvzone.HandTrackingModule detect the hand in the frame.

The bounding box around the hand is dynamically cropped to focus on the hand region.

# Gesture Classficaton

The cropped hand region is preprocessed and fed into the TensorFlow model.

The model predicts the gesture and displays the corresponding label on the screen.

# Feedback Display

The application overlays the predicted label and bounding box on the webcam feed.

Labels such as "No Hands Detected" are shown when no hand is present in the frame.

Model Information:
Model Type: TensorFlow SavedModel format.

Input Size: 300x300 pixels.

Output: Predicted gesture label based on the input image.

# Limitations
Works best with a well-lit environment.

May struggle with overlapping hands or multiple hands in the frame.

Accuracy depends on the quality of the training dataset.

# Future Enhancements
Add support for additional gestures.

Improve gesture classification accuracy with a larger dataset.

Implement multi-hand detection and classification.

Create a web-based interface for easier deployment.
