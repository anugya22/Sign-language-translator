import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math

# Initialize components
cap = cv2.VideoCapture(0)  # Open the webcam
detector = HandDetector(maxHands=1)  # Detect a maximum of one hand
model = load_model("Model/keras_model.keras")  # Load the trained model
offset = 20  # Padding around the hand bounding box
imgSize = 300  # Input size for the model

# Load labels from the labels file
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f]

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to access the webcam.")
        break

    imgOutput = img.copy()  # Make a copy of the frame to display predictions
    hands, img = detector.findHands(img, flipType=False)  # Detect hands in the frame

    if hands:
        # Get the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Bounding box around the hand

        # Ensure cropping bounds are valid
        if y - offset < 0 or x - offset < 0 or y + h + offset > img.shape[0] or x + w + offset > img.shape[1]:
            cv2.putText(imgOutput, "Hand partially outside frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Image", imgOutput)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Preprocess the hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Adjust aspect ratio and resize
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Normalize and predict
        imgWhite = imgWhite / 255.0  # Normalize pixel values to [0, 1]
        imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
        prediction = model.predict(imgWhite)
        index = np.argmax(prediction)  # Get the index of the highest probability class

        # Display the prediction on the image
        label = labels[index]
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x + w + offset, y - offset), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

    else:
        # Display "No Hands Detected" when no hands are visible
        cv2.putText(imgOutput, "No Hands Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the output
    cv2.imshow("Image", imgOutput)

    # Exit condition: press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
