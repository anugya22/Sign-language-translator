import cv2

camera = cv2.VideoCapture(0)  # Access the default camera

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to capture frame.")
        break

    cv2.imshow("Camera Test", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
