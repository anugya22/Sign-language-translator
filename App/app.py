from flask import Flask, render_template, Response, request, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math
import threading
import sqlite3

app = Flask(__name__)

# Initialize components
detector = HandDetector(maxHands=1)
model = load_model("Model/keras_model.keras")
offset = 20
imgSize = 300

# Load labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f]

camera = None
streaming_thread = None
is_streaming = False
camera_lock = threading.Lock()

# Initialize SQLite Database
db_name = "chat_history.db"


def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def initialize_camera():
    """Initialize the camera safely."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Error: Could not initialize the camera.")
                return False
    return True


def release_camera():
    """Release the camera safely."""
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None


def save_message_to_db(username, message):
    """Save recognized gesture as a chat message."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (username, message) VALUES (?, ?)", (username, message))
    conn.commit()
    conn.close()


@app.route('/save_message', methods=['POST'])
def save_message():
    """Save a message to the database."""
    data = request.get_json()
    username = data.get('username')
    message = data.get('message')
    save_message_to_db(username, message)
    return jsonify({'status': 'success'})


@app.route('/get_history', methods=['GET'])
def get_history():
    """Retrieve chat history for a user."""
    username = request.args.get('username')
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT message FROM chat_history WHERE username=?", (username,))
    messages = cursor.fetchall()
    conn.close()
    return jsonify({'messages': [msg[0] for msg in messages]})


def generate_frames(username):
    """Generate video frames with real-time gesture recognition."""
    global is_streaming, camera

    if not initialize_camera():
        return

    while is_streaming:
        with camera_lock:
            success, img = camera.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img, flipType=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            if y - offset < 0 or x - offset < 0 or y + h + offset > img.shape[0] or x + w + offset > img.shape[1]:
                cv2.putText(imgOutput, "Hand partially outside frame", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

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

                imgWhite = imgWhite / 255.0
                imgWhite = np.expand_dims(imgWhite, axis=0)
                prediction = model.predict(imgWhite)
                index = np.argmax(prediction)

                label = labels[index]
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x + w + offset, y - offset), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, label, (x, y - offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

                # Save prediction to database
                save_message_to_db(username, label)

        else:
            cv2.putText(imgOutput, "No Hands Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    username = request.args.get('username')
    global is_streaming, streaming_thread

    if is_streaming:
        return "Stream is already running.", 200

    is_streaming = True
    return Response(generate_frames(username), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_feed', methods=['GET'])
def stop_feed():
    global is_streaming
    is_streaming = False
    release_camera()
    return '', 204


if __name__ == '__main__':
    app.run(debug=True)
