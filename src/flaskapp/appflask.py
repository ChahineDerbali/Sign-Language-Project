import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response, jsonify
import threading
import time

app = Flask(__name__)

model = load_model(
    "C:/Users/CHPde/OneDrive/Bureau/Projects/Sign Language Project/ADS-Project/models/sign_language_model_cropped100.h5"
)
class_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space",
]

written_word = []
frame_skip = 5
frame_counter = 0
cap = cv2.VideoCapture(0)
latest_prediction = {"letter": ""}


def capture_video():
    global frame_counter, latest_prediction
    ret, frame = cap.read()
    if not ret:
        return None
    frame_counter += 1
    if frame_counter % frame_skip != 0:
        return None
    height, width, _ = frame.shape
    min_side = min(height, width)
    center_x, center_y = width // 2, height // 2
    cropped_frame = frame[
        center_y - min_side // 2 : center_y + min_side // 2,
        center_x - min_side // 2 : center_x + min_side // 2,
    ]
    resized_frame = cv2.resize(cropped_frame, (100, 100))
    normalized_frame = resized_frame / 255.0
    uint8_frame = np.uint8(normalized_frame * 255)
    rgb_frame = cv2.cvtColor(uint8_frame, cv2.COLOR_BGR2RGB)
    model_input = np.expand_dims(rgb_frame, axis=0)
    prediction = model.predict(model_input)
    predicted_class = np.argmax(prediction)
    predicted_letter = class_labels[predicted_class]
    latest_prediction["letter"] = predicted_letter
    cv2.rectangle(
        frame,
        (center_x - min_side // 2, center_y - min_side // 2),
        (center_x + min_side // 2, center_y + min_side // 2),
        (0, 255, 0),
        2,
    )
    ret, jpeg = cv2.imencode(".jpg", frame)
    if ret:
        frame_data = jpeg.tobytes()
        return frame_data
    return None


def generate():
    while True:
        frame = capture_video()
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/add_letter/<letter>")
def add_letter(letter):
    global written_word
    if letter == "del":
        if written_word:
            written_word.pop()
    elif letter == "space":
        written_word.append(" ")
    else:
        written_word.append(letter)
    return jsonify({"word": "".join(written_word)})


@app.route("/current_prediction")
def current_prediction():
    return jsonify(latest_prediction)


@app.route("/add_letter_confirm")
def add_letter_confirm():
    global written_word
    letter = latest_prediction["letter"]
    if letter == "del":
        if written_word:
            written_word.pop()
    elif letter == "space":
        written_word.append(" ")
    elif letter and letter not in ["nothing"]:
        written_word.append(letter)
    return jsonify({"word": "".join(written_word)})


import atexit


@atexit.register
def release_camera():
    if cap.isOpened():
        cap.release()


if __name__ == "__main__":
    app.run(debug=True)
