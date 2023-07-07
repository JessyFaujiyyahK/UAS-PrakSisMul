from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('model_emotion.h5')
emotion_labels = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_roi = cv2.resize(face_roi, (48, 48))
        normalized_roi = resized_roi / 255.0
        reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))

        emotion_prediction = emotion_model.predict(reshaped_roi)
        max_index = np.argmax(emotion_prediction[0])
        emotion_label = emotion_labels[max_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def generate_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        processed_frame = detect_emotion(frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
