from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS, cross_origin
from waitress import serve
import keras
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

from keras.models import model_from_json
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

model = FacialExpressionModel("model.json", "model_weights.h5")

@app.route('/')
def index():
    return render_template('index.html')

#request from springboot app
@app.route('/test', methods=['POST'])
def testing():
    data = request.get_data()
    #to remove this section of the request body (data:image/png;base64,)
    b = base64.b64decode(data[22:])
    # use cv library to decode the base 64 encoded string and convert to single channel grayscale image
    file_bytes = np.asarray(bytearray(b), dtype =np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = detector.detectMultiScale(gray_fr, 1.3, 3)
    if len(faces) == 0:
        print("Please show your face clearly")
        a = {
            "prediction" : "Unable to make a prediction on your expression. Take note to show your face clearly when capturing photo on the web screen. Please try again."}
        return jsonify(a)
    else:
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            print(pred)
            a = {
            "prediction" : pred}
        return jsonify(a)

#request from android app
@app.route('/android', methods=['POST'])
def handle_request():
    encoded = request.get_data().decode('utf-8')
    decoded = base64.b64decode(encoded)
    # use cv library to decode the base 64 encoded string and convert to single channel grayscale image
    file_bytes = np.asarray(bytearray(decoded), dtype =np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = detector.detectMultiScale(gray_fr, 1.3, 3)
    if len(faces) == 0:
        return "Unable to make a prediction on your expression. Take note to show your face clearly when capturing photo on the web screen. Please try again."
    else:
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            print(pred)
        return pred

if __name__ == '__main__':
    print("Starting the server.....")
    serve(app, host="0.0.0.0", port=8080)