from flask import Flask, jsonify, request
from Apple_yoloV5 import ObjectDetection
from datetime import datetime
from PIL import Image
import io
import time
import base64
import json
import cv2
import numpy as np
import os

app = Flask(__name__)


##model = Work_Model()
##model.load_network()

detection = ObjectDetection()

class Myclass:
    b64_string = ""
    cords = 0
    boolll = False


def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

@app.route("/model/predict", methods=["GET", "POST"])
def predict():
    data = json.loads(request.data)
    #img = data.get("imageString")
    img = data
    imgFromcs = stringToImage(img)
    numpyImage = np.array(imgFromcs)
    print("Image converted to numpy successfuly orinf server")
    start=time.time()
    openCvImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2BGR)
    results = detection.score_frame(openCvImage)
    cords = detection.plot_boxes(results, openCvImage)
    print(jsonify(cords))
    return jsonify(cords)



@app.route("/server", methods=["GET", "POST"])
def server():
    return "running"


if __name__ == "__main__":
    app.run(port=5030, use_reloader=False, debug=True, threaded=True)
