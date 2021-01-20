from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
from emotion_recognition import inference
import base64
from PIL import Image
import io
 


app = Flask(__name__)

@app.route('/index', methods=['POST'])
def test():
    im_b64 = request.json['image']
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    img = Image.open(io.BytesIO(img_bytes))
    img_arr = np.asarray(img)      

    threshold = request.json['threshold']
    file_name = request.json['file_name']
    inference(img_arr, threshold, file_name)
    response = {'message': 'image received'}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)