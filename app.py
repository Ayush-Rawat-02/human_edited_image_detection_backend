from flask import Flask,request,jsonify
import fake_image_detection
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

app=Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def sayHi():
    return "HI"

@app.route('/',methods=['POST'])
def predictImageValues():
    image_data = request.json['text']
    image_bytes = BytesIO(base64.b64decode(image_data))
    detection_results = fake_image_detection.detect_image(image_bytes)
    print(detection_results)
    return  jsonify(detection_results)

if __name__=='__main__':
    app.run(host="127.0.0.9", port=8080, debug=True)
