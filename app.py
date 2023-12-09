from flask import Flask,request,jsonify
import fake_image_detection
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
app=Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/')
def index():
    response=jsonify({'caption': ''})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
@app.route('/',methods=['POST'])
def caption():
    if request.method == 'POST':
        image_data = request.json['text']
        image_bytes = BytesIO(base64.b64decode(image_data))
        detection_results = fake_image_detection.detect_image(image_bytes) # utility to extract features from image and guess its caption
        print(detection_results)
        response=jsonify(detection_results)
        response.headers.add('Access-Control-Allow-Origin', '*')
    return  response

if __name__=='__main__':
    app.run(debug=True) #to start flask application
