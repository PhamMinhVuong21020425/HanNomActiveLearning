# This file is part of https://github.com/jainamoswal/Flask-Example.
# Usage covered in <IDC lICENSE>
# Jainam Oswal. <jainam.me> 


# Import Libraries 
import os
import time
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin 
from pyngrok import ngrok
from dotenv import load_dotenv
from firebase import db
from predict import Predictor

load_dotenv()
predictor = Predictor()

# Define app.
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Config CORS.
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["SECRET_KEY"] = os.environ.get('SECRET_KEY')

# Define root route.
@app.route("/", methods=['GET'])
@cross_origin() # Allow Cross-Origin Resource Sharing
def home():
    return jsonify({
        "data": "We support Han Nom image detection and recognition.",
    })

@app.route("/test", methods=['GET'])
@cross_origin()
def test():
    source = './yolov5/data/images/test4.png'
    json_objects = predictor.predict(source, classify=False)
    return jsonify(json_objects)


@app.route("/api/detect", methods=['POST'])
@cross_origin()
def object_detection():
    start = time.time()
    files = request.files.getlist("files")

    output = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, f'temp.{file.filename.split(".")[-1]}')
        file.save(file_path)

        json_objects = predictor.predict(source=file_path, classify=True)
        output.append({
            'objects_detection': json_objects,
            'url_image': os.path.join(UPLOAD_FOLDER, file.filename),
            'image_name': file.filename
        })

        # Remove image after processing
        os.remove(file_path)

    end = time.time()
    print(f"Time taken: {end - start:.3f}s")
    return jsonify(output)

@app.route('/favicon.ico')
def favicon():
    path = os.path.join(app.root_path, 'static')
    return send_from_directory(path, 'favicon.ico')

ngrok.set_auth_token(os.environ.get('NGROK_AUTH_TOKEN'))
url = ngrok.connect(os.environ.get('PORT', 5000)).public_url
db.update({"server_url": url})
print('Global NGROK URL:', url)
