from flask import Flask, request, render_template, jsonify, send_from_directory
import pickle
import io
import json
import os
import base64
from datetime import datetime
from PIL import Image
from FaceAligner import *
from transparentMouthVisualization import *
import dlib
import logging
import tensorflow as tf
logging.basicConfig(level=logging.DEBUG)
detector = pickle.load(open('models/detector.pkl', 'rb'))
predictor = pickle.load(open('models/predictor.pkl', 'rb'))
mouth_detector = tf.keras.models.load_model("models/mouth_detector")
mouth_detector.load_weights("models/mouth_detector_weights.h5")
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


def face_aligner(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Image loaded ...")
    except:
        logging.info("Image format unreadable!")
    logging.info("cv2 image loaded")
    # face detection
    faces = detector(gray, 1)
    (x, y, w, h) = face_utils.rect_to_bb(faces[0])
    # originalDetectedFace = image[y:y + h, x:x + w]

    # face alignment
    fa = FaceAligner(predictor)
    original_points, affine_transformation = fa.align(image, gray, faces[0])

    # cut out mouth mask prediction
    reverse_mask = predict_mouth(mouth_detector, image, x, y, w, h)

    # grayFaceAligned = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)
    # (x2, y2, w2, h2) = (0, 0, grayFaceAligned.shape[0], grayFaceAligned.shape[1])
    # cords = dlib.rectangle(x2, y2, w2, h2)
    # shape = predictor(grayFaceAligned, cords)
    # points = face_utils.shape_to_np(shape)
    # logging.info("image analysis done!")
    logging.info("image analysis done!")
    return original_points, reverse_mask, affine_transformation


### Starting Flask Server ###

app = Flask(__name__)
logging.info("starting server")
print("staring server")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'images/favicon.ico')

@app.route("/ping", methods=['GET'])
def ping():
    now = datetime.now().strftime("%H:%M:%S")
    logging.info(f"time of ping arrival {now}")
    # return (f"Hello! time: {now}")
    return render_template('index.html', data=f"Hello! ping time: {now}")

@app.route("/predict", methods=['POST'])
def test_method():
    logging.info("request received")

    # print("Post keys: ", request.json.keys())
    if not request.form or 'image' not in request.form:
        logging.info(400)
    else:
        logging.info(f"received image form: {request.form['image'][:100]} ...")
    logging.info(f"dict keys are: {request.form.to_dict().keys()}")

    # get the base64 encoded string
    logging.info("start image decoding ...")

    try:
        im_b64 = request.form["image"]
        logging.info(f"Received image is:  {im_b64[:100]} ...")
    except ValueError:
        logging.info("Oops! That was no valid image data.  Try again...")

    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)
    logging.info("image decoded successfully!")

    # calculating results
    original_points, mask, affine_matrix = face_aligner(img_arr)
    logging.info("prepare to return analysis ...")

    transparent_image = transparentImage(img_arr, mask)
    logging.info("prepare to return transparent image ...")

    # # converting numpy array to base64 string
    # _, alignedImgBuffer = cv2.imencode('.png', alignedFace)
    # alignedImg = (base64.b64encode(alignedImgBuffer)).decode()
    buffered = io.BytesIO()
    transparent_image.save(buffered, format="PNG")
    byte_data = buffered.getvalue()
    transImg_str = base64.b64encode(byte_data).decode()

    result_dict = {# 70 detected face points
                   # 'original_points': json.dumps(original_points.tolist()),
                   'original_points': original_points.tolist(),
                   # pupil aligned face Affine Matrix
                   # 'affine_matrix': json.dumps(affine_matrix.tolist()),
                   'affine_matrix': affine_matrix.tolist(),
                   # cut out mouth
                   'transparentMouth': transImg_str
                   }
    # return result_dict
    return jsonify(result_dict)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='5000')
