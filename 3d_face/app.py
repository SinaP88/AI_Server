from flask import Flask, request, render_template, jsonify, send_from_directory
import io
import os
import base64
from datetime import datetime
from PIL import Image
import logging
logging.basicConfig(level=logging.INFO)
from logging.config import dictConfig
import sys
import yaml
from model.TDDFA import TDDFA
from model.FaceBoxes import FaceBoxes
from model.utils.serialization import ser_to_ply
import numpy as np
import json

# logger configuration
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

### Starting Flask Server ###

app = Flask(__name__)
logging.info("starting server")
print("staring server")


class NumpyArrayEncoder(json.JSONEncoder):
    """ Custom JSON Encoder to Serialize NumPy nd.array """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def predict_ply(img):
    cfg = yaml.load(open('model/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    tddfa = TDDFA(gpu_mode=False, **cfg)
    face_boxes = FaceBoxes()
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        logging.info('No face detected, exit')
        sys.exit(-1)
    logging.info(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)
    # print(f'params: {param_lst}')
    # print(f'rois: {roi_box_lst}')
    flag = 'ply'
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=flag)
    lm_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    landmark_list = [[row[i] for row in lm_lst[0]] for i in range(len(lm_lst[0][0]))]
    # print(f'len landmarks: {len(landmark_list)}')
    # print(f'len lm_lst: {len(ver_lst[0][0])}')
    # print(f'len ver_lst: {len(ver_lst[0][0])}')


    wfp = ser_to_ply(img, ver_lst, tddfa.tri, height=img.shape[0])
    logging.info(f'first 1000 string of ply file:    {wfp.getvalue()[:500]} ...')
    return wfp, landmark_list


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


@app.route("/predict3d", methods=['POST'])
def predict3D():
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
    ply_3d, landmark_list = predict_ply(img_arr)
    logging.info("prepare to return 3D face ...")

    string_data = ply_3d.getvalue()
    # transImg_str = base64.b64encode(byte_data).decode()
    # print(f'String ply file : {string_data[:500]}')
    landmark_list = np.array(landmark_list)
    result_dict = {
                   'face3d_ply': string_data,
                   'landmark_3d': landmark_list
                   }
    # return result_dict
    # return jsonify(result_dict)
    return json.dumps(result_dict, cls=NumpyArrayEncoder)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='3000')
    # app.run()
    # img = cv2.imread('static/img.jpg', cv2.IMREAD_COLOR)
    # predict_ply(img)

