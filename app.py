import re
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from helper.lbp import lbp_histogram, save_feature, metric, load_feature_label, test_one

UPLOAD_FOLDER = "static/uploads"

app = Flask(__name__, static_url_path='/static')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/', methods=["GET", "POST"])
def main():
  if request.method == "GET":
    return render_template('login.html')
  elif request.method == "POST":
    if "image_file" not in request.files:
      return render_template('login.html', message='FALSE')
    
    image_file = request.files["image_file"]
    color_space = request.form.get('color_space')

    path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
    image_file.save(path)

    path_feature = os.path.join(app.config["UPLOAD_FOLDER"], "image_feature.npy")
    save_feature(path,0,path_feature,color_space)

    image_predict_proba, image_predict = test_one(path_feature, color_space)
    app.logger.info(image_predict_proba)
    app.logger.info(image_predict)
    if image_predict[0] == 1:
      message = "TRUE"
    else:
      message = "FALSE"

    return render_template('login.html', message=message, uploaded_image=path, image_predict_proba=image_predict_proba)