from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json
from backend import dcm_to_json

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')

file_name=""


@app.route("/") 
def home():
	return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        if image.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(file_path)
            print(file_path)
            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result = dcm_to_json(file_path, weights, mimix_csv)
            print(result)
            os.remove(file_path)
            return result
        

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/fetchimage')
def fetch_image():
    return jsonify({"filename": "scaled_image.jpg"})


@app.route('/deleteimage')
def delete_image():
     print(current_dir)
     file_path = file_path = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
     os.remove(file_path)
     return {"true": True}


if __name__ == "__main__":
    app.run(debug=True)