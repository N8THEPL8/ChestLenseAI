from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, session
from backend import dcm_to_json
import os
from flask_sqlalchemy import SQLAlchemy
import json

# import pydicom
# from pydicom.filebase import DicomBytesIO
# def dicom_to_binary(file_path):
#     dicom_image = pydicom.dcmread(file_path)
#     dicom_byte_arr = DicomBytesIO()
#     dicom_image.save_as(dicom_byte_arr, write_like_original=True)
#     return dicom_byte_arr.parent.getvalue()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://sql5688745:t11rn2ntwf@sql5.freesqldatabase.com:3306/sql5688745'
db = SQLAlchemy(app)

class Doctor(db.Model):
    __tablename__ = 'doctor_table'
    d_id = db.Column(db.String(100), primary_key=True)
    d_username = db.Column(db.String(100), nullable=False)
    d_password = db.Column(db.String(100), nullable=False)

class Patient(db.Model):
    __tablename__ = 'patient_table'
    p_id = db.Column(db.String(100), primary_key=True)
    d_id = db.Column(db.String(100), db.ForeignKey('doctor_table.d_id'))
    p_name = db.Column(db.String(100), nullable=False)
    p_sex = db.Column(db.String(100), nullable=False)
    p_birthdate = db.Column(db.String(100), nullable=False)

class NewScan(db.Model):
    __tablename__ = 'new_scan_table'
    s_id = db.Column(db.String(100), primary_key=True)
    p_id = db.Column(db.String(100), db.ForeignKey('patient_table.p_id'))
    # s_dicom = db.Column(db.LargeBinary, nullable=False)
    s_comment = db.Column(db.String(100), nullable=False)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        existing_doctor = Doctor.query.filter_by(d_username=request.form['username'], d_password=request.form['password']).first()
        if existing_doctor:
            session['d_id'] = existing_doctor.d_id
            return redirect(url_for('doctor'))
    return render_template('login.html')

@app.route("/doctor") 
def doctor():
    if 'd_id' in session:
        doctor_id = session['d_id']
        patients = Patient.query.filter_by(d_id=doctor_id).all()
        return render_template("doctor.html", patients=patients)
    return redirect(url_for('login'))

@app.route('/index/<patient_id>')
def index(patient_id):
    patient = Patient.query.get(patient_id)
    if patient:
        scans = NewScan.query.filter_by(p_id=patient_id).all()
        print(scans)
        return render_template('index.html', patient=patient, scans=scans)
    return redirect(url_for('doctor'))

@app.route("/upload", methods=['POST'])
def upload():
    print('prebuilt')
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        if image.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            # binary_data = dicom_to_binary(file_path)
            image.save(file_path)
            # with open(file_path, 'rb') as file:
            #     image_data = file.read()
            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result = dcm_to_json(file_path, weights, mimix_csv)
            os.remove(file_path)
            result2 = json.loads(result)
            print(result2)
            print(file_path)
            existing_scan = NewScan.query.filter_by(s_id=result2['Study_ID']).first()
            if existing_scan is not None:
                pass # duplicate entry causes issues
            else:
                new_scan = NewScan(
                    s_id=result2['Study_ID'],
                    p_id=result2['Patient_ID'],
                    s_comment='Mar 11, 9:46.'
                )
                db.session.add(new_scan)
                db.session.commit()
            return result

# change this once we have our model
@app.route("/upload-our-model", methods=['POST'])
def upload_our_model():
    print('our model')
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        if image.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            # binary_data = dicom_to_binary(file_path)
            image.save(file_path)
            # with open(file_path, 'rb') as file:
            #     image_data = file.read()
            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result = dcm_to_json(file_path, weights, mimix_csv)
            os.remove(file_path)
            result2 = json.loads(result)
            print(result2)
            print(file_path)
            existing_scan = NewScan.query.filter_by(s_id=result2['Study_ID']).first()
            if existing_scan is not None:
                pass # duplicate entry causes issues
            else:
                new_scan = NewScan(
                    s_id=result2['Study_ID'],
                    p_id=result2['Patient_ID'],
                    s_comment='Mar 11, 9:46'
                )
                db.session.add(new_scan)
                db.session.commit()
            return result

@app.route('/index/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/fetchimage')
def fetch_image():
    return jsonify({"filename": "scaled_image.jpg"})

@app.route('/deleteimage')
def delete_image():
    file_path = file_path = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
    os.remove(file_path)
    return {"true": True}

if __name__ == "__main__":
    app.run(host='localhost', port=9874, debug=True)