from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, session
from backend import dcm_to_json
from backend_new import test_single_image_no_csv, convert_dcm_to_jpg, image_to_base64, run_with_no_csv, thresholds, model, device
import os
from flask_sqlalchemy import SQLAlchemy
import json

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

# not done
class NewScan(db.Model):
    __tablename__ = 'new_scan_table'
    s_id = db.Column(db.String(100), primary_key=True)
    p_id = db.Column(db.String(100), db.ForeignKey('patient_table.p_id'))
    s_dicom = db.Column(db.LargeBinary, nullable=True)
    # s_colorjpg = db.Column(db.LargeBinary, nullable=True)
    # s_comment = db.Column(db.String(100), nullable=True)
    # s_name = db.Column(db.String(100), nullable=True)
    # s_sex = db.Column(db.String(100), nullable=True)
    # s_birthdate = db.Column(db.String(100), nullable=True)
    # s_acqdate = db.Column(db.String(100), nullable=True)
    # s_pos = db.Column(db.String(100), nullable=True)
    # s_orientation = db.Column(db.String(100), nullable=True)
    # s_age = db.Column(db.String(100), nullable=True)

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
        return render_template('index.html', patient=patient, scans=scans)
    return redirect(url_for('doctor'))

# change this once we have prebuilt
@app.route("/upload", methods=['POST'])
def upload():
    print('prebuilt')
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        if image.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(file_path)
            print(file_path) # /Users/ant.vu/Developer/ai-for-chest-x-ray/src/client/uploads/demo1.dcm
            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result = dcm_to_json(file_path, weights, mimix_csv)
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'scaled_image.jpg')
            with open(file_path2, 'rb') as file:
                jpg = file.read()
            # os.remove(file_path)
            result2 = json.loads(result)
            existing_scan = NewScan.query.filter_by(s_id=result2['Study_ID']).first()
            if existing_scan is not None:
                pass # duplicate entry causes issues
            else:
                new_scan = NewScan(
                    s_id=result2['Study_ID'],
                    p_id=result2['Patient_ID'],
                    s_dicom=jpg
                    # s_name=result2['Patient_Name'],
                    # s_sex = result2['Patient_Sex'],
                    # s_birthdate = result2['Patient_Birth_Date'],
                    # s_acqdate = result2['Acquisition_Date'],
                    # s_pos = result2['View_Position'],
                    # s_orientation = result2['Patient_Orientation'],
                    # s_age = result2['Patient_Age_at_Time_of_Acquisition']
                )
                db.session.add(new_scan)
                db.session.commit()
            return result

#temporarily trying new backend
@app.route("/upload-our-model", methods=['POST'])
def upload_our_model():
    print('our model')
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        if image.filename != '':
            # #old way
            # file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            # image.save(file_path)
            # print(file_path) # /Users/ant.vu/Developer/ai-for-chest-x-ray/src/client/uploads/demo1.dcm
            # weights = "densenet121-res224-mimic_ch"
            # mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            # result = dcm_to_json(file_path, weights, mimix_csv)
            # file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'scaled_image.jpg')
            # with open(file_path2, 'rb') as file:
            #     jpg = file.read()
            # # os.remove(file_path)
            # result2 = json.loads(result)
            # existing_scan = NewScan.query.filter_by(s_id=result2['Study_ID']).first()
            # if existing_scan is not None:
            #     pass # duplicate entry causes issues
            # else:
            #     new_scan = NewScan(
            #         s_id=result2['Study_ID'],
            #         p_id=result2['Patient_ID'],
            #         s_dicom=jpg
            #         # s_name=result2['Patient_Name'],
            #         # s_sex = result2['Patient_Sex'],
            #         # s_birthdate = result2['Patient_Birth_Date'],
            #         # s_acqdate = result2['Acquisition_Date'],
            #         # s_pos = result2['View_Position'],
            #         # s_orientation = result2['Patient_Orientation'],
            #         # s_age = result2['Patient_Age_at_Time_of_Acquisition']
            #     )
            #     db.session.add(new_scan)
            #     db.session.commit()
            
            # new way
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(file_path)
            convert_dcm_to_jpg(file_path)
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], "scaled_image.jpg")
            image.save(file_path2)
            run_with_no_csv("file_path2")

            # filepath = "/Users/ant.vu/Developer/ai-for-chest-x-ray/src/client/uploads/8c0171a3-925313ff-f63faed5-3007b5ad-d1bbb676.jpg"
            # csv_file_path = "/Users/ant.vu/Developer/ai-for-chest-x-ray/src/client/Validation_Partial.csv"
            # thresholds = [0.53880334, 0.48418066, 0.36754248, 0.5815063, 0.54026645, 0.47844747]
            # print('helllo')
            # original_image, grad_cam_image = test_single_image(filepath, csv_file_path, thresholds, model, device)
            # print(original_image)
            # print(grad_cam_image[0])
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
    # os.remove(file_path)
    return {"true": True}

if __name__ == "__main__":
    app.run(host='localhost', port=9874, debug=True)