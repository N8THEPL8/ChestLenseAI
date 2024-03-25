from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, session
from backend_new import dcm_to_json, convert_dcm_to_jpg, run_with_no_csv
from backend_prebuilt import run_with_no_csv_prebuilt
import os
from flask_sqlalchemy import SQLAlchemy
import json
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = '192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'

patientID = None

current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://sql5688745:t11rn2ntwf@sql5.freesqldatabase.com:3306/sql5688745'
db = SQLAlchemy(app)

# Models for the database
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
    s_dicom = db.Column(db.LargeBinary, nullable=True)
    s_comment = db.Column(db.String(100), nullable=True)
    s_name = db.Column(db.String(100), nullable=True)
    s_sex = db.Column(db.String(100), nullable=True)
    s_birthdate = db.Column(db.String(100), nullable=True)
    s_acqdate = db.Column(db.String(100), nullable=True)
    s_pos = db.Column(db.String(100), nullable=True)
    s_age = db.Column(db.String(100), nullable=True)

# Checks for a valid username and password in the database
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        existing_doctor = Doctor.query.filter_by(d_username=request.form['username'], d_password=request.form['password']).first()
        if existing_doctor:
            session['d_id'] = existing_doctor.d_id
            return redirect(url_for('doctor'))
    return render_template('login.html')

# Determines which doctor is logged in and displays the corresponding patients
@app.route("/doctor") 
def doctor():
    if 'd_id' in session:
        doctor_id = session['d_id']
        patients = Patient.query.filter_by(d_id=doctor_id).all()
        return render_template("doctor.html", patients=patients)
    return redirect(url_for('login'))

# Updates the comment for a scan in the database
@app.route('/comments', methods=['POST'])
def comments():
    data = request.json
    textarea_content = data.get('textarea_content')
    scan_id = data.get('scan_id')
    comment_updated = NewScan.query.filter_by(s_id=scan_id).update({'s_comment': textarea_content})
    db.session.commit()
    return jsonify({'scan_ID' : scan_id, 'comment': textarea_content})

# Displays the scans for a patient
@app.route('/index/<patient_id>')
def index(patient_id):
    global patientID
    patientID = patient_id
    patient = Patient.query.get(patient_id)
    if patient:
        scans = NewScan.query.filter_by(p_id=patient_id).all()
        return render_template('index.html', patient=patient, scans=scans)
    return redirect(url_for('doctor'))

# Prebuilt model
@app.route("/upload", methods=['POST'])
def upload():
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        selected_scan_id = request.form['uploadedImages']

        # Uploads the image and runs the model
        if image.filename != '':
            # Save the dicom image and convert it to jpg
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(file_path)
            convert_dcm_to_jpg(file_path)

            # Run the model with the jpg image
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
            result = run_with_no_csv_prebuilt(file_path2)

            # Save the information of the scan to the database
            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result2 = dcm_to_json(file_path, weights, mimix_csv)

            with Image.open(file_path2) as img:
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='JPEG')
                jpg = byte_arr.getvalue()

            # Merge json results
            result3 = json.loads(result2)
            comment = NewScan.query.filter_by(s_id=result3['Study_ID']).value(NewScan.s_comment)
            commentJSON = {'comment' : comment}
            dict1 = json.loads(result2)
            dict2 = json.loads(result)
            merged_dict = dict1.copy()
            merged_dict.update(dict2)
            merged_dict.update(commentJSON)

            no_save = 'noSave' in request.form
            if no_save == False:
                existing_scan = NewScan.query.filter_by(s_id=result3['Study_ID']).first()
                if existing_scan is not None:
                    pass # duplicate entry causes issues
                else:
                    s_pos = ''
                    if result3['View_Position'] == 'PA':
                        s_pos = 'Posterior-Anterior'
                    elif result3['View_Position'] == 'AP':
                        s_pos = 'Anterior-Posterior'

                    new_scan = NewScan(
                        s_id = result3['Study_ID'],
                        p_id = result3['Patient_ID'],
                        s_dicom = jpg,
                        s_name = result3['Patient_Name'],
                        s_sex = result3['Patient_Sex'],
                        s_birthdate = result3['Patient_Birth_Date'],
                        s_acqdate = result3['Acquisition_Date'],
                        s_pos = s_pos if s_pos else result3['View_Position'],
                        s_age = result3['Patient_Age_at_Time_of_Acquisition'],
                        s_comment = ''
                    )
                    db.session.add(new_scan)
                    db.session.commit()

            return json.dumps(merged_dict)
        
        # Runs the model on an existing scan
        else:
            # Get the existing scan and run the model
            existing_scan = NewScan.query.filter_by(s_id=selected_scan_id).first()
            image_bytes = existing_scan.s_dicom
            image = Image.open(io.BytesIO(image_bytes))
            image.save('uploads/image.jpg')
            result = run_with_no_csv_prebuilt('uploads/image.jpg')

            if existing_scan:
                scan_details = {
                    'Patient_Name': str(existing_scan.s_name),
                    'Patient_ID': str(existing_scan.p_id),
                    'Patient_Sex': str(existing_scan.s_sex),
                    'Patient_Birth_Date': str(existing_scan.s_birthdate),
                    'Acquisition_Date': str(existing_scan.s_acqdate),
                    'View_Position': str(existing_scan.s_pos),
                    'Patient_Age_at_Time_of_Acquisition': str(existing_scan.s_age)
                }
                comment = NewScan.query.filter_by(s_id=selected_scan_id).value(NewScan.s_comment)
                commentJSON = {'comment' : comment}
                json_scan_details = json.dumps(scan_details, indent=4)
                dict1 = json.loads(json_scan_details)
                dict2 = json.loads(result)
                merged_dict = dict1.copy()
                merged_dict.update(dict2)
                merged_dict.update(commentJSON)

            return json.dumps(merged_dict)

# Our model
@app.route("/upload-our-model", methods=['POST'])
def upload_our_model():
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        selected_scan_id = request.form['uploadedImages']

        # Uploads the image and runs the model
        if image.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(file_path)
            convert_dcm_to_jpg(file_path)
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
            result = run_with_no_csv(file_path2)

            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result2 = dcm_to_json(file_path, weights, mimix_csv)

            with Image.open(file_path2) as img:
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='JPEG')
                jpg = byte_arr.getvalue()

            result3 = json.loads(result2)
            comment = NewScan.query.filter_by(s_id=result3['Study_ID']).value(NewScan.s_comment)
            commentJSON = {'comment' : comment}
            dict1 = json.loads(result2)
            dict2 = json.loads(result)
            merged_dict = dict1.copy()
            merged_dict.update(dict2)
            merged_dict.update(commentJSON)

            no_save = 'noSave' in request.form
            if no_save == False:
                existing_scan = NewScan.query.filter_by(s_id=result3['Study_ID']).first()
                if existing_scan is not None:
                    pass # duplicate entry causes issues
                else:
                    s_pos = ''
                    if result3['View_Position'] == 'PA':
                        s_pos = 'Posterior-Anterior'
                    elif result3['View_Position'] == 'AP':
                        s_pos = 'Anterior-Posterior'

                    new_scan = NewScan(
                        s_id = result3['Study_ID'],
                        p_id = result3['Patient_ID'],
                        s_dicom = jpg,
                        s_name = result3['Patient_Name'],
                        s_sex = result3['Patient_Sex'],
                        s_birthdate = result3['Patient_Birth_Date'],
                        s_acqdate = result3['Acquisition_Date'],
                        s_pos = s_pos if s_pos else result3['View_Position'],
                        s_age = result3['Patient_Age_at_Time_of_Acquisition'],
                        s_comment = ''
                    )
                    db.session.add(new_scan)
                    db.session.commit()

            return json.dumps(merged_dict)
        
        # Runs the model on an existing scan
        else:
            existing_scan = NewScan.query.filter_by(s_id=selected_scan_id).first()
            image_bytes = existing_scan.s_dicom
            image = Image.open(io.BytesIO(image_bytes))
            image.save('uploads/image.jpg')
            result = run_with_no_csv('uploads/image.jpg')

            if existing_scan:
                scan_details = {
                    'Patient_Name': str(existing_scan.s_name),
                    'Patient_ID': str(existing_scan.p_id),
                    'Patient_Sex': str(existing_scan.s_sex),
                    'Patient_Birth_Date': str(existing_scan.s_birthdate),
                    'Acquisition_Date': str(existing_scan.s_acqdate),
                    'View_Position': str(existing_scan.s_pos),
                    'Patient_Age_at_Time_of_Acquisition': str(existing_scan.s_age)
                }
                comment = NewScan.query.filter_by(s_id=selected_scan_id).value(NewScan.s_comment)
                commentJSON = {'comment' : comment}
                json_scan_details = json.dumps(scan_details, indent=4)
                dict1 = json.loads(json_scan_details)
                dict2 = json.loads(result)
                merged_dict = dict1.copy()
                merged_dict.update(dict2)
                merged_dict.update(commentJSON)

            return json.dumps(merged_dict)

@app.route('/index/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/fetchimage')
def fetch_image():
    return jsonify({"filename": "image.jpg"})

@app.route('/deleteimage')
def delete_image():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
    # os.remove(file_path)
    return {"true": True}

if __name__ == "__main__":
    app.run(host='localhost', port=9874, debug=True)