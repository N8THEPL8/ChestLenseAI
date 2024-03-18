from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, session
from backend_new import dcm_to_json, convert_dcm_to_jpg, run_with_no_csv, base64_to_image, byte_array_to_image
import os
from flask_sqlalchemy import SQLAlchemy
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

patientID = None

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
    s_comment = db.Column(db.String(100), nullable=True)
    s_name = db.Column(db.String(100), nullable=True)
    s_sex = db.Column(db.String(100), nullable=True)
    s_birthdate = db.Column(db.String(100), nullable=True)
    s_acqdate = db.Column(db.String(100), nullable=True)
    s_pos = db.Column(db.String(100), nullable=True)
    s_orientation = db.Column(db.String(100), nullable=True)
    s_age = db.Column(db.String(100), nullable=True)
    s_colour_ate = db.Column(db.LargeBinary, nullable=True)
    s_colour_car = db.Column(db.LargeBinary, nullable=True)
    s_colour_con = db.Column(db.LargeBinary, nullable=True)
    s_colour_ede = db.Column(db.LargeBinary, nullable=True)
    s_colour_eff = db.Column(db.LargeBinary, nullable=True)
    s_colour_nof = db.Column(db.LargeBinary, nullable=True)

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

@app.route('/comments', methods=['POST'])
def comments():
    data = request.json
    textarea_content = data.get('textarea_content')
    scan_id = data.get('scan_id')

    comment_updated = NewScan.query.filter_by(s_id=scan_id).update({'s_comment': textarea_content})
    db.session.commit()

    return jsonify({'comment': textarea_content})

@app.route('/index/<patient_id>')
def index(patient_id):
    global patientID
    patientID = patient_id
    patient = Patient.query.get(patient_id)
    if patient:
        scans = NewScan.query.filter_by(p_id=patient_id).all()
        return render_template('index.html', patient=patient, scans=scans, byte_array_to_image=byte_array_to_image)
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

@app.route("/upload-our-model", methods=['POST'])
def upload_our_model():
    print('our model')

    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        selected_scan_jpg = request.form['uploadedImages']

        if image.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(file_path)
            convert_dcm_to_jpg(file_path)
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
            result = run_with_no_csv(file_path2)

            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result2 = dcm_to_json(file_path, weights, mimix_csv)
            print(result2)

            with open(file_path2, 'rb') as file:
                jpg = file.read()
            result3 = json.loads(result2)

            comment = NewScan.query.filter_by(s_id=result3['Study_ID']).value(NewScan.s_comment)
            commentJSON = {'comment' : comment}
            dict1 = json.loads(result2)
            dict2 = json.loads(result)
            merged_dict = dict1.copy()
            merged_dict.update(dict2)
            merged_dict.update(commentJSON)

            existing_scan = NewScan.query.filter_by(s_id=result3['Study_ID']).first()
            if existing_scan is not None:
                pass # duplicate entry causes issues
            else:
                image_array_ate = merged_dict['diseasesData'][0]['gradCamImage']
                img_ate = base64_to_image(image_array_ate)
                img_ate.save('uploads/output_ate.jpg')
                with open('uploads/output_ate.jpg', 'rb') as file:
                    jpg_ate = file.read()
                
                image_array_car = merged_dict['diseasesData'][1]['gradCamImage']
                img_car = base64_to_image(image_array_car)
                img_car.save('uploads/output_car.jpg')
                with open('uploads/output_car.jpg', 'rb') as file:
                    jpg_car = file.read()
                
                image_array_con = merged_dict['diseasesData'][2]['gradCamImage']
                img_con = base64_to_image(image_array_con)
                img_con.save('uploads/output_con.jpg')
                with open('uploads/output_con.jpg', 'rb') as file:
                    jpg_con = file.read()
                
                image_array_ede = merged_dict['diseasesData'][3]['gradCamImage']
                img_ede = base64_to_image(image_array_ede)
                img_ede.save('uploads/output_ede.jpg')
                with open('uploads/output_ede.jpg', 'rb') as file:
                    jpg_ede = file.read()
                
                image_array_eff = merged_dict['diseasesData'][4]['gradCamImage']
                img_eff = base64_to_image(image_array_eff)
                img_eff.save('uploads/output_eff.jpg')
                with open('uploads/output_eff.jpg', 'rb') as file:
                    jpg_eff = file.read()
                
                image_array_nof = merged_dict['diseasesData'][5]['gradCamImage']
                img_nof = base64_to_image(image_array_nof)
                img_nof.save('uploads/output_nof.jpg')
                with open('uploads/output_nof.jpg', 'rb') as file:
                    jpg_nof = file.read()
                
                new_scan = NewScan(
                    s_id = result3['Study_ID'],
                    p_id = result3['Patient_ID'],
                    s_dicom = jpg,
                    s_name = result3['Patient_Name'],
                    s_sex = result3['Patient_Sex'],
                    s_birthdate = result3['Patient_Birth_Date'],
                    s_acqdate = result3['Acquisition_Date'],
                    s_pos = result3['View_Position'],
                    s_orientation = result3['Patient_Orientation'],
                    s_age = result3['Patient_Age_at_Time_of_Acquisition'],
                    s_colour_ate = jpg_ate,
                    s_colour_car = jpg_car,
                    s_colour_con = jpg_con,
                    s_colour_ede = jpg_ede,
                    s_colour_eff = jpg_eff,
                    s_colour_nof = jpg_nof
                )
                db.session.add(new_scan)
                db.session.commit()

            return json.dumps(merged_dict)
        
        else:
            selected_jpg , selected_scan_id = selected_scan_jpg.split('|', 1)

            #img_ate = base64_to_image(selected_jpg)
            #img_ate.save('uploads/image.jpg')

            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
            result = run_with_no_csv(file_path2)

            existing_scan = NewScan.query.filter_by(s_id=selected_scan_id).first()
            if existing_scan:
                scan_details = {
                    'Patient_Name': str(existing_scan.s_name),
                    'Patient_ID': str(existing_scan.p_id),
                    'Patient_Sex': str(existing_scan.s_sex),
                    'Patient_Birth_Date': str(existing_scan.s_birthdate),
                    'Acquisition_Date': str(existing_scan.s_acqdate),
                    'View_Position': str(existing_scan.s_pos),
                    'Patient_Orientation': str(existing_scan.s_orientation),
                    'Patient_Age_at_Time_of_Acquisition': str(existing_scan.s_age)
                }

                json_scan_details = json.dumps(scan_details, indent=4)

                dict1 = json.loads(json_scan_details)
                dict2 = json.loads(result)
                merged_dict = dict1.copy()
                merged_dict.update(dict2)

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