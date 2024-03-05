from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for
from backend import dcm_to_json
import os
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://sql5688745:t11rn2ntwf@sql5.freesqldatabase.com:3306/sql5688745'
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'test_doctor'  # Name of the existing table in the database
    username = db.Column(db.String(100), primary_key=True)
    password = db.Column(db.String(100), nullable=False)


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        existing_user = User.query.filter_by(username=request.form['username'], password=request.form['password']).first()

        if existing_user:
             return redirect(url_for('doctor'))
        else:
             error = 'NOT WORKING YUVI'

        #return render_template('login.html', error=error)
        #if request.form['username'] != '18group' or request.form['password'] != '18group':
         #   error = 'Invalid Credentials. Please try again.'
        #else:
         #   return redirect(url_for('doctor'))
    return render_template('login.html', error=error)

@app.route("/doctor") 
def doctor():
	return render_template("doctor.html")

@app.route("/index") 
def index():
	return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    if 'xrayImage' in request.files:
        image = request.files['xrayImage']
        if image.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(file_path)
            weights = "densenet121-res224-mimic_ch"
            mimix_csv = "mimic-cxr-2.0.0-chexpert.csv"
            result = dcm_to_json(file_path, weights, mimix_csv)
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
    file_path = file_path = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
    os.remove(file_path)
    return {"true": True}

if __name__ == "__main__":
    app.run(host='localhost', port=9874, debug=True)