import torchxrayvision
from pandas import Index
import torchxrayvision as xrv
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
import pydicom
import cv2 as cv
from PIL import Image
from datetime import datetime
import json
from flask import Flask

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')


def calculate_age(birth_date, current_date):
    # Calculation
    years = current_date.year - birth_date.year
    months = current_date.month - birth_date.month
    days = current_date.day - birth_date.day

    # Adjust for negative differences
    if days < 0:
        months -= 1
        days += get_days_in_month(birth_date.month, birth_date.year)
    if months < 0:
        years -= 1
        months += 12

    return years, months, days

def get_days_in_month(month, year):
    # Returns the number of days in a given month and year
    if month == 2:  # February
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return 29  # Leap year
        else:
            return 28
    elif month in [4, 6, 9, 11]:  # April, June, September, November
        return 30
    else:
        return 31

def dcm_patient_data(path_file):
    dcm_data = pydicom.dcmread(path_file)
    patient_data = {}
    have_patient_dob = False
    have_acquisition_date = False
    patient_birth_date = ""
    patient_age_now = ""
    acquisition_date = ""
    acquisition_age = ""

    patient_data.update({"Study_ID": str(dcm_data.StudyID)})
    patient_data.update({"Patient_ID": str(dcm_data.PatientID)})
    patient_data.update({"Patient_Name": str(dcm_data.PatientName)})
    patient_data.update({"Patient_Sex": str(dcm_data.PatientSex)})

    if dcm_data.PatientBirthDate == "":
        patient_birth_date = ""
        patient_age_now = ""
    else:
        have_patient_dob = True
        birth_date = datetime.utcfromtimestamp(int(dcm_data.PatientBirthDate))
        if birth_date <= datetime.now():
            age_years, age_months, age_days = calculate_age(birth_date, datetime.now())
            patient_age_now = f"{age_years} years, {age_months} months, {age_days} days"
            patient_birth_date = birth_date.strftime('%Y-%m-%d')

    if dcm_data.AcquisitionDate == "":
        acquisition_date = ""
        acquisition_age = ""
    else:
        have_acquisition_date = True
        acquisition_date = datetime.utcfromtimestamp(int(dcm_data.AcquisitionDate))
        patient_acquisition_date = acquisition_date.strftime('%Y-%m-%d')

    if have_patient_dob and have_acquisition_date:

        if birth_date <= acquisition_date:
            age_years, age_months, age_days = calculate_age(birth_date, acquisition_date)
            acquisition_age = f"{age_years} years, {age_months} months, {age_days} days"

    patient_data.update({"Patient_Birth_Date": str(patient_birth_date)})
    patient_data.update({"Patient_Current_Age": str(patient_age_now)})
    patient_data.update({"Series_Number": str(dcm_data.SeriesNumber)})
    patient_data.update({"Acquisition_Number": str(dcm_data.AcquisitionNumber)})
    patient_data.update({"Acquisition_Date": str(patient_acquisition_date)})
    patient_data.update({"Instance_Number": str(dcm_data.InstanceNumber)})
    patient_data.update({"View_Position": str(dcm_data.ViewPosition)})
    patient_data.update({"Patient_Orientation": str(dcm_data.PatientOrientation)})
    patient_data.update({"Patient_Age_at_Time_of_Acquisition": str(acquisition_age)})

    return patient_data

def run_model(file_path, weight_string):
    img = skimage.io.imread(file_path)
    img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...]  # Make single color channel

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    img = transform(img)
    img = torch.from_numpy(img)

    model = xrv.models.DenseNet(weights=weight_string)
    outputs = model(img[None, ...])
    probability_dictionary = dict(zip(model.pathologies, outputs[0].detach().numpy()))

    model_dict = {}

    model_dict.update({"Model_Atelectasis": round(float(probability_dictionary['Atelectasis'])*100, 2)})
    model_dict.update({"Model_Cardiomegaly": round(float(probability_dictionary['Cardiomegaly'])*100, 2)})
    model_dict.update({"Model_Consolidation": round(float(probability_dictionary['Consolidation'])*100, 2)})
    model_dict.update({"Model_Edema": round(float(probability_dictionary['Edema'])*100, 2)})
    model_dict.update({"Model_Effusion": round(float(probability_dictionary['Effusion'])*100, 2)})

    return model_dict

def convert_dcm_to_jpg(file_path):
    ds = pydicom.dcmread(file_path)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
    final_image.save(file_path)

    image = cv.imread(file_path)
    RGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    new_image = Image.fromarray(RGB)
    new_image = new_image.resize((1024, 1024))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "scaled_image.jpg")
    new_image.save(file_path)
    return

def actual_dcm_dict(file_path, study_id):
    line_content = []

    with open(file_path) as myFile:
        for num, line in enumerate(myFile, 0):
            if study_id in line:
                line_content = line.split(",")

    Atelectasis_index = 2
    Cardiomegaly_index = 3
    Consolidation_index = 4
    Edema_index = 5
    Pleural_Effusion_index = 11

    actual_dictionary = {
        'Actual_Atelectasis': 0,
        'Actual_Cardiomegaly': 0,
        'Actual_Consolidation': 0,
        'Actual_Edema': 0,
        'Actual_Effusion': 0
    }

    if line_content[Atelectasis_index] == "1.0":
        actual_dictionary['Actual_Atelectasis'] = 1
    if line_content[Cardiomegaly_index] == "1.0":
        actual_dictionary['Actual_Cardiomegaly'] = 1
    if line_content[Consolidation_index] == "1.0":
        actual_dictionary['Actual_Consolidation'] = 1
    if line_content[Edema_index] == "1.0":
        actual_dictionary['Actual_Edema'] = 1
    if line_content[Pleural_Effusion_index] == "1.0":
        actual_dictionary['Actual_Effusion'] = 1

    return actual_dictionary

def create_json(original_dict):
    new_dict = {}
    for x in original_dict:
        new_dict.update(x)
        new_dict.update(x)
        new_dict.update(x)

    with open("output.json", "w") as out_file:
        json.dump(new_dict, out_file, indent=4)

    return json.dumps(new_dict, indent=4)

def dcm_to_json(file_path, weight_string, csv_file_path):
    convert_dcm_to_jpg(file_path)
    patient_data = dcm_patient_data(file_path)
    actual_tags = actual_dcm_dict(csv_file_path, patient_data["Study_ID"])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "scaled_image.jpg")
    model_output = run_model(file_path, weight_string)
    json_value = create_json([patient_data, model_output, actual_tags])
    return json_value