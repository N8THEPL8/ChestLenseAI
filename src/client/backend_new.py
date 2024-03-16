from flask import Flask
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from torch.optim import SGD
from skimage import io as skio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from skimage import io
import cv2 as cv
import os
import torchvision.models as models
import base64
import json
import skimage.io as skio
import io
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tabulate import tabulate
from datetime import datetime
import torchxrayvision as xrv
import skimage
import torchvision
import pydicom

app = Flask(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')
# checkpoint_path = "/Users/yuvra/Desktop/CAPSTONE/ai-for-chest-x-ray/src/client/bce_masked_adam8.pth.tar"
checkpoint_path = "bce_masked_adam8.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def actual_dcm_dict(file_path, study_id):
    line_content = []
    with open(file_path) as myFile:
        for _, line in enumerate(myFile, 0):
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

def create_densenet121(num_classes):
    # Load the DenseNet121 model without pre-trained weights
    model = models.densenet121(weights=None)
    # Modify the last fully connected layer to have the output features equal to the number of classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def generate_gradcam(model, input_tensor, original_image, target_category=None):
    target_layers = model.features.denseblock4.denselayer16.conv2
    #target_layers = model.features[-1]
    # Initialize Grad-CAM with the specified target layers
    cam = GradCAM(model=model, target_layers=[target_layers])  # Encapsulate target_layers in a list
    # Define targets based on the specified target category
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    # Generate CAM mask
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    # Create visualization with the specified alpha for the overlay transparency
    visualization = show_cam_on_image(np.array(original_image) / 255.0, grayscale_cam, use_rgb=True, alpha=0.3) # Adjust alpha as needed
    return visualization

# Function to load checkpoint
def local_load_checkpoint(model, optimizer,device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()  # Set the model to evaluation mode
    print(f"Checkpoint loaded from '{checkpoint_path}'")
    return model, optimizer

def show_cam_on_image(img, mask, use_rgb=True, alpha=0.5):
    # Convert mask to a heatmap
    heatmap = cv.applyColorMap(np.uint8(255 * mask), cv.COLORMAP_JET)
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
        img = cv.cvtColor(np.uint8(255 * img), cv.COLOR_BGR2RGB)
    else:
        img = np.uint8(255 * img)
    # Blend the heatmap with the original image
    overlayed_img = cv.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img

def test_single_image(filepath, csv_file_path, thresholds, model, device):
    # Read and preprocess the image
    image = cv.imread(filepath)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    original_image = Image.fromarray(image)
    original_image = original_image.resize((256, 256))
    # Convert the image to a PyTorch tensor and enable gradient
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(original_image).unsqueeze(0).requires_grad_(True)
    image_tensor = image_tensor.to(device)
    # Ensure the model is on the correct device and set it to training mode temporarily
    model = model.to(device)
    model.train()  # Enable tracking of gradients
    prediction = []
    true_labels = []
    grad_cam_image = []
    # Perform forward pass to get predictions and the last convolutional layer's activations
    with torch.enable_grad():  # Ensure gradients are computed
        outputs = model(image_tensor)
        probability = torch.sigmoid(outputs).detach().cpu().numpy()
        # Generate Grad-CAM for the highest scoring category
        _, target_category = torch.max(outputs, 1)
        for x in range (0,6):
            grad_cam_image.append(generate_gradcam(model,image_tensor, original_image, target_category=x))
    # Optionally, switch back to evaluation mode
    model.eval()
    # Extract true labels from CSV file
    filename_without_extension = os.path.splitext(os.path.basename(filepath))[0]
    with open(csv_file_path, 'r') as f:
        datareader = csv.reader(f)
        for row in datareader:
            if row[0] == filename_without_extension:
                true_labels.extend([int(float(row[i])) if row[i] == '1.0' else 0 for i in [12, 13, 14, 15, 20, 21]])
                break
    # Compare predictions and true labels
    for i in range(6):
        prediction.append(1 if probability[0][i] >= thresholds[i] else 0)
    print("probability: ", probability)
    print("prediction: ", prediction)
    print("true lables: ", true_labels)
    # Create a table for visualization
    array1 = ['Atelectasis', probability[0][0], prediction[0], true_labels[0]]
    array2 = ['Cardiomegaly', probability[0][1], prediction[1], true_labels[1]]
    array3 = ['Consolidation', probability[0][2], prediction[2], true_labels[2]]
    array4 = ['Edema', probability[0][3], prediction[3], true_labels[3]]
    array5 = ['No Finding', probability[0][4], prediction[4], true_labels[4]]
    array6 = ['Pleural Effusion', probability[0][5], prediction[5], true_labels[5]]
    table = [['Disease', 'Model Probability', 'Model Prediction', 'True Labels'], array1, array2, array3, array4, array5, array6]
    # Print the table
    print("\n")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    return original_image, grad_cam_image, prediction

def test_single_image_no_csv(filepath, thresholds, model, device):
    # Read and preprocess the image
    image = cv.imread(filepath)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    original_image = Image.fromarray(image)
    original_image = original_image.resize((256, 256))
    # Convert the image to a PyTorch tensor and enable gradient
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(original_image).unsqueeze(0).requires_grad_(True)
    image_tensor = image_tensor.to(device)
    # Ensure the model is on the correct device and set it to training mode temporarily
    model = model.to(device)
    model.train()  # Enable tracking of gradients
    prediction = []
    true_labels = []
    grad_cam_image = []
    # Perform forward pass to get predictions and the last convolutional layer's activations
    with torch.enable_grad():  # Ensure gradients are computed
        outputs = model(image_tensor)
        probability = torch.sigmoid(outputs).detach().cpu().numpy()
        # Generate Grad-CAM for the highest scoring category
        _, target_category = torch.max(outputs, 1)
        for x in range (0,6):
            grad_cam_image.append(generate_gradcam(model,image_tensor, original_image, target_category=x))
    # Optionally, switch back to evaluation mode
    model.eval()
    # Compare predictions and true labels
    for i in range(6):
        prediction.append(1 if probability[0][i] >= thresholds[i] else 0)
    print("probability: ", probability)
    print("prediction: ", prediction)
    # Create a table for visualization
    array1 = ['Atelectasis', probability[0][0], prediction[0] ]
    array2 = ['Cardiomegaly', probability[0][1], prediction[1]]
    array3 = ['Consolidation', probability[0][2], prediction[2]]
    array4 = ['Edema', probability[0][3], prediction[3]]
    array5 = ['No Finding', probability[0][4], prediction[4]]
    array6 = ['Pleural Effusion', probability[0][5], prediction[5]]
    table = [['Disease', 'Model Probability', 'Model Prediction'], array1, array2, array3, array4, array5, array6]
    # Print the table
    print("\n")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    return original_image, grad_cam_image, prediction

def plot_images(original_image, grad_cam_image, diseaseName):
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the original image
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')  # Hide the axes ticks
    # Plot the Grad-CAM heatmap
    axs[1].imshow(grad_cam_image)
    axs[1].set_title('Grad-CAM Heatmap for ' + diseaseName)
    axs[1].axis('off')  # Hide the axes ticks
    # Display the plot
    # plt.show()

def image_to_base64(image_array):
    image = Image.fromarray(np.uint8(image_array)).convert('RGB')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

# Function to preprocess and convert image to byte array
def preprocess_and_convert_to_byte_array(image_path):
    # Preprocess the image
    image = skio.imread(image_path)
    RGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    new_image = Image.fromarray(RGB)
    new_image = new_image.resize((256, 256))
    # Convert to byte array
    img_byte_arr = io.BytesIO()
    new_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

# Function to convert byte array back to image
def byte_array_to_image(byte_array):
    img_byte_arr = io.BytesIO(byte_array)
    img = Image.open(img_byte_arr)
    return img

# Test Single Image:
def run_with_no_csv(filepath):
    #model = MyNeuralNet()
    model = create_densenet121(6)
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    loaded_model, loaded_optimizer = local_load_checkpoint(model, optimizer,device)
    loaded_model.eval()
    loaded_model.to(device)
    #thresholds are old and determine if model predictions are 1 or 0 based on model probability
    thresholds = [0.53880334, 0.48418066, 0.36754248, 0.5815063, 0.54026645, 0.47844747]
    diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]
    print("thresholds: ", thresholds)
    #is you use test_single_image or test_single_image_no_csv depends on if you have the true labels
    original_image, grad_cam_image, predictions = test_single_image_no_csv(filepath, thresholds, model, device)
    for x in range(0,6):
        plot_images(original_image, grad_cam_image[x], diseaseNames[x])
    grad_cam_images_base64 = [image_to_base64(img) for img in grad_cam_image]
    diseases_data = []
    for i, disease_name in enumerate(diseaseNames):
        diseases_data.append({
            "diseaseName": disease_name,
            "prediction": predictions[i],
            "gradCamImage": grad_cam_images_base64[i]
        })
    data = { "diseasesData": diseases_data }
    # Convert the structured data to a JSON string
    json_data = json.dumps(data, indent=4)
    with open('diseases_predictions_and_images.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return json_data

def convert_dcm_to_jpg(file_path):
    ds = pydicom.dcmread(file_path)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")
    final_image.save(file_path)
    return