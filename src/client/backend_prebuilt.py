from flask import Flask
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
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
import base64
import json
import io
import skimage.io as skio
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tabulate import tabulate

app = Flask(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "model.pth.tar"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Use GPU 2

def jpeg_to_png(im):
    png_image_io = io.BytesIO()
    im.save(png_image_io, format='PNG')
    png_image_io.seek(0)
    return Image.open(png_image_io)

class MIMIC_Dataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = self.get_transform()
    def get_transform(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0] + ".jpg")
        image = Image.open(img_path).convert('RGB')
        image = jpeg_to_png(image)
        # Apply the transformations
        image = self.transform(image)
        # Extract labels for each condition
        labels = self.annotations.iloc[index, [12, 13, 14, 15, 20, 21]].values.astype(float)
        y_label = torch.tensor(labels, dtype=torch.float32)
        return image, y_label

def show_cam_on_image(img, mask, use_rgb=True, alpha=0.5):
    # Convert mask to a heatmap
    heatmap = cv.applyColorMap(np.uint8(255 * mask), cv.COLORMAP_JET)
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    img = np.uint8(img * 255)  # Ensure img is scaled properly
    # Resize heatmap to match img dimensions
    if img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    # Ensure both images are of the same type
    img = np.uint8(img)
    heatmap = np.uint8(heatmap)
    # Blend the heatmap with the original image
    overlayed_img = cv.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img

def generate_gradcam(model, input_tensor, original_image, target_category=None):
    # target_layers = model.features.denseblock4.denselayer16.conv2
    target_layers = model.module.densenet121.features.denseblock4.denselayer16.conv2
    # Initialize Grad-CAM with the specified target layers
    cam = GradCAM(model=model, target_layers=[target_layers])  # Encapsulate target_layers in a list
    # Define targets based on the specified target category
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    # Generate CAM mask
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    # Create visualization with the specified alpha for the overlay transparency
    visualization = show_cam_on_image(np.array(original_image) / 255.0, grayscale_cam, use_rgb=True, alpha=0.3) # Adjust alpha as needed
    return visualization

class DenseNet121(nn.Module):
    """
    Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.densenet121(x)
        return x

def load_checkpoint(filepath, model, device):
    if os.path.isfile(filepath):
        print("=> loading checkpoint")
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
    return model

def analyze_model(data_loader, model, device):
    # "No Finding" is placed at index 4
    diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]
    bestThresholds = []
    true_labels = []
    predicted_probs = []
    model.eval()  # Ensure the model is in evaluation mode
    count = 0 
    with torch.no_grad():
        for images, labels in data_loader:
            print(count)
            count += 1
            images = images.to(device)
            labels = labels.to(device)
            # Handle TenCrop
            if images.dim() == 5:
                batch_size, crops, c, h, w = images.size()
                images = images.view(-1, c, h, w)
                outputs = model(images)
                outputs = outputs.view(batch_size, crops, -1).mean(1)
            else:
                outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            # Adjust indices according to the diseases selected, excluding "No Finding"
            selected_indexes = [0, 1, 8, 2, 9]  # Example indices, adjust based on actual mapping
            selected_probs = probabilities[:, selected_indexes]
            # Append "No Finding" probabilities based on other diseases
            no_finding_probs = (1 - selected_probs[:, :-1].max(axis=1).values).reshape(-1, 1)
            selected_probs = torch.cat((selected_probs[:, :-1], no_finding_probs, selected_probs[:, -1:]), axis=1)
            predicted_probs.append(selected_probs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    predicted_probs = np.concatenate(predicted_probs)
    true_labels = np.concatenate(true_labels)
    true_labels = np.nan_to_num(true_labels)  # Convert NaN to 0
    # Update "No Finding" labels based on other diseases
    no_finding_labels = (true_labels.sum(axis=1) == 0).astype(int)
    true_labels = np.insert(true_labels, 4, no_finding_labels, axis=1)  # Insert at index 4
    # Calculate ROC curve, AUC, and F1 score for each disease
    for i, disease in enumerate(diseaseNames):
        binary_true_labels = true_labels[:, i]
        pred_probs_i = predicted_probs[:, i].flatten()
        binary_true_labels = np.where(binary_true_labels > 0, 1, 0)  # Ensure binary labels
        fpr, tpr, thresholds = roc_curve(binary_true_labels, pred_probs_i)
        roc_auc = auc(fpr, tpr)
        best_threshold_index = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_index]
        bestThresholds.append(best_threshold)
        print(f"{disease}:")
        print(f"   AUC: {roc_auc:.4f}")
        print(f"   Best Threshold: {best_threshold:.4f}")
        binary_predictions = (pred_probs_i > best_threshold).astype(int)
        f1 = f1_score(binary_true_labels, binary_predictions)
        print(f"   F1 Score Based on Best Threshold: {f1:.4f}")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - {disease}')
        plt.legend(loc="lower right")
        plt.show()
    return bestThresholds

def test_single_image(filepath, csv_file_path, thresholds, model):
    selected_indexes = [0, 1, 8, 9, 2]
    true_labels = []
    prediction = []
    grad_cam_images = []
    # Replicate the transform used in MIMIC_Dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
    # Load and preprocess the image
    image = Image.open(filepath).convert('RGB')
    # Convert the JPEG image to PNG format
    image = jpeg_to_png(image)
    image_crops = transform(image).requires_grad_(True)  # This should already be a stacked tensor of crops
    # Move the crops to the correct device
    image_crops = image_crops.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Flatten the crops into the batch dimension and pass through the model
        outputs = model(image_crops.view(-1, 3, 224, 224))
        # Calculate the mean across the crops to get a single prediction vector
        outputs = outputs.view(10, -1).mean(0, keepdim=True)
        probabilities = torch.sigmoid(outputs)
    probability = probabilities[0, selected_indexes].cpu().numpy()
    for index in selected_indexes:
        grad_cam_image = generate_gradcam(model, image_crops, image, index)
        grad_cam_images.append(grad_cam_image)
    grad_cam_images.insert(4, Image.open(filepath).convert('RGB'))
    model.eval()  # Switch back to evaluation mode
    # Extract true labels from CSV file
    filename_without_extension = os.path.splitext(os.path.basename(filepath))[0]
    with open(csv_file_path, 'r') as f:
        datareader = csv.reader(f)
        for row in datareader:
            if row[0] == filename_without_extension:
                true_labels.extend([int(float(row[i])) if row[i] == '1.0' else 0 for i in [12, 13, 14, 15, 20, 21]])
                break
    # Compare predictions and true labels
    for i in range(5):
        prediction.append(1 if probability[i] >= thresholds[i] else 0)
    allZero = True
    for i in range(5):
        if prediction[i] == 1:
            allZero = False
            break
    if allZero == True:
        prediction.insert(4, 1)
        probability = probability.astype(object)
        probability = np.insert(probability, 4, "-")
    else:
        prediction.insert(4, 0)
        probability = probability.astype(object)
        probability = np.insert(probability, 4, "-")
    print("probability: ", probability)
    print("prediction: ", prediction)
    print("true labels: ", true_labels)
    # Create a table for visualization
    array1 = ['Atelectasis', probability[0], prediction[0], true_labels[0]]
    array2 = ['Cardiomegaly', probability[1], prediction[1], true_labels[1]]
    array3 = ['Consolidation', probability[2], prediction[2], true_labels[2]]
    array4 = ['Edema', probability[3], prediction[3], true_labels[3]]
    array5 = ['No Finding', probability[4], prediction[4], true_labels[4]]
    array6 = ['Pleural Effusion', probability[5], prediction[5], true_labels[5]]
    table = [['Disease', 'Model Output', 'Model Prediction', 'True Labels'],
             array1, array2, array3, array4, array5, array6]
    # Print the table
    print("\n")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    return image, grad_cam_images, prediction

def test_single_image_no_csv(filepath, thresholds, model):
    selected_indexes = [0, 1, 8, 9, 2]
    prediction = []
    grad_cam_images = []
    # Replicate the transform used in MIMIC_Dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
    # Load and preprocess the image
    image = Image.open(filepath).convert('RGB')
    # Convert the JPEG image to PNG format
    image = jpeg_to_png(image)
    image_crops = transform(image).requires_grad_(True)  # This should already be a stacked tensor of crops
    # Move the crops to the correct device
    image_crops = image_crops.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Flatten the crops into the batch dimension and pass through the model
        outputs = model(image_crops.view(-1, 3, 224, 224))
        # Calculate the mean across the crops to get a single prediction vector
        outputs = outputs.view(10, -1).mean(0, keepdim=True)
        probabilities = torch.sigmoid(outputs)
    probability = probabilities[0, selected_indexes].cpu().numpy()
    for index in selected_indexes:
        grad_cam_image = generate_gradcam(model, image_crops, image, index)
        grad_cam_images.append(grad_cam_image)
    grad_cam_images.insert(4, Image.open(filepath).convert('RGB'))
    model.eval()  # Switch back to evaluation mode
    # Compare predictions and true labels
    for i in range(5):
        prediction.append(1 if probability[i] >= thresholds[i] else 0)
    allZero = True
    for i in range(5):
        if prediction[i] == 1:
            allZero = False
            break
    if allZero == True:
        prediction.insert(4, 1)
        probability = probability.astype(object)
        probability = np.insert(probability, 4, "-")
    else:
        prediction.insert(4, 0)
        probability = probability.astype(object)
        probability = np.insert(probability, 4, "-")
    print("probability: ", probability)
    print("prediction: ", prediction)
    # Create a table for visualization
    array1 = ['Atelectasis', probability[0], prediction[0]]
    array2 = ['Cardiomegaly', probability[1], prediction[1]]
    array3 = ['Consolidation', probability[2], prediction[2]]
    array4 = ['Edema', probability[3], prediction[3]]
    array5 = ['No Finding', probability[4], prediction[4]]
    array6 = ['Pleural Effusion', probability[5], prediction[5]]
    table = [['Disease', 'Model Output', 'Model Prediction'],
             array1, array2, array3, array4, array5, array6]
    # Print the table
    print("\n")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    return image, grad_cam_images, prediction

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
    plt.show()

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
def run_with_csv(filepath, csv_file_path):
    model = DenseNet121(14)  # Assuming 14 is the number of output classes
    model = torch.nn.DataParallel(model).to(device)  # Only wrap and move to device once
    loaded_model = load_checkpoint(CKPT_PATH, model, device)
    loaded_model.eval()  # Set to evaluation mode
    #thresholds are old and determine if model predictions are 1 or 0 based on model probability
    thresholds = [0.5841402, 0.519332, 0.63460743, 0.5737186, 0.56760746]
    diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]
    print("thresholds: ", thresholds)
    #is you use test_single_image or test_single_image_no_csv depends on if you have the true labels
    original_image, grad_cam_image, predictions = test_single_image(filepath, csv_file_path, thresholds, model)
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
    return json_data

# Test Single Image:
def run_with_no_csv_prebuilt(filepath):
    model = DenseNet121(14)  # Assuming 14 is the number of output classes
    model = torch.nn.DataParallel(model).to(device)  # Only wrap and move to device once
    loaded_model = load_checkpoint(CKPT_PATH, model, device)
    loaded_model.eval()  # Set to evaluation mode
    #thresholds are old and determine if model predictions are 1 or 0 based on model probability
    thresholds = [0.5841402, 0.519332, 0.63460743, 0.5737186, 0.56760746]
    diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]
    print("thresholds: ", thresholds)
    #is you use test_single_image or test_single_image_no_csv depends on if you have the true labels
    original_image, grad_cam_image, predictions = test_single_image_no_csv(filepath, thresholds, model)
    #original_image, grad_cam_image = test_single_image_no_csv(filepath, thresholds, model, device)
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
    return json_data