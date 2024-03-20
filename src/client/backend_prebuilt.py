from flask import Flask
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage import io as skio
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from skimage import io
import cv2 as cv
import os
import base64
import json
import io
import torchxrayvision as xrv
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv
from PIL import Image
import json
import skimage.io as skio
from sklearn.metrics import roc_curve, auc, f1_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tabulate import tabulate

app = Flask(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MIMIC_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
 
    def __len__(self):
        return len(self.annotations)
 
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img = skimage.io.imread(img_path + ".jpg") 
        img = xrv.datasets.normalize(img, 255)  
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        # Add color channel
        img = img[None, :, :]
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        image = transform(img)
        image = torch.from_numpy(image).to(device)
        results = []
        # Atelectasis
        results.append(int(self.annotations.iloc[index, 12] == 1.0))
        # Cardiomegaly
        results.append(int(self.annotations.iloc[index, 13] == 1.0))
        # Consolidation
        results.append(int(self.annotations.iloc[index, 14] == 1.0))
        # Edema
        results.append(int(self.annotations.iloc[index, 15] == 1.0))
        # No Finding
        results.append(int(self.annotations.iloc[index, 20] == 1.0))
        # Pleural Effusion
        results.append(int(self.annotations.iloc[index, 21] == 1.0))
        y_label = torch.tensor(results, dtype=torch.float32)
        print("Shape: ", image.shape)
        print("image type", type(image))
        return image, y_label

def show_cam_on_image(img, mask, use_rgb=True, alpha=0.5):
    # Convert Image object to NumPy array
    img_array = np.array(img)
    # Check if img_array is a valid array
    if img_array.size == 0:
        raise ValueError("Invalid input array: img must be a non-empty NumPy array.")
    # Convert mask to a heatmap
    heatmap = cv.applyColorMap(np.uint8(255 * mask), cv.COLORMAP_JET)
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    # Resize heatmap to match img_array dimensions
    if img_array.shape[:2] != heatmap.shape[:2]:
        heatmap = cv.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    # Ensure heatmap is of the same type as img_array
    img_array = np.uint8(img)
    heatmap = np.uint8(heatmap)
    # Blend the heatmap with the original image
    overlayed_img = cv.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img

def generate_gradcam(model, input_tensor, original_image_path, target_category=None):
    target_layers = model.features.denseblock4.denselayer16.conv2
    # Initialize Grad-CAM with the specified target layers
    cam = GradCAM(model=model, target_layers=[target_layers])
    # Define targets based on the specified target category
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    # Generate CAM mask
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    # Load the original image
    original_image = Image.open(original_image_path).convert('RGB')
    # Create visualization with the specified alpha for the overlay transparency
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True, alpha=0.3)  # Adjust alpha as needed
    return visualization

def analyze_model(data_loader, model, device):
    diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]
    bestThresholds = []
    true_labels = []
    predicted_probs = []
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # Adjust indices according to the diseases selected, excluding "No Finding"
            selected_indexes = [0,10,1,4,7]
            selected_probs = outputs[:, selected_indexes]
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
    selected_indexes = [0, 10, 1, 4, 7]
    true_labels = []
    prediction = []
    grad_cam_images = []
    img = skimage.io.imread(filepath)
    img = xrv.datasets.normalize(img, 255)
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")
    # Add color channel
    img = img[None, :, :]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    image = transform(img)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        outputs = model(image)
        probability = outputs[0, selected_indexes].cpu().numpy()
        print("Output: ", outputs)
        print("Probability: ", probability)
    for index in selected_indexes:
        grad_cam_image = generate_gradcam(model, image, filepath, index)
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
    selected_indexes = [0, 10, 1, 4, 7]
    prediction = []
    grad_cam_images = []
    img = skimage.io.imread(filepath)
    img = xrv.datasets.normalize(img, 255)
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")
    # Add color channel
    img = img[None, :, :]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    image = transform(img)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        outputs = model(image)
        probability = outputs[0, selected_indexes].cpu().numpy()
        print("Output: ", outputs)
        print("Probability: ", probability)
    for index in selected_indexes:
        grad_cam_image = generate_gradcam(model, image, filepath, index)
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

def plot_images(original_image, heatmap_image, disease_name):
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the original image
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')  # Hide the axes ticks
    # Normalize the heatmap image if necessary
    if isinstance(heatmap_image, np.ndarray) and heatmap_image.dtype != np.uint8:
        heatmap_image = (heatmap_image - heatmap_image.min()) / (heatmap_image.max() - heatmap_image.min())
    # Plot the heatmap image
    axs[1].imshow(heatmap_image, cmap='hot', alpha=0.5)
    axs[1].set_title('Heatmap for ' + disease_name)
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

def run_with_csv(filepath, csv_file_path):
    weight_string = "densenet121-res224-mimic_nb"
    model = xrv.models.get_model(weight_string).to(device)
    model.eval()  # Set to evaluation mode
    #thresholds are old and determine if model predictions are 1 or 0 based on model probability
    thresholds = [0.67758954, 0.6397526, 0.5293094, 0.5173051, 0.2235725, 0]
    diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]
    print("thresholds: ", thresholds)
    _, grad_cam_image, predictions = test_single_image(filepath, csv_file_path, thresholds, model)
    for x in range(0,6):
        plot_images(Image.open(filepath).convert("RGB"), grad_cam_image[x], diseaseNames[x])
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

def run_with_no_csv_prebuilt(filepath):
    weight_string = "densenet121-res224-mimic_nb"
    model = xrv.models.get_model(weight_string).to(device)
    model.eval()  # Set to evaluation mode
    #thresholds are old and determine if model predictions are 1 or 0 based on model probability
    thresholds = [0.67758954, 0.6397526, 0.5293094, 0.5173051, 0.2235725, 0]
    diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]
    print("thresholds: ", thresholds)
    _, grad_cam_image, predictions = test_single_image_no_csv(filepath, thresholds, model)
    for x in range(0,6):
        plot_images(Image.open(filepath).convert("RGB"), grad_cam_image[x], diseaseNames[x])
    grad_cam_images_base64 = [image_to_base64(img) for img in grad_cam_image]
    predictions[5] = "NaN"
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