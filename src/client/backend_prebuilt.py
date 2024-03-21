from flask import Flask
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from PIL import Image
from skimage import io
import cv2 as cv
import os
import base64
import json
import io
import torchxrayvision as xrv
import skimage
import skimage.io as skio
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tabulate import tabulate

app = Flask(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    target_layers = model.features[-1]
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