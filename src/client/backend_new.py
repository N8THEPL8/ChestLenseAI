import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv
import os
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tabulate import tabulate

checkpoint_path = "bce_masked_adam8.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return original_image, grad_cam_image

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
    return original_image, grad_cam_image

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

model = create_densenet121(6)
optimizer = torch.optim.Adam(model.parameters(), 0.001)
loaded_model, loaded_optimizer = local_load_checkpoint(model, optimizer,device)
loaded_model.eval()
loaded_model.to(device)

filepath = "uploads/8c0171a3-925313ff-f63faed5-3007b5ad-d1bbb676.jpg"
csv_file_path = "Validation_Partial.csv"

#thresholds are old and determine if model predictions are 1 or 0 based on model probability
thresholds = [0.53880334, 0.48418066, 0.36754248, 0.5815063, 0.54026645, 0.47844747]
diseaseNames = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "No Finding", "Pleural Effusion"]

print("thresholds: ", thresholds)

#is you use test_single_image or test_single_image_no_csv depends on if you have the true labels
original_image, grad_cam_image = test_single_image(filepath, csv_file_path, thresholds, model, device)
original_image, grad_cam_image = test_single_image_no_csv(filepath, thresholds, model, device)

for x in range(0,6):
    plot_images(original_image, grad_cam_image[x], diseaseNames[x])