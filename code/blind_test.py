import torch
import os
import shutil
from PIL import Image
from torchvision import transforms
from fruit_class import FruitClassifierCNN

# Function to predict the class of an image using the trained model
def predict_image(model, image, device, class_names):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to calculate gradients for inference
        image = image.to(device)
        outputs = model(image.unsqueeze(0))  # Add batch dimension (1, C, H, W)
        _, predicted = torch.max(outputs, 1)  # Get the index of the class with the highest score
        return class_names[predicted.item()]  # Return the predicted class name

# Function to sort images into class folders
def sort_images(model, input_folder, output_folder, device, class_names, transform):
    # Check if output folders for classes exist, create them if not
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    # Loop over all images in the input folder
    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image types
            img_path = os.path.join(input_folder, img_name)

            # Open and preprocess the image
            image = Image.open(img_path).convert('RGB')
            image = transform(image)

            # Predict the class
            predicted_class = predict_image(model, image, device, class_names)

            # Move or copy the image to the respective class folder
            output_class_folder = os.path.join(output_folder, predicted_class)
            shutil.move(img_path, os.path.join(output_class_folder, img_name))  # Move image
            print(f"Moved '{img_name}' to '{predicted_class}' folder.")

def main():
    # Path settings
    input_folder = os.path.join('..', 'split_data/blind_test')  # Folder with unlabeled images
    output_folder = os.path.join('..', 'sorted_test')  # Folder to save sorted images
    model_path = 'fruit_classifier.pth'  # Path to the trained model

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # GPU/CPU setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Define image transformations (same as during training)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])

    # Class labels (same as used during training)
    class_names = ['cherry', 'strawberry', 'tomato']

    # Load the trained model
    model = FruitClassifierCNN(num_classes=len(class_names))
    checkpoint = torch.load(model_path, map_location=device)  # Load model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("\nSorting images...")
    # Sort images into their respective class folders
    sort_images(model, input_folder, output_folder, device, class_names, transform)
    print("\nAll images have been sorted!")

if __name__ == '__main__':
    main()
