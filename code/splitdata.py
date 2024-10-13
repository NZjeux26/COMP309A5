import os
import shutil
import random

# Define source directories for each class
base_dir = '../train_data'
classes = ['cherry', 'strawberry', 'tomato']

# Directories for splits
output_base_dir = '../split_data'
train_dir = os.path.join(output_base_dir, 'train')
test_dir = os.path.join(output_base_dir, 'test')
blind_test_dir = os.path.join(output_base_dir, 'blind_test')

# Create the split directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(blind_test_dir, exist_ok=True)

# Define the number of images for each split
test_size = 400
blind_test_size = 100

# Iterate through each class folder
for cls in classes:
    # Path to the current class folder
    class_dir = os.path.join(base_dir, cls)
    
    # Get all image filenames in the class folder
    images = os.listdir(class_dir)
    
    # Randomize the image list
    random.shuffle(images)
    
    # Split the data
    test_images = images[:test_size]
    blind_test_images = images[test_size:test_size + blind_test_size]
    train_images = images[test_size + blind_test_size:]
    
    # Create class subdirectories in the train, test, and blind_test folders
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(blind_test_dir, cls), exist_ok=True)
    
    # Copy test images
    for img in test_images:
        src = os.path.join(class_dir, img)
        dest = os.path.join(test_dir, cls, img)
        shutil.copy(src, dest)
    
    # Copy blind test images
    for img in blind_test_images:
        src = os.path.join(class_dir, img)
        dest = os.path.join(blind_test_dir, cls, img)
        shutil.copy(src, dest)
    
    # Copy train images
    for img in train_images:
        src = os.path.join(class_dir, img)
        dest = os.path.join(train_dir, cls, img)
        shutil.copy(src, dest)

print("Data split completed successfully!")
