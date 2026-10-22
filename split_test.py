import os
import shutil
import random

def move_images_to_test(train_folder, test_folder, num_images):
    # Create test folder if it doesn't exist
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Iterate through each folder in train_folder
    for folder_name in os.listdir(train_folder):
        folder_path = os.path.join(train_folder, folder_name)
        if os.path.isdir(folder_path):
            # Create a corresponding folder in the test folder
            test_folder_path = os.path.join(test_folder, folder_name)
            if not os.path.exists(test_folder_path):
                os.makedirs(test_folder_path)
            
            # List all images in the current folder
            images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            # Choose random images to move to the test folder
            images_to_move = random.sample(images, min(num_images, len(images)))
            
            # Move the selected images to the test folder
            for image_name in images_to_move:
                src = os.path.join(folder_path, image_name)
                dst = os.path.join(test_folder_path, image_name)
                shutil.move(src, dst)

train_folder = "/home/jstephen/Desktop/EyeDIseaseDetection/dataset/train"
test_folder = "/home/jstephen/Desktop/EyeDIseaseDetection/dataset/"
num_images_to_move = 1000

move_images_to_test(train_folder, test_folder, num_images_to_move)
