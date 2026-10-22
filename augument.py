import os
import random
import cv2         
import imgaug as ia
from imgaug import augmenters as iaa
import uuid

def augment_images_in_folder(folder_path, desired_count):
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    existing_count = len(image_files)

    seq = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10)),  # Rotate up to 10 degrees
        iaa.Affine(scale=(0.9, 1.1)),  # Zoom in/out
        iaa.Fliplr(0.5),  # Flip horizontally with a 50% chance
        iaa.Flipud(0.5),  # Flip vertically with a 50% chance
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Add Gaussian blur with a small sigma
        iaa.AdditiveGaussianNoise(scale=(0, 0.01*255))  # Add a small amount of Gaussian noise
    ], random_order=True)

    i = 1
    while existing_count < desired_count:
        file = random.choice(image_files)
        image = cv2.imread(os.path.join(folder_path, file))
        augmented_image = seq.augment_image(image)
        
        unique_filename = str(uuid.uuid4())[:8]  
        output_file = f"{unique_filename}_augmented{os.path.splitext(file)[1]}"
        output_path = os.path.join(folder_path, output_file)
        cv2.imwrite(output_path, augmented_image)
        
        existing_count += 1
        print("Running for image ", i)
        i += 1

    print(f"A total of {existing_count} images were augmented in {folder_path}.")

input_folders = [
    r'/home/jstephen/Desktop/EyeDIseaseDetection/dataset/cataract',
    r'/home/jstephen/Desktop/EyeDIseaseDetection/dataset/diabetic',
    r'/home/jstephen/Desktop/EyeDIseaseDetection/dataset/glaucoma',
    r'/home/jstephen/Desktop/EyeDIseaseDetection/dataset/normal',
]

desired_image_count = 6000 

ia.seed(1)

for folder_path in input_folders:
    augment_images_in_folder(folder_path, desired_image_count)

print("Augmentation completed.")