from PIL import Image
import os

def resize_images(input_paths, target_size=(256, 256)):
    for input_folder in input_paths:
        output_folder = input_folder + "_resized"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = os.listdir(input_folder)
        i = 1
        for file in files:
            input_path = os.path.join(input_folder, file)
            print("Running for file ", i, " in folder ", input_folder)
            if os.path.isfile(input_path):
                with Image.open(input_path) as img:
                    rgb_img = img.convert('RGB')
                    resized_img = rgb_img.resize(target_size)
                    output_path = os.path.join(output_folder, file)
                    resized_img.save(output_path)
                    os.remove(input_path)  # Delete the old image
            i += 1

if __name__ == "__main__":
    input_paths = [
    r'/home/jstephen/Desktop/EyeDIseaseDetection/dataset/other',
    ]

    target_size = (256, 256)

    resize_images(input_paths, target_size)