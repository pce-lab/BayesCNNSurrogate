import os
from PIL import Image

def crop_to_square(image_path, output_path, crop_size):
    """
    Crop the given image to a square of the specified size.

    :param image_path: Path to the input image.
    :param output_path: Path to save the cropped image.
    :param crop_size: The size (width and height) to crop the image to.
    """
    try:
        # Open the image
        image = Image.open(image_path)
        width, height = image.size

        # Calculate cropping box coordinates
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # Save the cropped image
        cropped_image.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images(input_folder, output_folder, crop_size):
    """
    Process all images in the input folder, cropping them to a square,
    and save the cropped images to the output folder.

    :param input_folder: Folder containing the input images.
    :param output_folder: Folder to save the cropped images.
    :param crop_size: The size (width and height) to crop the images to.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            crop_to_square(input_path, output_path, crop_size)
            print(f"Processed {filename}")

input_folder = 'BayesCNN/Aerogel_data/last500/Danial_largeFoam'  
output_folder = 'BayesCNN/Aerogel_data/last500/Danial_largeFoam/cropped'  
crop_size = 350  # Replace with your desired crop size

process_images(input_folder, output_folder, crop_size)
