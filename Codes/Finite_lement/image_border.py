from PIL import Image, ImageOps
import os

def add_black_border(input_path, output_path, border_size):
    image = Image.open(input_path)

    image_with_border = ImageOps.expand(image, border=(border_size, border_size), fill='black')

    image_with_border.save(output_path)

input_directory = 'BayesCNN/Aerogel_data/last500/Danial_largeFoam/cropped'

output_directory = 'BayesCNN/Aerogel_data/last500/Danial_largeFoam/bordered'

border_size = 4  # You can adjust this size according to your needs

for filename in os.listdir(input_directory):
    if filename.endswith(".png"):  # Adjust the file extension based on your image format
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, f"{filename}")

        add_black_border(input_image_path, output_image_path, border_size)