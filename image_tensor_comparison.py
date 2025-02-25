from torchvision import transforms
import os
import torch
from torch import device
from torch import cuda
from PIL import ImageFile
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time

device = device("cuda" if cuda.is_available() else "cpu")

base_image_dir = "./Base_Images"
input_dir = "./Input_Images"

input_starting_index = 0  # the index of the image within the input folder to start with
max_number_of_images = 10000

resized_size = 64

transform = transforms.Compose([
    transforms.Resize(size=(resized_size, resized_size)),
    transforms.ToTensor()
])

base_image_counts = {}
for class_folder in os.listdir(base_image_dir):
    class_path = os.path.join(base_image_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        base_image_counts[class_folder] = num_images

base_image_data = []
base_image_filenames = []

for image in range(len(os.listdir(base_image_dir))):

    fname = os.listdir(base_image_dir)[image]
    base_image_filenames.append(fname)
    fpath = base_image_dir + "/" + fname

    try:
        img = ImageFile.Image.open(fpath)
        img_tensor = transform(img)
        base_image_data.append(img_tensor)

    except OSError:
        print("Cannot load : {}".format(fpath))

time.sleep(.3)

print("\nAnalyzing input images...")

input_data = []
input_image_filenames = []

processed_image_counter = 0

for image in tqdm(range(len(os.listdir(input_dir)))):

    if image < input_starting_index:
        continue
    elif processed_image_counter >= max_number_of_images:
        print("\nProcessed max number of images: " + str(max_number_of_images))
        break
    else:
        fname = os.listdir(input_dir)[image]
        input_image_filenames.append(fname)
        fpath = input_dir + "/" + fname

        try:
            img = ImageFile.Image.open(fpath)
            img_tensor = transform(img)
            input_data.append(img_tensor)
            processed_image_counter += 1

        except:
            print("Cannot load : {}".format(fpath))
            input_data.append("Cannot load " + str(fname))
            processed_image_counter += 1

time.sleep(.3)

print("\nComparing base images to input image data...")

output_data = []

for i in tqdm(range(len(input_data))):

    output = "No Match"

    for b in range(len(base_image_data)):
        try:
            if torch.equal(input_data[i], base_image_data[b]):
                output = "Match Found"
        except:
            output = "No Match"

    output_data.append(output)

time.sleep(.3)

print("\nAnalysis complete. Saving output file...")

image_filenames_series = pd.Series(input_image_filenames)
output_data_series = pd.Series(output_data)

output_df = pd.DataFrame(columns=['image_filename', 'analysis'])
output_df['image_filename'] = image_filenames_series
output_df['analysis'] = output_data_series

output_df.to_csv(path_or_buf='./Output/tensor_comparison_analysis_' +
                             str(datetime.now().strftime('%Y-%m-%d_%H.%M.%S')) +
                             '.csv', sep=',', encoding='utf-8', index=False)

print("\nSuccess!\n")

exit()
