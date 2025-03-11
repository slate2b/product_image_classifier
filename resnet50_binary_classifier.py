from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from PIL import ImageFile
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch
from torch import nn
from torch import device
from torch import cuda
import pandas as pd
from datetime import datetime

"""
--------------------------
RESNET50 BINARY CLASSIFIER
--------------------------
by Thomas Vaughn
Date: 2/24/2025

This is a prediction script which uses a fine-tuned ResNet50 model
trained to perform a binary classification task.

"""

device = device("cuda" if cuda.is_available() else "cpu")

MODEL_PATH = './Models/image_classifier_resnet50_08_1.pt'

input_dir = "./Input_Images"

POSITIVE_PRED_MAX = 0.01  # Range: 0.0 - 1.0, the higher the value the more likely to be considered a match

_batch_size = 16
resized_size = 128

criterion = torch.nn.BCELoss()

input_transform = transforms.Compose([
    transforms.Resize(size=(resized_size, resized_size)),
    transforms.ToTensor(),
])

train_class_counts = {}
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        train_class_counts[class_folder] = num_images

input_data = []
image_filenames = []

for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        train_class_counts[class_folder] = num_images

        file_list_from_os = os.listdir(class_path)
        file_list_from_os.sort()  # otherwise lowercase and uppercase would get sorted differently from ImageFolder

        for y in range(len(file_list_from_os)):

            fname = file_list_from_os[y]
            image_filenames.append(fname)
            fpath = class_path + "/" + fname

            try:
                img = ImageFile.Image.open(fpath)

            except OSError:
                print("Cannot load : {}".format(fpath))

input_data = datasets.ImageFolder(root=input_dir, transform=input_transform)

print(f"Input data:\n{input_data}")

input_set = DataLoader(dataset=input_data,
                       batch_size=_batch_size,
                       num_workers=0,
                       shuffle=False)

model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

model.load_state_dict((torch.load(MODEL_PATH, map_location=torch.device('cpu'))))
model.to(device)

print("\nInitiating prediction...")


def predict(mdl, loader):
    """
    Predicts whether the input images match the "match" images from binary training.

    :param mdl: The model used for prediction
    :param loader: The DataLoader for prediction
    :return: predicted lbls (list), predicted probabilities (list)
    """

    mdl.eval()  # set model to evaluation mode

    predicted_lbls = []
    predicted_probabilities = []

    with torch.no_grad():

        for inputs in loader:

            input_for_model = inputs[0]  # Point to the correct dimension since ImageFolder expects multiple classes
            outputs = mdl(input_for_model)

            for j in range(len(outputs)):
                if outputs[j] < POSITIVE_PRED_MAX:
                    predicted_lbls.append(0)
                else:
                    predicted_lbls.append(1)
                predicted_probabilities.append(outputs[j])

    return predicted_lbls, predicted_probabilities


model_output = predict(mdl=model, loader=input_set)

print("\nAnalysis complete.  Collecting output data...")

label_data = model_output[0]
predicted_labels = []

for i in label_data:

    predicted_labels.append(i)

predicted_buckets = []

for i in range(len(predicted_labels)):
    if predicted_labels[i] == 0:
        bucket = "Possible Match"
    elif predicted_labels[i] == 1:
        bucket = "Not a Match"
    else:
        bucket = "Error"
    predicted_buckets.append(bucket)

predicted_labels_series = pd.Series(predicted_labels)
predicted_buckets_series = pd.Series(predicted_buckets)
image_filenames_series = pd.Series(image_filenames)
predicted_buckets_df = pd.DataFrame(columns=['binary_label', 'analysis', 'image_filename'])
predicted_buckets_df['binary_label'] = predicted_labels_series
predicted_buckets_df['analysis'] = predicted_buckets_series
predicted_buckets_df['image_filename'] = image_filenames_series

predicted_buckets_df.to_csv(path_or_buf='./Output/resnet50_binary_classifier_output' +
                                        str(datetime.now().strftime('%Y-%m-%d_%H.%M.%S')) +
                                        '.csv', sep=',', encoding='utf-8', index=False)

print("\nOutput file saved.  Please close program window when ready.")

exit()
