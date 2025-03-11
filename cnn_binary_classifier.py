from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from PIL import ImageFile
import torch
from torch import nn
from torch import device
from torch import cuda
import torchsummary as torchsummary
import pandas as pd
from datetime import datetime

"""
--------------------------
CNN BINARY CLASSIFIER
--------------------------
by Thomas Vaughn
Date: 3/10/2025

This is a prediction script which uses a Custom CNN model to perform 
a binary classification task.

"""


device = device("cuda" if cuda.is_available() else "cpu")

MODEL_PATH = './image_binary_classifier_cnn_01_1'

input_dir = "./Input_Images"

POSITIVE_PRED_MAX = 0.6  # Range: 0.0 - 1.0, the higher the value the more likely to be considered a match

_batch_size = 16
resized_size = 64
kernel_size = 3
padding = 1
input_features = 3

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


class CNNTripleBlock(nn.Module):
    def __init__(self, in_features):
        super(CNNTripleBlock, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                              padding=padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(in_channels=in_features, out_channels=in_features * 2, kernel_size=kernel_size,
                              padding=padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(in_channels=in_features * 2, out_channels=in_features * 4, kernel_size=kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(self.cnn3.out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.cnn3(x)
        x = self.bn(x)
        x = self.relu3(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features):
        super(LinearBlock, self).__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.do1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.do1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.cnnblock1 = CNNTripleBlock(in_features=input_features)
        self.fltn = nn.Flatten()
        self.linearblock = LinearBlock(in_features=49152)

    def forward(self, x):

        x = self.cnnblock1(x)
        x = self.fltn(x)
        x = self.linearblock(x)
        x = nn.functional.sigmoid(x)

        return x


model = CustomCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

torchsummary.summary(model, (3, resized_size, resized_size))

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

predicted_buckets_df.to_csv(path_or_buf='./Output/cnn_binary_classifier_output_' +
                                        str(datetime.now().strftime('%Y-%m-%d_%H.%M.%S')) +
                                        '.csv', sep=',', encoding='utf-8', index=False)

print("\nOutput file saved.  Please close program window when ready.")

exit()
