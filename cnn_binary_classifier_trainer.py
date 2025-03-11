from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import torchsummary as torchsummary
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import device
from torch import cuda
from torch.optim import SGD
import time

"""
----------------------------------
CNN BINARY CLASSIFIER TRAINER
----------------------------------
by Thomas Vaughn
Date: 3/10/2025

This training script is to create and train a CNN model for a binary 
classification use case.  The model employs a block of 3 CNN 
layers without any max pooling which then feed a Linear block which
performs the final classification.  

I performed many experiments with different configurations, including
models with multiple multi-layer CNN blocks with max pooling 
between each block, but this simpler configuration actually tested
better than all the rest for the given use case.  

"""


device = device("cuda" if cuda.is_available() else "cpu")

_model_save_path = './image_binary_classifier_cnn_01_1'
_optimizer_save_path = './image_binary_classifier_cnn_01_1_optimizer'

train_dir = "./Training_Images/train"
validation_dir = "./Training_Images/validation"

batch_size = 16
resized_size = 64
kernel_size = 3
padding = 1
input_features = 3
learning_rate = 0.001
num_epochs = 15
start_epoch = 0

train_transform = transforms.Compose([
    transforms.Resize(size=(resized_size, resized_size)),
    transforms.ToTensor(),
])

validation_transform = transforms.Compose([
    transforms.Resize(size=(resized_size, resized_size)),
    transforms.ToTensor(),
])


def checkpoint(mdl, optim, filename):
    """
    Saves a model checkpoint with a filename which identifies the epoch.

    :param mdl: The model to be trained
    :param optim: The optimizer to use for training
    :param filename: The string which designates the specific epoch
    :return: None
    """

    model_fp = _model_save_path + filename
    optimizer_fp = _optimizer_save_path + filename
    torch.save(mdl.state_dict(), model_fp)
    torch.save(optim.state_dict(), optimizer_fp)


def resume(mdl, optim, filename):
    """
    Resumes training from a specific epoch using a previously saved checkpoint.

    :param mdl: The model to be trained
    :param optim: The optimizer to use for training
    :param filename: The string which designates the specific epoch
    :return: predicted labels (list), predicted probabilities (list)
    """

    model_fp = _model_save_path + filename
    optimizer_fp = _optimizer_save_path + filename
    mdl.load_state_dict(torch.load(model_fp))
    optim.load_state_dict(torch.load(optimizer_fp))


train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
validation_data = datasets.ImageFolder(root=validation_dir, transform=validation_transform)

print(f"Train data:\n{train_data}\n\nValidation data:\n{validation_data}")

train_set = DataLoader(dataset=train_data,
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=True)

validation_set = DataLoader(dataset=validation_data,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True)

train_dir = train_dir
validation_dir = validation_dir

train_class_counts = {}
for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        train_class_counts[class_folder] = num_images

validation_class_counts = {}
for class_folder in os.listdir(validation_dir):
    class_path = os.path.join(validation_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        validation_class_counts[class_folder] = num_images

print(train_class_counts)
print(validation_class_counts)

dataloaders = {'train': train_set, 'validation': validation_set}

input_features = 3
input_square_dim = (len(train_data[0][0][0]))  # for a 32x32 image, this value is 32


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

torchsummary.summary(model, (3, resized_size, resized_size))

criterion = torch.nn.BCELoss()
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def train_model(mdl, loaders, crit, optim, num_eps):
    """
    Predicts the similarity of an image given multiple images as input.

    :param mdl: The model to be trained
    :param loaders: The dictionary of dataloaders for training and validation
    :param crit: The criterion used for optimization
    :param optim: The optimizer used during training
    :param num_eps: The maximum number of epochs to train
    :return: mdl (list), history_dict (list)
    """

    since = time.time()

    # Lists to hold train and validation metrics for plotting
    validation_acc_history = []
    validation_loss_history = []
    train_acc_history = []
    train_loss_history = []

    is_finished = False

    best_acc = 0.0
    best_loss = 100.0  # initializing with an extremely high value for early stopping logic

    # if resuming training at a specific epoch
    if start_epoch > 0:
        resume_epoch = start_epoch - 1
        resume(mdl, optim, f"epoch-{resume_epoch}.pt")

    # The training loop
    for epoch in range(num_eps):

        if not is_finished:

            print('Epoch {}/{}'.format(epoch+1, num_eps))

            total_acc, total_count = 0, 0
            epoch_accuracy = ""
            epoch_loss = ""

            # using phases to utilize much of the same code for training and validation
            for phase in ['train', 'validation']:

                if phase == 'train':
                    mdl.train()  # training mode
                else:
                    mdl.eval()   # evaluate mode

                # Loop through the data in each batch
                for inputs, labels in loaders[phase]:

                    if not is_finished:

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optim.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):

                            # Forward pass
                            outputs = mdl(inputs)
                            loss = crit(outputs, labels.unsqueeze(1).float())

                            # Backward pass
                            if phase == 'train':
                                loss.backward()
                                optim.step()

                        # Capturing values for training metrics
                        total_acc += (outputs.round() == labels.unsqueeze(1)).sum().item()
                        total_count += labels.size(0)
                        epoch_accuracy = total_acc / total_count
                        epoch_loss = loss.item()

                print('{} loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

                if phase == 'validation':

                    # Check to see if model failed to improve significantly since last epoch
                    is_acc_less = False
                    is_acc_not_much_better = False
                    is_loss_worse = False

                    if epoch_accuracy <= best_acc:
                        is_acc_less = True
                        print("\nValidation accuracy for this epoch is less than or equal to best accuracy.")

                    if epoch_accuracy - best_acc < 0.005:
                        print("Best validation accuracy: " + str(best_acc))
                        is_acc_not_much_better = True
                        print("\nValidation accuracy did not significantly improve in this epoch")

                    if epoch_loss > best_loss:
                        print("Best validation loss: " + str(best_loss))
                        is_loss_worse = True
                        print("\nValidation loss for this epoch is higher than best loss.")

                    if epoch > 0 and (is_acc_less and is_loss_worse) or \
                            (not is_acc_less and is_acc_not_much_better and is_loss_worse):
                        print("\nEarly stopped training at epoch %d" % epoch)

                        resume_epoch = epoch  # will print the correct epoch from an ordinal standpoint (1st, 2nd, etc)
                        resume(mdl, optim, f"_epoch_{resume_epoch}.pt")

                        is_finished = True

                        train_acc_history.pop(len(train_acc_history) - 1)
                        train_loss_history.pop(len(train_loss_history) - 1)

                        break  # exit the current loop and return to the is_finished condition check

                    if epoch_accuracy > best_acc:
                        best_acc = epoch_accuracy
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                    if best_acc == epoch_accuracy or best_loss == epoch_loss:
                        checkpoint(mdl, optim, f"_epoch_{epoch+1}.pt")

                    validation_acc_history.append(epoch_accuracy)
                    validation_loss_history.append(epoch_loss)

                if phase == 'train' and not is_finished:
                    train_acc_history.append(epoch_accuracy)
                    train_loss_history.append(epoch_loss)

            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))

    history_dict = {'train_loss': train_loss_history, 'train_accuracy': train_acc_history,
                    'validation_loss': validation_loss_history, 'validation_accuracy': validation_acc_history}

    torch.save(mdl.state_dict(), _model_save_path + ".pt")
    torch.save(optim.state_dict(), _optimizer_save_path + ".pt")

    return mdl, history_dict


history = train_model(model, dataloaders, criterion, optimizer, num_epochs)

plt.figure(figsize=(17, 5))
plt.subplot(121)
plt.plot(history[1]['train_loss'], '-o')
plt.plot(history[1]['validation_loss'], '-o')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Validation loss')

plt.subplot(122)
plt.plot(history[1]['train_accuracy'], '-o')
plt.plot(history[1]['validation_accuracy'], '-o')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Validation Accuracy')

plt.show()

exit()
