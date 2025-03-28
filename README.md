# Product Image Classifier

A collection of scripts used to create, train, and fine-tune ML models for a very specific use case -- identifying 
images which match a given base-set of product images.  

The first step in the end-to-end process is to determine whether or not an input image comes from the base-image 
catalog.  I found that binary classifiers performed this task exceedingly well.

After training and testing multiple architectures and configurations for the binary classification task, I found that a 
fine-tuned ResNet50 model and a custom CNN model performed best, with the custom CNN slightly outperforming the ResNet50
model. 

## Project Insights

* For highly specialized use cases where accuracy is a top priority, a focused solution can perform better than a general
solution.  
* Smaller models which are built and trained from the ground up can be more effective for specific use cases than more 
complex models which are designed for a broader scope.  

## Model Details

### CNN Binary Image Classifier

Simple CNN model consisting of 3 CNN layers and 2 Linear layers.  I performed tests using MaxPooling, but the model 
performed better for this use case without any MaxPooling layers.  I used a sigmoid activation function on the output, 
I used the basic SGD optimizer during training.

* Total Params: 25,167,785
* Size on Disk: 0.098 GB

Input images are resized using torchvision transforms Resize() method to a size of (64, 64).

* Conv2d kernel size: 3
* Padding: 1

The images used for training consisted of a 185000 image train set and a 30000 image validation set.  The 2 classes
(matching and non-matching) were evenly represented within both the train and validation sets.  I used augmentation to 
generate thousands of images based off the original base-set of product images.  The augmented images varied slightly 
the originals in order to teach the model to recognize images which were slightly modified from the original since the 
business requirements stated that we needed to find images even if they were compressed, resized, or cropped.

The model converged nicely after only 5 epochs of training.  

* Validation Accuracy: 99.96%
* Validation Loss: 0.0004

![binary_image_classifier_cnn_01_1_TrainingLossAccuracyPlot.png](..%2Fproduct_image_classification%2Fadlife_known_models%2Fbinary_image_classifier_cnn_01_1_TrainingLossAccuracyPlot.png)


