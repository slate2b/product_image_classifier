# Product Image Classifier

A collection of scripts used to create, train, and fine-tune ML models for a very specific use case -- identifying 
images which match a given base-set of product images.  

The first step in the end-to-end process is to determine whether or not an input image comes from the base-image 
catalog.  I found that binary classifiers performed this task exceedingly well.

After training and testing multiple architectures and configurations for the binary classification task, I found that a 
fine-tuned ResNet50 model and a custom CNN model performed best, with the custom CNN slightly outperforming the ResNet50
model. 