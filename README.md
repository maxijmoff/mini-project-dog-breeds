# Mini Project Dog Breeds Image Classification

## What are you making?

This repository contains a deep learning project that classifies dog breeds from images. The model takes an input image of a dog and predicts which breed the dog belongs to, enabling accurate identification based on visual features.

## What architecture do you use?

The classification model is built using a convolutional neural network (CNN) architecture. The model consists of multiple convolutional blocks followed by batch normalization and max pooling layers. The architecture is designed to extract features from images effectively and includes the following layers:

1. **Convolutional Blocks**: 
   - Each block consists of two convolutional layers followed by batch normalization and a max pooling layer.
   - The model has five convolutional blocks with increasing filter sizes (32, 64, 128, 256, 512).

2. **Global Average Pooling**: 
   - This layer reduces the spatial dimensions of the feature maps to a single value per feature map.

3. **Fully Connected Layers**: 
   - A dense layer with 512 units and ReLU activation.
   - A dropout layer with a rate of 0.4 to prevent overfitting.
   - The output layer with softmax activation for multi-class classification (70 classes for different dog breeds).

## What library do you use?

- `tensorflow` for model building, training, and evaluation.
- `numpy` for numerical operations.
- `pandas` for data manipulation and analysis.
- `matplotlib` for data visualization.
- `scikit-learn` for model evaluation and metrics.
- `tqdm` for progress bars in loops.
- `tensorflow_datasets` for easy access to datasets.
- `gdown` for downloading datasets from Google Drive.
- `zipfile` for handling zip files.

## How to run your model

## Dataset References

## Paper Research References

1. [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)  

3. [Dogs Breed Classification Using Deep Learning](https://ijcrt.org/papers/IJCRT2104679.pdf)  
