# Convolutional Neural Network for Image Recognition

## Introduction
This code demonstrates the implementation of a Convolutional Neural Network (CNN) for image recognition using the CIFAR-10 dataset with TensorFlow/Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes represent various objects such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Code Structure

### 1. Import Libraries
The required libraries for building and training the CNN model are imported, including TensorFlow, Keras layers and models, and Matplotlib for data visualization.

### 2. Load and Preprocess Data
The CIFAR-10 dataset is loaded and split into training and test sets. The pixel values of the images are normalized to the range [0, 1], and the class labels are converted to binary class matrices compatible with the categorical cross-entropy loss function.

### 3. Build the CNN Model
A sequential CNN model is constructed using the Keras API. The model architecture consists of the following layers:

1. Convolutional layer with 32 filters, 3x3 kernel, and ReLU activation
2. Max-pooling layer with 2x2 pool size
3. Convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
4. Max-pooling layer with 2x2 pool size
5. Convolutional layer with 63 filters, 3x3 kernel, and ReLU activation
6. Flattening layer
7. Dense layer with 64 units and ReLU activation
8. Output dense layer with 10 units and softmax activation for multi-class classification

### 4. Compile the Model
The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy metric for evaluation.

### 5. Train the Model
The model is trained for 10 epochs using the training data and labels. The validation data is used for monitoring the model's performance during training.

### 6. Evaluate the Model
After training, the model's performance is evaluated on the test data, and the test loss and accuracy are reported.

### 7. Visualize Training History
The training and validation accuracy and loss curves are visualized using Matplotlib, providing insights into the model's performance over the training epochs.

### 8. Visualize Predictions
As an additional step, the model's predictions on the test data are obtained, and the predicted labels are compared with the true labels. A figure with 10 subplots is created, displaying test images along with their true and predicted labels, allowing for visual inspection of the model's performance.

## Conclusion
This code demonstrates the implementation of a CNN for image recognition using the CIFAR-10 dataset. It covers the essential steps of loading and preprocessing data, building the model architecture, training the model, evaluating its performance, and visualizing the results. This code can serve as a starting point for further experimentation and improvement of the CNN model for image classification tasks.
