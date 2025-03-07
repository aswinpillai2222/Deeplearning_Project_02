Convolutional Neural Network (CNN) for Image Classification

Overview

This project implements a Convolutional Neural Network (CNN) for image classification using the MNIST dataset. The model is trained with data augmentation to improve generalization and evaluated using common performance metrics.

Features

Loads and preprocesses the MNIST dataset (handwritten digits 0-9)

Applies data augmentation to increase dataset diversity

Defines and trains a CNN model with convolutional, pooling, and fully connected layers

Evaluates the model using metrics such as accuracy, precision, recall, and F1-score

Uses TensorBoard for visualization of training performance

Dataset

The MNIST dataset consists of:

60,000 training images

10,000 test images

Images are grayscale, 28×28 pixels, representing digits 0-9

Installation

To run this project, install the required dependencies:

pip install tensorflow numpy matplotlib

Model Architecture

The CNN model consists of the following layers:

Convolutional Layer (Conv2D) - Extracts features using 3×3 filters

Activation Function (ReLU) - Introduces non-linearity

Pooling Layer (MaxPooling2D) - Reduces spatial dimensions

Fully Connected Layer (Dense) - Learns high-level patterns

Softmax Layer - Outputs class probabilities

Data Augmentation

To improve model generalization, the following data augmentation techniques are applied:

Rotation: Rotates images by ±30 degrees

Flipping: Randomly flips images horizontally

Shifting: Moves images left/right or up/down

Zooming: Randomly zooms in/out

Brightness Adjustment: Varies brightness levels

Training the Model

The model is trained using categorical cross-entropy loss and the SGD optimizer:

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

Evaluation Metrics

The CNN model is evaluated using:

Accuracy: Measures overall correctness of predictions

Precision: Measures how many predicted positives are actually correct

Recall: Measures how well the model detects true positives

F1-Score: Balances precision and recall

Confusion Matrix: Visualizes classification performance

Running the Model

Run the script to train and evaluate the CNN:

python cnn_train.py

To visualize training performance with TensorBoard:

tensorboard --logdir=logs/mnist_experiment

Results

After training, the model achieves high accuracy on test data, demonstrating strong generalization. Data augmentation significantly improves performance on unseen images.

Next Steps

Experiment with different architectures (e.g., deeper CNNs, ResNet)

Use transfer learning with pre-trained models

Apply CNNs to more complex datasets (e.g., CIFAR-10, ImageNet)

License

This project is open-source under the MIT License.
