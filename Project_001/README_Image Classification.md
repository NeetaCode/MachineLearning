Deep Learning Models for Image Classification
This repository contains several implementations of image classification models using TensorFlow and Keras on popular datasets such as MNIST, CIFAR-10, and Imagenette. Models include pre-trained architectures like VGG16 and ResNet50 as well as custom CNNs. Both pre trained weights and training from scratch are demonstrated.

Table of Contents
•	Overview
•	Datasets
•	Models Implemented
•	Setup and Requirements
•	How to Run
•	Results
•	Acknowledgments

Overview
This repository demonstrates:
•	Loading and visualizing datasets (MNIST, CIFAR-10, Imagenette).
•	Fine-tuning pre-trained models (VGG16, ResNet50) on these datasets.
•	Training models from scratch without pre trained weights.
•	Custom CNN architecture for Imagenette dataset.
•	Use of data augmentation to improve generalization.
•	Plotting training history (accuracy and loss).

Datasets
•	MNIST: Handwritten digits dataset (grayscale, 28x28).
•	CIFAR-10: 10-class natural image dataset (colour, 32x32).
•	Imagenette: Subset of ImageNet dataset with 10 classes.
•	TensorFlow Datasets (TFDS) is used to load Imagenette.

Models Implemented
1.	VGG16
•	Pre trained on ImageNet and fine-tuned on MNIST and CIFAR-10.
•	Training from scratch (no pre trained weights) on MNIST and CIFAR-10.
•	Input images resized and normalized to fit VGG16 input requirements.

2.	ResNet50
•	Pre trained on ImageNet and fine-tuned on CIFAR-10 and Imagenette.
•	Training from scratch (no pre trained weights) on CIFAR-10 and Imagenette.
•	Includes data augmentation for CIFAR-10 to improve robustness.

3.	Custom CNN (for Imagenette)
•	Simple CNN architecture built with Conv2D, MaxPooling2D, Dense, and Dropout layers.
•	Demonstrates training on Imagenette dataset with image pre-processing.

Setup and Requirements
•	Python 3.7 or higher
•	TensorFlow 2.x
•	TensorFlow Datasets (tensorflow-datasets)
•	OpenCV (opencv-python) for image resizing and pre-processing
•	Matplotlib for plotting

Install dependencies with pip:
pip install tensorflow tensorflow-datasets opencv-python matplotlib

How to Run
1.	Clone the repository.
2.	Ensure all dependencies are installed.
3.	Run the desired script or notebook cell for each model/dataset.
4.	Visualizations of sample images and training history plots will be displayed.
5.	Pre trained models save checkpoints (where applicable) for best validation accuracy.

Results
•	Training and validation accuracy/loss plots for each model.
•	Visualizations of sample dataset images.
•	Fine-tuned models show improved accuracy over training from scratch.

Notes
•	Input image sizes are adjusted to fit model input requirements (e.g., 48x48 or 224x224).
•	CIFAR-10 images are normalized to [0, 1].
•	Data augmentation is applied for CIFAR-10 when training ResNet50.
•	Use early stopping and model check pointing to prevent overfitting.

Acknowledgments
•	Pre trained model weights provided by Keras Applications.
•	Dataset loading through TensorFlow Datasets.
•	Inspired by standard deep learning tutorials on image classification.
