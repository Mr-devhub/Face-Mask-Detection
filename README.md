# Face Mask Detection Using Machine Learning
This project aims to detect whether a person is wearing a face mask or not using machine learning techniques. The model is built using Python programming language, OpenCV, and TensorFlow.

Dataset
The dataset used in this project is a collection of images of people with and without face masks. The dataset contains approximately 6,000 images and is divided into two classes: with mask and without mask.

Model Architecture
The model architecture used in this project is a convolutional neural network (CNN) with four convolutional layers, followed by two fully connected layers. The model is trained using the Adam optimizer and binary cross-entropy loss function.

Getting Started
To get started with this project, follow these steps:

Clone the repository to your local machine.
Install the required libraries by running pip install -r requirements.txt.
Download the dataset and place it in the data folder.
Run train.py to train the model.
Run detect.py to detect face masks in images or videos.
Results
The model achieved an accuracy of 97% on the test dataset. The model is capable of detecting face masks in real-time on images or videos.

Future Improvements
The model can be further improved by using a larger dataset.
The model can be trained to detect different types of face masks, such as N95 masks or surgical masks.
The model can be deployed on a web application or a mobile application.

Credits
The dataset used in this project is from https://www.kaggle.com/datasets/omkargurav/face-mask-dataset.
The face detection algorithm is based on the work of https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector.
License
This project is licensed under the MIT License. See the LICENSE file for details.
