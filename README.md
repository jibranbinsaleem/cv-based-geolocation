# Computer Vision based Geolocation

This project implements a computer vision-based geolocation system that utilizes deep learning techniques and transfer learning with the ResNet architecture. The system aims to accurately predict the longitude and latitude coordinates of a location based on input images, providing a powerful tool for geolocation and mapping applications.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributors](#contributors)

## Introduction

Geolocation plays a crucial role in various domains, including navigation, urban planning, and location-based services. Traditional methods rely on GPS coordinates or address-based information, but they may not always be available or accurate. By leveraging the power of computer vision and transfer learning, this system offers an alternative approach that can automatically extract location information from visual data.

## Dataset

The dataset used for training the model consists of labeled images of various buildings within the University of Salford. The dataset can be downloaded from [here](https://figshare.com/articles/dataset/UoS_Buildings_Image_Dataset_for_Computer_Vision_Algorithms/20383155).

## Model Architecture

The model architecture used for computer vision based geolocation is transfer learning. We Mobile net and fine tuned by adding a global average pooling layer and a fully connected layer with the desired number of output units

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/cv-based-geolocation.git
   ```
2. Install the required dependencies

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Access the web application in your browser at `http://localhost:5000/home`.

## Evaluation

The model is evaluated on the test dataset and it returned us a good Accurracy score of **94%**. These metrics provide an assessment of the model's performance in correctly classifying bot and non-bot accounts. Flask app is built to test for real world data.


## Contributors

- [Muhammad Jibran Bin Saleem](https://github.com/jibranbinsaleem)
- [Ahmad Faraz Sheikh](https://github.com/FarazSheikh16)
