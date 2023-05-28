# Computer Vision-based Geolocation

This project implements a computer vision-based geolocation system that utilizes deep learning techniques and transfer learning with the ResNet architecture. The system extracts longitude and latitude coordinates from images and opens the corresponding location on Google Maps.

## Dataset

The dataset used for training the model consists of labeled images of various buildings within the University of Salford. The dataset can be downloaded from [here](https://figshare.com/articles/dataset/UoS_Buildings_Image_Dataset_for_Computer_Vision_Algorithms/20383155).

## System Architecture

The system follows the following architecture:

1. **Data Collection**: Gathered a labeled dataset of University of Salford building images and associated longitude and latitude coordinates.

2. **Preprocessing**: Prepared the dataset by resizing images, normalizing pixel values, and splitting into training and validation sets.

3. **Transfer Learning**: Used transfer learning with the pre-trained ResNet model to extract features from the images.

4. **Model Training**: Trained the modified ResNet model to predict longitude and latitude coordinates.

5. **Inference**: Deployed the trained model to predict coordinates for input images.

6. **Google Maps Integration**: Opened the predicted location on Google Maps based on the coordinates.


## Conclusion

The computer vision-based geolocation system demonstrates accurate prediction of longitude and latitude coordinates from images. By integrating with Google Maps, it provides a user-friendly geolocation experience. The system has potential applications in navigation, landmark identification, and location-based services. Further improvements can be made by incorporating additional data sources, refining the training process, and exploring advanced computer vision algorithms.


