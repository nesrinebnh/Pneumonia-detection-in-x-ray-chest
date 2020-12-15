# Convolution-Neural-Network-with-keras-and-django

## :writing_hand: Description

Pneumonia is an infection induced by diverse organisms. The seriousness of the disease depends on multiple factors:
* Clinical symptoms like fatigue, chest pain, shortness of breath. 
* Patient background like a weakened immune system. 

Doctors diagnose pneumonia by combining the results of clinical examination with the outcomes of the x-ray chest that should reveal areas of opacity. Our aim in this project is to create an application that uses an AI model, which takes an x-ray chest and classify it whether it concerns pneumonia or not. To do so, we will:
1. first download our dataset, and preprocess it. 
2. Then, implement several architectures. 
3. Finally, we will choose the best according to metrics.
  
## Project structure

### Application

This folder contains the source code of the application named inTouch, which is a web application build with the Django framework. It receives as an input the x-ray chest and returns the diagnosis.

mini demo :eyes: : https://cnn-django-intouch.herokuapp.com/ 

### Models

This folder contains all the hyperparameters' configurations tested in this project.

### Scripts

A comparative folder that allows to compare between the different hyperparameters and models to pick the best one.

## :zap: Usage

You can use this project as a reference if you wish to work with binary image classification and Convolution Neural Network.


