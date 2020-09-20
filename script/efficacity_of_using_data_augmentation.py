#Please to find the chart of this script in charts\comparison\efficacity_of_using_data_augmentation.PNG

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.metrics import categorical_crossentropy
from keras.layers.normalization  import BatchNormalization
from keras.layers.convolutional import *
from tensorflow.python.keras.models import load_model, model_from_json
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import time
import numpy as np
import dataset
import utils

#extract labels and features
list = dataset.prepare_dataset()
x_train = list[0]
y_train = list[1]
x_test = list[2]
y_test = list[3]
x_val = list[4]
y_val = list[5]

#apply data augumentation
datagen = ImageDataGenerator(
	rotation_range = 30,  
	zoom_range = 0.2, 
	width_shift_range=0.1,  
	height_shift_range=0.1,  
)

#principle program
#create a list to save the results of the differet models 
nets = 2
list = [0]*nets
#create a list of models
model = [0]*nets
#epochs
e=10
#use or not data augumentation
data_aug = ['data augmentation', 'without data augmentation'] 
#The first loop changes the number of filters per convolution layer
j = 0
test = False
for opt in data_aug:
	#initialize the model j
	model[j] = Sequential()
	#the first convolution layer
	model[j].add(Conv2D(8 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
	model[j].add(BatchNormalization())
	model[j].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
	#the second convolution layer
	model[j].add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
	model[j].add(BatchNormalization())
	model[j].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
	#the third conv layer
	model[j].add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
	model[j].add(BatchNormalization())
	model[j].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
	#the forth convolution layer
	model[j].add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
	model[j].add(BatchNormalization())
	model[j].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
	#the fifth convolution layer
	model[j].add(Conv2D(128, (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
	model[j].add(BatchNormalization())
	model[j].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
	#the sixth convolution layer
	model[j].add(Conv2D(128, (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
	model[j].add(BatchNormalization())
	model[j].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
	#add flatten layer
	model[j].add(Flatten())
	#add fully-connected layer
	model[j].add(Dense(units = 256 , activation = 'relu'))
	#the output layer (binary classification) one neuron is enough
	model[j].add(Dense(units = 1 , activation = 'sigmoid'))
	#compile the model
	model[j].compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
	#initialize the name of the model
	name = data_aug[j]
	#clear the session before retrain a model 10 times
	tf.keras.backend.clear_session()
	#retrain the model i 10 times
	if name == 'data augmentation':
		datagen.fit(x_train)
		test = True
	result = utils.trainFcn(model[j],name, e, datagen, x_train, y_train, x_test, y_test, x_val, y_val,32, test)
	#save the results of the training process in list
	list[j] = [name,result,e]
	#test with another optimizer
	j += 1
#print the results of the different models
legend_names = []
j=0
for l in list:
	name = l[0]
	k = l[1]
	epoch = [i for i in range(e)]
	print("results of ", data_aug[j])
	print(str(k[0]) +" | "+ str(k[1])+" | "+ str(k[2])+" | "+ str(k[3])+" | "+str(k[4])+" | "+ str(k[5])+ " | "+ str(k[6]))
	print("max train accuracy: ", max(k[7].history['accuracy']))
	print("max train validation: ", max(k[7].history['val_accuracy']))
	plt.plot(epoch, k[7].history['val_accuracy'])
	legend_names.append(data_aug[j])
	j +=1
utils.plot_muliplt_line_chart(plt, 'efficacity_of_using_data_augmentation', legend_names)