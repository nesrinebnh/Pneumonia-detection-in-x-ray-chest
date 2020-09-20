#Please to find the different charts of this script in the folder charts\LeNET. LeNET contains subfolders that contains subfolders.
#Each subfolder refers to a parameter the first folder the number of filters in each conv layer
#the second subfolder: the number of units in the fully-connected layers
#the third sub folder: the epochs
#then we can find the best variation of accuracy and loss in the validation and train data for that configuration.
# we also saved the confusion matrix 

#import the necessary packages
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

#load labels and features
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
datagen.fit(x_train)

#principle program
#create a list to save the results of the differet models 
nets = 4
list = [0]*64
#create a list of models
model = [0]*64
m=0
#The first loop changes the number of filters per convolution layer
for j in range(nets): #4 iteration
	#The second loop concerns the neurons of the fully-connected layer
	for k in range(4):# 4iteration
		for e in range(5,21,5): #4 iteration 4*4*4=64
			#initialize the model i
			model[j] = Sequential()
			#the first convolution layer
			c = 8*(j+1)#c is the filter of the first convolution layer (8,16,24,32)
			model[m].add(Conv2D(c , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
			model[m].add(BatchNormalization())
			model[m].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
			#the second convolution layer
			c1=16*(j+1)#c1 is the filter of the 2nd conv layer (16,32,48,64)
			model[m].add(Conv2D(c1 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
			model[m].add(BatchNormalization())
			model[m].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
			#the third conv layer
			c2=32*(j+1) #(32,64,96,128)
			model[m].add(Conv2D(c2 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
			model[m].add(BatchNormalization())
			model[m].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
			#the firth convolution layer
			c3=64*(j+1)#(64,128,192,256)
			model[m].add(Conv2D(c3 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
			model[m].add(BatchNormalization())
			model[m].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
			#the fifth convolution layer
			c4=128*(j+1)#(128,256,384,512)
			model[m].add(Conv2D(c4, (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
			model[m].add(BatchNormalization())
			model[m].add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
			#add flatten layer
			model[m].add(Flatten())
			#add fully-connected layer
			h1 = 64*pow(2,k)#number of neurons in the fully-connected layer 64-128-256-512
			model[m].add(Dense(units = h1 , activation = 'relu'))
			#the output layer (binary classification) one neuron is enough
			model[m].add(Dense(units = 1 , activation = 'sigmoid'))
			#compile the model
			opt = 'adam'#with the previous experimental, we found that adam performs better
			model[m].compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])
			#initialize the name of the model
			name = str(c)+"-"+str(c1)+"-"+str(c2)+"-"+str(c3)+"-"+str(c4)+" "+str(h1)+" "+str(e)+" "+str(opt)
			#clear the session before retrain a model 10 times
			tf.keras.backend.clear_session()
			#retrain the model i 10 times
			result = utils.trainFcn(model[m],name, e, datagen, x_train, y_train, x_test, y_test, x_val, y_val,32, True)
			#save the results of the training process in list
			list[j] = [name,result,e]
#print the results of the different models
print("Print the results of the different models")
for res in list:
	name = res[0]
	k = res[1]
	epoch = res[2]
	print("name => avg acc | avg loss | best acc | worst acc | best loss | worst loss | time")
	print(name+" => "+str(k[0]) +" | "+ str(k[1])+" | "+ str(k[2])+" | "+ str(k[3])+" | "+str(k[4])+" | "+ str(k[5])+ " | "+ str(k[6]))
	#plot the best model
	epochs = [i for i in range(epoch)]
	train_acc = k[7].history['accuracy']
	train_loss = k[7].history['loss']
	val_acc = k[7].history['val_accuracy']
	val_loss = k[7].history['val_loss']
	utils.plot_best_validation_loss_accuracy(train_acc, val_acc, epochs, train_loss, val_loss, name)
	#load the best model
	best_model_name = "best_model"+name
	model = load_model("{}.h5".format(best_model_name))
	#make predictions for confusion matrix
	predictions = model.predict_classes(x_test)
	predictions = predictions.reshape(1,-1)[0]
	#precision, recall, f1-score
	print(classification_report(y_test, predictions, target_names = ['Pneumonia','Normal']))
	#confusion matrix
	utils.plot_confusion_matrix(y_test, predictions,  best_model_name)
	#load the worst model
	worst_model_name = "worst_model"+name
	model = load_model("{}.h5".format(worst_model_name))
	#make predictions for confusion matrix
	predictions = model.predict_classes(x_test)
	predictions = predictions.reshape(1,-1)[0]
	#confusion matrix
	utils.plot_confusion_matrix(y_test, predictions,  worst_model_name)
	
	
# the comparison  models is done manually to better understand and analyze the results. 

