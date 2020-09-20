#Please to find the graph of this script in charts\comparison\comparing_the_variation_of_validation_accuracy_in_function_of_epochs_by_using_different_kernel_size.PNG

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten  , BatchNormalization
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


net = 4
list = [0]*net

model = [0]*net

optimizer = [(2,2), (3,3)]
max = [(2,2), (3,3)]
size = 2
e=10
k = 0
for i in range(size):
	for j in range(size):
		model[k] = Sequential()
		c = 8
		model[k].add(Conv2D(c , optimizer[i] , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
		model[k].add(BatchNormalization())
		model[k].add(MaxPool2D(max[j] , strides = 2 , padding = 'same'))

		c1=16
		model[k].add(Conv2D(c1 , optimizer[i] , strides = 1 , padding = 'same' , activation = 'relu'))
		model[k].add(BatchNormalization())
		model[k].add(MaxPool2D(max[j] , strides = 2 , padding = 'same'))

		c2=32
		model[k].add(Conv2D(c2 , optimizer[i] , strides = 1 , padding = 'same' , activation = 'relu'))
		model[k].add(BatchNormalization())
		model[k].add(MaxPool2D(max[j] , strides = 2 , padding = 'same'))

		c3=64
		model[k].add(Conv2D(c3 , optimizer[i] , strides = 1 , padding = 'same' , activation = 'relu'))
		model[k].add(BatchNormalization())
		model[k].add(MaxPool2D(max[j] , strides = 2 , padding = 'same'))

		c4=128
		model[k].add(Conv2D(c4, optimizer[i] , strides = 1 , padding = 'same' , activation = 'relu'))
		model[k].add(BatchNormalization())
		model[k].add(MaxPool2D(max[j] , strides = 2 , padding = 'same'))

		model[k].add(Flatten())
		h1 = 256
		model[k].add(Dense(units = h1 , activation = 'relu'))
		model[k].add(Dense(units = 1 , activation = 'sigmoid'))
		opt = 'adam'
		model[k].compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])
		name = str(c)+"-"+str(c1)+"-"+str(c2)+"-"+str(c3)+"-"+str(c4)+" "+str(h1)+" 20 "+str(opt)+" "+str(opt)+" "+str(optimizer[i])+str(max[j])
		tf.keras.backend.clear_session()
		result = utils.trainFcn(model[k],name, e, datagen, x_train, y_train, x_test, y_test, x_val, y_val,32, True)
		list[k] = [name,result,e]
		k+=1
#print the results of the different models
legend_names = []
j=0
optimizers=['(2,2),(2,2)' , '(2,2),(3,3)', '(3,3),(2,2)', '(3,3),(3,3)']
for l in list:
	name = l[0]
	k = l[1]
	
	epoch = [i for i in range(e)]
	print("results of ", optimizers[j])
	print(str(k[0]) +" | "+ str(k[1])+" | "+ str(k[2])+" | "+ str(k[3])+" | "+str(k[4])+" | "+ str(k[5])+ " | "+ str(k[6]))
	plt.plot(epoch, k[7].history['val_accuracy'])
	legend_names.append(optimizers[j])
	j +=1
utils.plot_muliplt_line_chart(plt, 'comparing_the_variation_of_validation_accuracy_in_function_of_epochs_by_using_different_kernels', legend_names)