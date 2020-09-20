import tensorflow as tf
import keras
import time
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

#trainFcn is a function that retrain a model ten times and return different results concerning the training process
def trainFcn(model,name, epoch, datagen, x_train, y_train, x_test, y_test, x_val, y_val, batch_s, test):
	#initialize some useful variables
	somme_accuracy, somme_loss, moy_time = 0, 0, 0
	#train the model for the first time
	n = 1
	i=0
	history = [0]*n
	start_time = time.time()
	#train the model and save the result in history[i]
	if test:
		history[i] =   model.fit(datagen.flow(x_train,y_train, batch_size = batch_s) ,epochs = epoch , validation_data = datagen.flow(x_val, y_val))
	else:
		history[i] =   model.fit(x_train,y_train, batch_size = batch_s ,epochs = epoch , validation_data = (x_val, y_val))
	#initializr the parameters to return later on
	best_history = history[i]
	best_loss =   model.evaluate(x_test,y_test)[0]
	worst_loss =   model.evaluate(x_test,y_test)[0]
	best_accuracy =   model.evaluate(x_test,y_test)[1]*100
	worst_accuracy =   model.evaluate(x_test,y_test)[1]*100
	#save the model of the first iteration in case it is the best or the worst
	model.save("worst_model"+name+".h5")
	model.save("best_model"+name+".h5")
	#start measuring the avg loss and avg accuracy and time
	somme_loss += best_loss
	somme_accuracy += best_accuracy
	moy_time += time.time() - start_time
	#retrain the model for 9 times that rest
	i=1
	while (i<n):
		start_time = time.time()
		if test:
			#train the model and store each the results in history[i]
			history[i] =   model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = epoch , validation_data = datagen.flow(x_val, y_val))
		else:
			history[i] =   model.fit(x_train,y_train, batch_size = 32 ,epochs = epoch ,validation_data = (x_val, y_val))
		#evaluate the performance of the model on the unseen data
		actual_loss =   model.evaluate(x_test,y_test)[0]
		actual_accuracy =   model.evaluate(x_test,y_test)[1]*100
		#keep measuring the avg loss, accuracy, and time
		somme_loss += actual_loss
		somme_accuracy += actual_accuracy
		moy_time += time.time() - start_time
		#save the best model and best accuracy according to the accuracy measured in line 34
		#save the result of the best model to plot it later
		if actual_accuracy > best_accuracy:
			best_accuracy = actual_accuracy
			best_history = history[i]
			model.save("best_model"+name+".h5")
		#save the worst model and worst accuracy
		if actual_accuracy < worst_accuracy:
			worst_accuracy = actual_accuracy
			model.save("worst_model"+name+".h5")
		#save the best loss according to the loss measured in line 33
		if actual_loss < best_loss:
			best_loss = actual_loss
		#save the worst loss 
		if actual_loss > worst_loss:
			worst_loss = actual_loss

		#retrain again
		i+=1
	#measure the avg of accuracy, loss and time
	moy_accuracy = somme_accuracy /n
	moy_loss = somme_loss/n
	moy_time /= n
	#return the all the results in an array
	result = [moy_accuracy, moy_loss, best_accuracy, worst_accuracy, best_loss, worst_loss, moy_time, best_history]
	return result
	
def plot_best_validation_loss_accuracy(train_acc, val_acc, epochs, train_loss, val_loss, name):
	fig , ax = plt.subplots(1,2)
	fig.set_size_inches(20,10)
	ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
	ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
	ax[0].set_title('Training & Validation Accuracy')
	ax[0].legend()
	ax[0].set_xlabel("Epochs")
	ax[0].set_ylabel("Accuracy")
	ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
	ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
	ax[1].set_title('Training & Validation Loss')
	ax[1].legend()
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("Loss")
	plot_name = 'validation_loss'+name
	plt.savefig('{}.png'.format(plot_name))

def plot_muliplt_line_chart(plt, name, legend_names):
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(legend_names, loc='upper left')
	plt.gca().set_ylim([0,1])
	plt.savefig('{}.png'.format(name))
	
def plot_confusion_matrix(y_test, predictions, name):
	cm = confusion_matrix(y_test,predictions)
	cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
	plt.figure(figsize = (10,10))
	labels = ['PNEUMONIA', 'NORMAL']
	sns.heatmap(cm,cmap= "Accent", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
	#save the confusion matrix
	plt.savefig('{}.png'.format(name))
  