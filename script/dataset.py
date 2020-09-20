#import the required packages
import glob
import numpy as np
import cv2


img_size = 150
#A function that exastract the content of a folder according to the structure of our dataset 
def get_data_from_dir(path):
	list = []
	paths = [path+ 'PNEUMONIA/*.jpeg',path + 'NORMAL/*.jpeg']
	for j in paths:
		dir_content = glob.glob(j)
		index = paths.index(j)
		for i in dir_content:
			img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
			resized = cv2.resize(img,(150,150))
			list.append([resized,index])
	return np.array(list, dtype=object)

#A function that help us to
# -load data 
# -normilize items
# -convert to 4-D tensor 
#Return the labels and features
def prepare_dataset():
	#load the samples located in the folder train
	train_data_dir = 'dataset/train/'
	train = get_data_from_dir(train_data_dir)
	print("done")
	#load the samples located in the folder test
	test_data_dir = 'dataset/test/'
	test = get_data_from_dir(test_data_dir)
	print("done")
	#load the samples located in the folder validation
	validation_data_dir = 'dataset/val/'
	val = get_data_from_dir(validation_data_dir)
	print("done")
	x_train, y_train, x_test, y_test, x_val, y_val = [], [], [], [], [], []
	#extract features and labels
	for feature, label in train:
		x_train.append(feature)
		y_train.append(label)
	for feature, label in test:
		x_test.append(feature)
		y_test.append(label)
	for feature, label in val:
		x_val.append(feature)
		y_val.append(label)
	# Normalize the data
	x_train = np.array(x_train) / 255
	x_val = np.array(x_val) / 255
	x_test = np.array(x_test) / 255
	# Convert the input images to 4-D tensors
	x_train = x_train.reshape(-1, img_size, img_size, 1)
	y_train = np.array(y_train)
	x_val = x_val.reshape(-1, img_size, img_size, 1)
	y_val = np.array(y_val)
	x_test = x_test.reshape(-1, img_size, img_size, 1)
	y_test = np.array(y_test)
	
	return [x_train, y_train, x_test, y_test, x_val, y_val]

prepare_dataset()
