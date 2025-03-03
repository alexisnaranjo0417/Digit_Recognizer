
# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

# Step 1: Import all required keras libraries

from keras.datasets import mnist # This is used to load mnist dataset later
from keras.utils import np_utils # This will be used to convert your test image to a categorical class (digit from 0 to 9)
import tensorflow as tf

#Sets the random seed for tensor flow to 0 to get reproducible results from the model.
tf.random.set_seed(0)

# Step 2: Load and return training and test datasets
def load_dataset():
	# 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
	#Loaded the train test with the mnist data.
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# 2b. reshape for X train and test vars - Hint: X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	#Reshaped the X train and X test variables.
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
 
	# 2c. normalize inputs from 0-255 to 0-1 - Hint: X_train = X_train / 255
	#Normalized the X train and X test variables.
	X_train = X_train / 255
	X_test = X_test / 255
 
	# 2d. Convert y_train and y_test to categorical classes - Hint: y_train = np_utils.to_categorical(y_train)
	#Converted the y train and the y test classes to catagorical classes.
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	# 2e. return your X_train, X_test, y_train, y_test
	#Returned X train, X test, y train, and y test.
	return X_train, X_test, y_train, y_test

# Step 3: define your CNN model here in this function and then later use this function to create your model
def digit_recognition_cnn():
	# 3a. create your CNN model here with Conv + ReLU + Flatten + Dense layers
	#Intializes the model
	cnn = tf.keras.models.Sequential()
 
	#Convolution with 30 filters
	cnn.add(tf.keras.layers.Conv2D(filters=30, kernel_size=3, activation='relu', input_shape=[28,28,1]))
 
	#Pooling
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
 
	#Convolution with 30 layers
	cnn.add(tf.keras.layers.Conv2D(filters=30, kernel_size=3, activation='relu', input_shape=[28,28,1]))
 
	#Pooling
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
 
	#Flattening
	cnn.add(tf.keras.layers.Flatten())
 
	#128 Dense layers
	cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
 
	#10 Dense layers
	cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))
 
	# 3b. Compile your model with categorical_crossentropy (loss), adam optimizer and accuracy as a metric
	#Compiles the model to show how the machine is performing during the training.
	cnn.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
 
	# 3c. return your model
	#Returns the model.
	return cnn

# Step 4: Call digit_recognition_cnn() to build your model
#Load the CNN model we created and intialize it to cnn.
cnn = digit_recognition_cnn()
#Loaded our X train, X test, y train, and y test from our loaded_dataset function to use them outside of the function.
X_train, X_test, y_train, y_test = load_dataset()

# Step 5: Train your model and see the result in Command window. Set epochs to a number between 10 - 20 and batch_size between 150 - 200
#Trains the model with the mnist dataset.
cnn.fit(X_train, y_train, batch_size=150, epochs=10, verbose=2, validation_data=(X_test, y_test))

# Step 6: Evaluate your model via your_model_name.evaluate() function and copy the result in your report
#Evaluated the data from the model with a verbose of 2.
cnn.evaluate(X_test, y_test, verbose=2)

# Step 7: Save your model via your_model_name.save('digitRecognizer.h5')
#Saves the CNN model to a file to be used later.
cnn.save('digitRecognizer.h5')

# Code below to make a prediction for a new image.

# Step 8: load required keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
 
# Step 9: load and normalize new image
def load_new_image(path):
	# 9a. load new image
	#Loads any new images we want to pass to the machine.
	newImage = load_img(path, color_mode='grayscale', target_size=(28, 28))
 
	# 9b. Convert image to array
	#Convert the new image into an array to be able to reshape and normalize it.
	newImage = img_to_array(newImage)
 
	# 9c. reshape into a single sample with 1 channel (similar to how you reshaped in load_dataset function)
	#Reshape the new image.
	newImage = newImage.reshape((1, 28, 28, 1)).astype('float32')
 
	# 9d. normalize image data - Hint: newImage = newImage / 255
	#Normalize the new image.
	newImage = newImage / 255
 
	# 9e. return newImage
	#Returns the newimage after reshaping it and normalizing it.
	return newImage

# Step 10: load a new image and predict its class
def test_model_performance():
	# 10a. Call the above load image function
	#Loading the new images we want our machine to make a perdiction on. Loading the all the images from the sample_images file(numbers 1-9).
	imgs = ['sample_images/digit1.png','sample_images/digit2.png','sample_images/digit3.png','sample_images/digit4.png','sample_images/digit5.png'
         ,'sample_images/digit6.png','sample_images/digit7.png','sample_images/digit8.png','sample_images/digit9.png']
 
	# 10b. load your CNN model (digitRecognizer.h5 file)
	#Load the model we created again to cnn to be used within the function.
	cnn = load_model('digitRecognizer.h5')
 
	# 10c. predict the class - Hint: imageClass = your_model_name.predict_classes(img)
	#Use a for loop to loop through all the new images that we loaded that we want to be predicted on.
	for i in imgs:
		#i loops through each file in imgs and sets that file to img.
		img = load_new_image(i)
		#That file that is set to img is then predicted upon by the machine.
		imageClass = cnn.predict(img)
		#Chooses the class with the highest percentage from the machine prediction and sets it to imageClassInt.
		imageClassInt = np.argmax(imageClass, axis=1)
		#Chooses the highest percentage and sets it to imageClassPercent.
		imageClassPercent = np.max(imageClass, axis=1)
  
	# 10d. Print prediction result
		#Prints the class and the percentage that it is that class from the machines predictions.
		print("Predicted Class and percentage for", i, ": ", imageClassInt[0], "and", imageClassPercent[0])
 
# Step 11: Test model performance here by calling the above test_model_performance function
#Calls the test_model_performance function to test the models performance.
test_model_performance()