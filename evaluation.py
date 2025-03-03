
# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

# Step 1: Import all required keras libraries
from keras.models import load_model # This is used to load your saved model
from keras.datasets import mnist # This is used to load mnist dataset later
from keras.utils import np_utils # This will be used to convert your test image to a categorical class (digit from 0 to 9)

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

# Step 3: Load your saved model 
#Load the CNN model we created in the digitalRecognizer file and intialized it to cnn.
cnn = load_model('digitRecognizer.h5')
#Loaded our X train, X test, y train, and y test from our loaded_dataset function to use them outside of the function.
X_train, X_test, y_train, y_test = load_dataset()

# Step 4: Evaluate your model via your_model_name.evaluate(X_test, y_test, verbose = 0) function
#Evaluated the data from the model with a verbose of 2.
cnn.evaluate(X_test, y_test, verbose=2)

# Code below to make a prediction for a new image.


# Step 5: This section below is optional and can be copied from your digitRecognizer.py file from Step 8 onwards - load required keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
 
# Step 6: load and normalize new image
def load_new_image(path):
	# 6a. load new image
	#Loads any new images we want to pass to the machine.
	newImage = load_img(path, color_mode='grayscale', target_size=(28, 28))
 
	# 6b. Convert image to array
	#Convert the new image into an array to be able to reshape and normalize it.
	newImage = img_to_array(newImage)
 
	# 6c. reshape into a single sample with 1 channel (similar to how you reshaped in load_dataset function)
	#Reshape the new image.
	newImage = newImage.reshape((1, 28, 28, 1)).astype('float32')
 
	# 6d. normalize image data - Hint: newImage = newImage / 255
	#Normalize the new image.
	newImage = newImage / 255
 
	# 6e. return newImage
	#Returns the newimage after reshaping it and normalizing it.
	return newImage

# Step 7: load a new image and predict its class
def test_model_performance():
	# 7a. Call the above load image function
	#Loading the new images we want our machine to make a perdiction on. Loading the all the images from the sample_images file(numbers 1-9).
	imgs = ['sample_images/digit1.png','sample_images/digit2.png','sample_images/digit3.png','sample_images/digit4.png','sample_images/digit5.png'
         ,'sample_images/digit6.png','sample_images/digit7.png','sample_images/digit8.png','sample_images/digit9.png']
 
	# 7b. load your CNN model (digitRecognizer.h5 file)
	#Load the model we created again to cnn to be used within the function.
	cnn = load_model('digitRecognizer.h5')
 
	# 7c. predict the class - Hint: imageClass = your_model_name.predict_classes(img)
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
 
	# 7d. Print prediction result
		#Prints the class and the percentage that it is that class from the machines predictions.
		print("Predicted Class and percentage for", i, ": ", imageClassInt[0], "and", imageClassPercent[0])
 
# Step 8: Test model performance here by calling the above test_model_performance function
#Calls the test_model_performance function to test the models performance.
test_model_performance()