# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # To create a NN which is sequential, other being graph
# The below 3 packages are for the 3 steps performed in convolutional algo
from keras.layers import Conv2D # This package will be used to perform convolutional function on image. The package is 2D since the images being used are 2d
from keras.layers import MaxPooling2D  # This package will be used to perform max pooling operation on the conoluted feature maps 
from keras.layers import Flatten # This package will platted the pooled feature maps

from keras.layers import Dense # Dense package is used to add layers to NN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32 tells the layer to create 32 feature detectors, 
# (3,3) means the number of rows and columns in the feature detector
# input_shape - shape of the input image. This helps in creating the images to a single size because we may have different dimensions image in the dataset. Input is the format in ehich the images will be converted. 64,64,3 means the image will be converted to a 3d matrix which contains 64*64 pixels which are colored(3)
# activation - the activation function used on the convoluted images
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) # 32 feature map of dmiiension 3x3

# Step 2 - Pooling - This step reduces the number of nodes obtained in flattening step which decreases the input nodes required which will help in reduing compute power.
# pool_size - the size to which the feature maps are scaled down. (2,2) means to reduce the feature maps by 2 times horizontally and 2 times vertically  [5*5 becomes 3*3]
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer - THis helps in creating a deeper neural network which helps in model to train more
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) # Since this model will give binary output( dog or cat), we use sigmoid function

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# This step, also called data augmentation, will create batches of the training images  we have, and each batch will have unique images. Each batch will help training more and more the NN without much training data. This will prevent overfitting and help generalize things in NN.
from keras.preprocessing.image import ImageDataGenerator
# THe ImageDataGenerator will play around with image to create new images. It will rescale the images, flip them, zoom in and out as well as shear(more features are also possible)
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# This tells from where the model has to pickup the training data, target_size is the input size which the CNN expects(setup 64*64 above),
# vatch_size tells the size of each training batch
# class_mode tells what types of images are there(binary means there are only 2 types of images possible, like dogs and cats)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#same as above
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)