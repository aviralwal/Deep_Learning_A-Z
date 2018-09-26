# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
# There are two types of NN - sequential and graph. 
from keras.models import Sequential # We are using sequntial model to create NN, other being graph
from keras.layers import Dense # Used to add layers to NN

# Initialising the ANN
classifier = Sequential() # Sequence of layers

# Adding the input layer and the first hidden layer
# Dense - Adds a fully connected layer(each node is connected to each node)
# units - the number of nodes in the hidden layer. The number of units is taken as average of input nodes and output nodes
# kernel_initializer - how should the weights of the layer be initialized. 'uniform' means that the weights are initialized close to zero. uniform also makes sure the weights are initialized according to uniform distribution
# activation - the activation function which should be used in the nodes of ll. Activation function sets what value will go to the next layer. relu means rectifier linear function
# input_dim - the layers takes input of dimension 11(this is the number of features present in the dataset
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# Since the output is going to be binary in nature, the activation function is sigmoid which will give either 0 or 1. Only 1 output node 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# Now that the layers are added, the ann is compiles.
# optimizer - the algo to use to find the optimum weights. 'adam' is an implemetation stochastic gradient algorithm
# loss - the loss function to use. 'binary_crossentropy' is a logarithmic loss function
# metrics - what should be the parameter for deciding if the model is good or not. Criteria to evaluate the model 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
