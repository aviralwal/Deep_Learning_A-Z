# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/aviral/learn/DeepLearning/UnsupervisedLearning/SelfOrganizingMaps/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
# The minisom package is an implementation of SOM
from minisom import MiniSom
# x,y - The size of how big the map is going to be
# input_len - The number of input features present in the dataset which will be passed to the SOM, not the original dataset(since we can have removed some features) 
# sigma - radius of different neighbours in SOM, i.e., how many neigbours will be in the vicinity of the selected node
# learning_rate - hyperparameter which tells by how much the weights will be modified in each iteration. Higher is the learning rate, faster the weights will converge and lower the value, slower the wights will converge
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) 
# random_weights_init - this method initializes random wights to the o/p nodes.
# X - dataset on which the map will train
som.random_weights_init(X)
# train_random - method which will perform the random_training of the SOM, i.e., input random row from dataset, calculate euclidean distance, update the node weights as well as the one's in the radius
# data = x - The dataset on which to train
# num_iteratiosn - The number of iterations for which the complete algo will run
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)