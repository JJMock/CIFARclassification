import tensorflow as tf
from tensorflow.keras import layers, models    

# other imports
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

#Load the data
cifar = tf.keras.datasets.cifar10

#Separate test/training sets
(trainImages, trainLabels), (testImages, testLabels) = cifar.load_data()

#Double checking separation
print(trainImages.shape, trainLabels.shape, testImages.shape, testLabels.shape)
#(50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

#Normalize
trainImages, testImages = trainImages / 255.0, testImages / 255.0

#Flatten 
trainLabels, testLabels = trainLabels.flatten(), testLabels.flatten()

