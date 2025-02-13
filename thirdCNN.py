import tensorflow as tf
from tensorflow.keras import layers, models    
import numpy as np
#import matplotlib.pyplot as plt - could not get this to work
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import time

#Load the data
cifar = tf.keras.datasets.cifar10

#Separate test/training sets
(trainImages, trainLabels), (testImages, testLabels) = cifar.load_data()

#Double checking separation
print(trainImages.shape, trainLabels.shape, testImages.shape, testLabels.shape)
#(50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

#Normalize
trainImages, testImages = trainImages.astype('float32') / 255.0, testImages.astype('float32') / 255.0

#Flatten 
trainLabels, testLabels = trainLabels.flatten(), testLabels.flatten()

#number of classes
K = len(set(trainLabels))

#print("number of classes: ", K) 
#10

trainImages = trainImages.reshape((trainImages.shape[0], 32, 32, 3))
testImages = testImages.reshape((testImages.shape[0], 32, 32, 3))

#Begin clock
start = time.time()

model = models.Sequential([
    #First layer: 32 3,3 kernels
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), padding='same'),
    layers.MaxPooling2D((2,2)),

    #Second Conv layer: 64 3,3 kernels
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    #Third Conv layer: 64 3,3 kernels
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(K,activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

history = model.fit(trainImages, trainLabels,
                    epochs=25,
                    batch_size=64,
                    validation_split=0.1)
testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=2)

#End Clock
end = time.time()

print(f'\nTest accuracy: {testAcc:.4f}\nTotal time: {end - start}')