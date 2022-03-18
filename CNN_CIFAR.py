#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports for visualization and preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn


# In[2]:


#imported dataset from keras
from keras.datasets import cifar10


# In[3]:


#loaded dataset into the test and training values
(X_train,y_train),(X_test,y_test) = cifar10.load_data()


# In[4]:


#confirmed the shape of the dataset
X_train.shape


# In[5]:


X_test.shape


# In[6]:


y_train.shape


# In[7]:


y_test.shape


# In[8]:


i = 1000
plt.imshow(X_train[i])


# In[9]:


#Visualized the dataset by incrementing through it and graphing it
W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize=(25,25))
axes = axes.ravel()
n_training = len(X_train)

for i in np.arange(0,L_grid * W_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(X_train[index])
    axes[i].axis('off')
plt.subplots_adjust(hspace = 0.4)


# In[10]:


#Data preparation
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

number_cat = 10


# In[11]:


y_train 


# In[12]:


import keras
from keras.utils.np_utils import to_categorical

#changed the y_train and y_test to binary values using the categorical tool

y_train = to_categorical(y_train,number_cat)
y_test = to_categorical(y_test,number_cat)


# In[13]:


# converted the X_train values to floats between 0 and 1 by dividing it by the number of colors 255
X_train = X_train/255
X_test = X_test/255


# In[14]:


# took the shape of each individual element for input in the model
Input_shape = X_train.shape[1:]


# In[15]:


#Imported the layers and optimizers to build convulutional neural networks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[16]:


cnn_model = Sequential()
#Input Layer of two conv2d layers using filter of 64 and passing kernal sizes of 3 X 3 
cnn_model.add(Conv2D(filters=64,kernel_size =(3,3),activation='relu',input_shape= Input_shape))
cnn_model.add(Conv2D(filters=64, kernel_size = (3,3),activation='relu'))
#Used the MaxPooling method to transform the image matrices down to a 2 X 2 matrix to identify the key features
cnn_model.add(MaxPooling2D(2,2))
#added Dropout to prevent overfitting of the model
cnn_model.add(Dropout(0.2))

#Additional layer of conv2D with twice as many filters
cnn_model.add(Conv2D(filters=128,kernel_size=(3,3), activation='relu'))
cnn_model.add(Conv2D(filters=128,kernel_size=(3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.1))

#Flatten the matrices to an array to be fed to the dense neural networks
cnn_model.add(Flatten())


cnn_model.add(Dense(units= 1024, activation = 'relu'))
cnn_model.add(Dense(units=1024, activation ='relu'))

#Given this is a classification problem the activation function was set to softmax to give probabilites between 0 and 1
cnn_model.add(Dense(units =10, activation = 'softmax'))


# In[17]:


import tensorflow
#Compiled the model using the loss function categorical_crossentropy and the RMS Prop is used to help with vanishing gradien with the minibatches
cnn_model.compile(loss='categorical_crossentropy', optimizer = tensorflow.keras.optimizers.RMSprop(lr=0.001),metrics=['accuracy'])


# In[18]:


#Fit the model and we get very bad accurracy also used shuffle so that model does not use the pattern of data to get predictions right
history = cnn_model.fit(X_train,y_train, batch_size = 32,epochs = 2, shuffle= True)


# In[19]:


#Saved the data to a model file to be used or loaded by using the same params
import os
directory = os.path.join(os.getcwd( ),'CNN_models')
if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5')


# In[20]:


import keras
from keras.datasets import cifar10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()


# In[21]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[22]:


y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)


# In[23]:


n = 8
X_train_sample = X_train[:n]


# In[24]:


from keras.preprocessing.image import ImageDataGenerator
dataget_train = ImageDataGenerator(rotation_range = 90)
dataget_train.fit(X_train)


# In[25]:


#Changed the dataset to further train our model
from PIL import Image
fig = plt.figure(figsize=(20,2))
for x_batch in dataget_train.flow(X_train_sample, batch_size = n):
    for i in range(0,n):
        ax = fig.add_subplot(1,n,i + 1)
        ax.imshow(Image.fromarray(np.uint8(x_batch[i])))
        
    fig.suptitle('Augmented Dataset')
    plt.show()
    break;


# In[26]:


datagen = ImageDataGenerator(rotation_range=90,width_shift_range=0.1, horizontal_flip = True, vertical_flip = True)


# In[27]:


datagen.fit(X_train)


# In[28]:


cnn_model.fit(datagen.flow(X_train, y_train, batch_size = 32), epochs=2)


# In[ ]:




