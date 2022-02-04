#!/usr/bin/env python
# coding: utf-8

# In[60]:


#standard imports
import numpy as np 

import time
import random
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# In[61]:


import torch
import torchvision
import torchvision.transforms as transforms


# # Logistic Regression Classification Model

# In[62]:


from numpy import log, dot, e
from numpy.random import rand
batch_size = 1000
epochs = 50
height = 50
width = 50
class LogisticRegression:
    def __init__(self,input_size, lr=0.05):
        self.weights = rand(input_size)
        self.lr = lr
    def sigmoid(self, z): return 1 / (1 + e**(-z))
    
    def fit(self, X, y): 
        N = len(X)
        # Predicting with sigmoid function
        y_hat = self.sigmoid(dot(X, self.weights))
        # Updating Weights using Gradient Descent
        self.weights -= self.lr * (dot(X.T,  y_hat - y) / N  )

    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]



# # Training

# In[63]:



print("Loading training data: ")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Dataset/train',
        target_size=(height,width),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        class_mode='binary')
batch_x,batch_y = train_generator.next()
batch_x = batch_x.reshape(batch_size,-1)
lr = LogisticRegression(batch_x.shape[1])
start_time = time.time()
print("Training model:")
for i in range(epochs):
    accuracy = 0
    for j in range(int(train_generator.samples / batch_size)):
        #print(f"Training Epoch: {i} Batch: {j}")
        lr.fit(batch_x,batch_y)
        batch_x,batch_y = train_generator.next()
        batch_x = batch_x.reshape(batch_size,-1)
        accuracy += accuracy_score(batch_y,lr.predict(batch_x))
    print(f"Epoch {i}/{epochs} Training Accuracy: {(accuracy / int(train_generator.samples / batch_size)) * 100}")
print(f"Training completed in {time.time() - start_time} seconds")


# # Testing

# In[64]:


start_time = time.time()
print("Loading test data: ")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'Dataset/test',
        target_size=(height,width),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        class_mode='binary')
test_x,test_y = test_generator.next()
test_x = test_x.reshape(batch_size,-1)  
    
print("Testing model:") 
test_accuracy = 0
for j in range(int(test_generator.samples / batch_size)):
    #print(f"Testing Batch: {j}")
    test_accuracy += accuracy_score(test_y,lr.predict(test_x))
    test_x,test_y = test_generator.next()
    test_x = test_x.reshape(batch_size,-1)
        
print(f"Testing accuracy: {(test_accuracy / int(test_generator.samples / batch_size)) * 100}")
print(f"Testing completed in {time.time() - start_time} seconds")


# # Pytorch Api Classification Model (NN - CNN)

# In[65]:


import torch.nn as nn
import torch.nn.functional as F
batch_size = 100
epochs = 150
height = 50
width = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(height * width, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# In[66]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# # Training

# In[67]:



print("Loading training data: ")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Dataset/train',
        target_size=(height,width),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        class_mode='binary')
batch_x,batch_y = train_generator.next()
batch_x = batch_x.reshape(batch_size,-1)
#batch_y = tf.keras.utils.to_categorical(batch_y, 2)
inputs, labels = batch_x,batch_y
def acc(y_true,y_pred):
    count = 0
    for i in range(len(y_true)):
        if(y_true[i] == np.argmax(y_pred[i])):
            count +=1
    return count

start_time = time.time()
print("Training Model: ")
for epoch in range(epochs):  # loop over the dataset multiple times
    accuracy = 0
    for j in range(int(train_generator.samples / batch_size)):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(torch.from_numpy(inputs))
        outputs_temp = outputs
        loss = criterion(outputs, torch.from_numpy(labels).long() )
        loss.backward()
        optimizer.step()
        accuracy = accuracy + acc(batch_y,outputs_temp.detach().numpy())
        #print(f"Training Epoch: {i} Batch: {j}")
        batch_x,batch_y = train_generator.next()
        batch_x = batch_x.reshape(batch_size,-1)
        inputs, labels = batch_x,batch_y
    print(f"Epoch {epoch}/{epochs} Training Accuracy: {accuracy / int(train_generator.samples / batch_size)}")
print(f'Training competed in {time.time() - start_time}')


# # Testing

# In[68]:


print("Loading test data: ")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'Dataset/test',
        target_size=(height,width),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        class_mode='binary')
test_x,test_y = test_generator.next()
test_x = test_x.reshape(batch_size,-1)  
start_time = time.time()
print("Testing model:") 
test_accuracy = 0
for j in range(int(test_generator.samples / batch_size)):
    #print(f"Testing Batch: {j}")
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = net(torch.from_numpy(test_x))
    test_accuracy = test_accuracy + acc(test_y,outputs.detach().numpy())
    test_x,test_y = test_generator.next()
    test_x = test_x.reshape(batch_size,-1)
        
print(f"Testing accuracy: {test_accuracy / int(test_generator.samples / batch_size)}")
print(f'Testing competed in {time.time() - start_time}')

