# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:38:32 2017

@author: david
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
#import util

# Load the data we are giving you
matfn = 'G:/Courses/CS391L ML/HW6/digits.mat'
data=sio.loadmat(matfn)
trainImages=data['trainImages']
trainLabels=data['trainLabels']
trainLabels=trainLabels.T
trainLabels = np.ndarray.flatten(trainLabels)
#np.reshape(trainImages[:,:,0,-1], (784,1))
A = np.empty([60000,28,28,1])
    
for i in range (0,60000):
    newImage = np.reshape(trainImages[:,:,0,i], (1,28,28,1))   # Reshape trainImages as [784 * i] 
    A[i,:,:,0] = newImage[0,:,:,0]   # Training data = A[784 * 60000]
    i = i + 1

print('Input shape: ' + str(A.shape))
print('Labels shape: ' + str(trainLabels.shape))

num_classes = 6




## Initialize model parameters
# Lets clear the tensorflow graph, so that you don't have to restart the notebook every time you change the network
tf.reset_default_graph()

# Set up your input placeholder
inputs = tf.placeholder(tf.float32, (None,28,28,1), name='input')

# Whenever you deal with image data it's important to mean center it first and subtract the standard deviation
mean = np.mean(A)
print(mean)
std = np.std(A)
print(std)
white_inputs = (inputs - mean) / std
print(white_inputs)


# Set up your label placeholders
labels = tf.placeholder(tf.int64, (None), name='labels')

# Step 1: define the compute graph of your CNN here
#   Use 5 conv2d layers (tf.contrib.layers.conv2d) and one pooling layer tf.contrib.layers.max_pool2d or tf.contrib.layers.avg_pool2d.
#   The output of the network should be a None x 1 x 1 x 6 tensor.
#   Make sure the last conv2d does not have a ReLU: activation_fn=None
h = tf.contrib.layers.conv2d(white_inputs, 19, (5,5), stride=2, scope="conv1")
h = tf.contrib.layers.conv2d(h, 20, (5,5), stride=2, scope="conv2")
h = tf.contrib.layers.conv2d(h, 50, (5,5), stride=2, scope="conv3")
h = tf.contrib.layers.conv2d(h, 70, (3,3), stride=2, scope="conv4")

#h = tf.contrib.layers.max_pool2d(h, (3,3), stride=2, scope="pool")
h = tf.contrib.layers.conv2d(h, 10, (1,1), stride=2, activation_fn=None, scope="conv5")
# The input here should be a   None x 1 x 1 x 6   tensor
output = tf.identity(tf.contrib.layers.flatten(h), name='output')

# Step 2: use a classification loss function (from assignment 3)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))

# Step 3: create an optimizer (from assignment 3)
optimizer = tf.train.MomentumOptimizer(0.001, 0.9)

# Step 4: use that optimizer on your loss function (from assignment 3)
opt = optimizer.minimize(loss)
correct = tf.equal(tf.argmax(output, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]), '/', 100000 )



## Train model
# Batch size
BS = 32

# Start a session
sess = tf.Session()

# Set up training
sess.run(tf.global_variables_initializer())

# This is a helper function that trains your model for several epochs un shuffled data
# train_func should take a single step in the optmimzation and return accuracy and loss
#   accuracy, loss = train_func(batch_images, batch_labels)
# HINT: train_func should call sess.run
def train(train_func):
    # An epoch is a single pass over the training data
    for epoch in range(5):
        # Let's shuffle the data every epoch
        np.random.seed(epoch)
        np.random.shuffle(A)
        np.random.seed(epoch)
        np.random.shuffle(trainLabels)
        # Go through the entire dataset once
        accs, losss = [], []
        for i in range(0, A.shape[0]-BS+1, BS):
            # Train a single batch
            batch_images, batch_labels = A[i:i+BS], trainLabels[i:i+BS]
            acc, loss = train_func(batch_images, batch_labels)
            accs.append(acc)
            losss.append(loss)
        print('[%3d] Accuracy: %0.3f  \t  Loss: %0.3f'%(epoch, np.mean(accs), np.mean(losss)))


# Train convnet
print('Convnet')
train(lambda I, L: sess.run([accuracy, loss, opt], feed_dict={inputs: I, labels: L})[:2])





## Input test data
testImages=data['testImages']
testLabels=data['testLabels']
testLabels=testLabels.T
testLabels = np.ndarray.flatten(testLabels)
#np.reshape(trainImages[:,:,0,-1], (784,1))
B = np.empty([10000,28,28,1])
    
for i in range (0,10000):
    newImage2 = np.reshape(testImages[:,:,0,i], (1,28,28,1))   # Reshape trainImages as [784 * i] 
    B[i,:,:,0] = newImage2[0,:,:,0]   # Training data = A[784 * 60000]
    i = i + 1

print('Input shape: ' + str(B.shape))
print('Labels shape: ' + str(testLabels.shape))

print('Input shape: ' + str(B.shape))
print('Labels shape: ' + str(testLabels.shape))

val_accuracy, val_loss = sess.run([accuracy, loss], feed_dict={inputs: B, labels: testLabels})
print("ConvNet Validation Accuracy: ", val_accuracy)
