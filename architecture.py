# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:40:53 2018

@author: david
"""

import tensorflow as tf
import numpy as np

#%%
'''Import Training Data'''
trainLabels = []
num = []

file = open('./output.dat')
print(file)

for line in file:
    trainLabels += [line[0]]
    num += [line[3:-22]]
for i in range(len(trainLabels)):
    trainLabels[i] = trainLabels[i].split(' ')
    num[i] = num[i].split(' ')
    
trainLabels = np.asarray(trainLabels).astype(np.float)
trainLabels = np.ndarray.flatten(trainLabels)   # Labels shape: (sample size, )
num = np.asarray(num)[:,:-1].astype(np.float)
num = np.reshape(num, (num.shape[0],64,1,1))   # Input shape: (sample size, 64, 1, 1)


print('Input shape: ' + str(num.shape))
print('Labels shape: ' + str(trainLabels.shape))

#%%
'''Model Defign'''

# Initialize model parameters
tf.reset_default_graph()

# Set up input placeholder
inputs = tf.placeholder(tf.float32, (None,64,1,1), name='input')
# Set up label placeholders
labels = tf.placeholder(tf.int64, (None), name='labels')

# Whitening
mean = np.mean(num)
print(mean)
std = np.std(num)
print(std)
white_inputs = (inputs - mean) / std
white_inputs = inputs - mean
print(white_inputs)

# Architecture
h = tf.contrib.layers.conv2d(white_inputs, 64, (80,1), stride=2)
h = tf.nn.dropout(h, keep_prob = 1,seed=0)
h = tf.contrib.layers.conv2d(h, 128, (80,1), stride=2)
h = tf.nn.dropout(h, keep_prob = 1,seed=0)
h = tf.contrib.layers.conv2d(h, 10, (1,1), activation_fn=None, stride=2)   # Output channel: 10
h = tf.nn.dropout(h, keep_prob = 1,seed=0)
output = tf.identity(tf.contrib.layers.flatten(h), name='output')

# Loss function: cross entropy (including softmax activation here)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))

# Optimizer
optimizer = tf.train.MomentumOptimizer(0.001, 0.9)

# Acc caluculation
opt = optimizer.minimize(loss)
correct = tf.equal(tf.argmax(output, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]), '/', 100000 )


#%%
''' Train model '''
# Batch size
BS = 10

# Start a session
sess = tf.Session()

# Set up training
sess.run(tf.global_variables_initializer())

def train(train_func):
    # An epoch is a single pass over the training data
    for epoch in range(50):
        # Shuffle the data every epoch
        np.random.seed(epoch)
        np.random.shuffle(num)
        np.random.seed(epoch)
        np.random.shuffle(trainLabels)
        # Go through the entire dataset once
        accs, losss = [], []
        for i in range(0, num.shape[0]-BS+1, BS):
            # Train a single batch
            batch_images, batch_labels = num[i:i+BS], trainLabels[i:i+BS]
            acc, loss = train_func(batch_images, batch_labels)
            accs.append(acc)
            losss.append(loss)
        print('[%3d] Accuracy: %0.3f  \t  Loss: %0.3f'%(epoch, np.mean(accs), np.mean(losss)))

# Train convnet
print('Convnet')
train(lambda I, L: sess.run([accuracy, loss, opt], feed_dict={inputs: I, labels: L})[:2])


#%%
'''Import Test Data'''
test_labels = []
test_num = []

file2 = open('G:/Courses/CS394N Neural Networks/HW1/test.dat')
# 'G:/Courses/CS394N Neural Networks/HW1/nist.dat'
print(file2)

for line in file2:
    test_labels += [line[0]]
    test_num += [line[3:-22]]
for i in range(len(test_labels)):
    test_labels[i] = test_labels[i].split(' ')
    test_num[i] = test_num[i].split(' ')
    
test_labels = np.asarray(test_labels).astype(np.float)
test_labels = np.ndarray.flatten(test_labels)
test_num = np.asarray(test_num)[:,:].astype(np.float)
test_num = np.reshape(test_num, (test_num.shape[0],64,1,1))

print('Test data shape: ' + str(test_num.shape))
print('Test labels shape: ' + str(test_labels.shape))

'''Test'''
val_accuracy, val_loss = sess.run([accuracy, loss], feed_dict={inputs: test_num, labels: test_labels})
print("ConvNet Test Accuracy: ", val_accuracy)