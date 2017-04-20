'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import print_function
import sklearn.model_selection as sk #used to partition data
import numpy as np
import cleandata
import preprocessing
import tensorflow as tf
import pdb
import sys

####################################################################
####################        CONSTANTS       ########################
####################################################################
LEARNING_RATE = 0.001 #how quickly the model updates while training
ITERATIONS = 1 #number of times to run through the data
BATCH_SIZE = 9 #how many data points to run through between each 
                #update to the model
'''display_step = 1''' #removed because it made the code more confusing

# Network Parameters
n_hidden_1 = 9 # number of nodes in 1st layer
n_hidden_2 = 9 # number of nodes in 2nd layer
#data_size =  window_size #number of features SHOULDN'T BE A CONSTANT
CLASSES = 3 #possible classifications to choose between


####################################################################
#################     CREATE_SUBCONNECTED_LAYER       ##############
####################################################################
#purpose: creates a subconnected layer (as opposed to a fully-connected layer)
#       For example, if num_subgraphs = 2, will produce 2 subgraphs that connect
#       to half (1/num_subgraphs) of the outputs from the previous layer.
#       This subconnected layer will produce the same number of outputs as
#       a fully-connected layer would.
#For documentation on slicing and joining see 
#https://www.tensorflow.org/api_guides/python/array_ops
#inputs:x- the previous layer that you want to connect to your subconnected layer
#       weights- the tensorflow variable determining the strength of the connections 
#               from the previous layer to this one. This is one of the things that
#               will be trained.
#       biases- the tensorflow variable determining the constant added to the output
#               of the previous layer. This is one of the things that will be 
#               trained.
#       num_subgraphs- int representing the number of subgraphs that will recieve
#               a fraction of the output from the previous layer. Each subgraph
#               will recieve 1/num_subgraphs of the outputs from the previous layer
#outputs: returns the subconnected layer

def create_subconnected_layer(x, weights, biases, num_subgraphs):
    slice_size = int(int(x.get_shape()[1]) / num_subgraphs)
    layer_list = [] #Will contain all of the slices
    for s in range(0, num_subgraphs):
        #create a slice of size slice_size starting at s*slice_size
        x_slice = tf.slice(x, [0, s*slice_size], [-1, slice_size])
        
        #create subgraph by multiplying by weights and adding in bias, as you
        #would with a fully-connected layer
        subgraph = tf.add(tf.matmul(x_slice, weights[s]), biases[s])
        subgraph = tf.nn.relu(subgraph)
        layer_list.append(subgraph)
    return tf.concat(layer_list, 1)



#######################################################
##########      MULTILAYER_PERCEPTRON   ###############
#######################################################
#purpose: create a model with 2 subconnected hidden layers and 1 output layer
#inputs:x- the tensorflow placeholder that will feed data into your model
#       weights- the tensorflow variable determining the strength of the connections 
#               from the previous layer to this one. This is one of the things that
#               will be trained.
#       biases- the tensorflow variable determining the constant added to the output
#               of the previous layer. This is one of the things that will be 
#               trained.
#outputs: returns predictions class labels 
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = create_subconnected_layer(x, weights['h1'], biases['b1'], layer_1_subgraphs)
    print(layer_1.get_shape().as_list())
    print(x.get_shape().as_list())
    
    
    # Hidden layer with RELU activation
    layer_2 = create_subconnected_layer(layer_1, weights['h2'], biases['b2'], layer_2_subgraphs)
    print(layer_2.get_shape().as_list())
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    print(out_layer.get_shape().as_list())
    return out_layer





#######################################################
############        MAIN            ###################
#######################################################
if __name__ == "__main__":
    input_file = str(sys.argv[3])
    label_file = str(sys.argv[4])
    window_size = int(sys.argv[1])
    num_examples = int(sys.argv[2])


# tf Graph input
x = tf.placeholder("float", [None, window_size])  #inputs 
y_ = tf.placeholder("float", [None, CLASSES])   #ground-truth labels


# Create model with subgraphs
layer_1_subgraphs = 3 # How many fully connected subgraphs in layer 1?
layer_2_subgraphs = 1 # How many fully connected subgraphs in layer 2?
assert window_size % layer_1_subgraphs == 0
assert n_hidden_1 % layer_1_subgraphs == 0
assert n_hidden_2 % layer_2_subgraphs == 0
assert CLASSES % layer_2_subgraphs == 0



#Store layers weight & bias
weights = {
    'h1': [tf.Variable(tf.random_normal([int(window_size/layer_1_subgraphs), int(n_hidden_1/layer_1_subgraphs)])) for s in range(0, layer_1_subgraphs)],
    'h2': [tf.Variable(tf.random_normal([int(n_hidden_1/layer_2_subgraphs), int(n_hidden_2/layer_2_subgraphs)])) for s in range(0, layer_2_subgraphs)],
    'out': tf.Variable(tf.random_normal([int(n_hidden_2), int(CLASSES)]))
}
biases = {
    'b1': [tf.Variable(tf.random_normal([int(n_hidden_1/layer_1_subgraphs)])) for s in range(0, layer_1_subgraphs)],
    'b2': [tf.Variable(tf.random_normal([int(n_hidden_2/layer_2_subgraphs)])) for s in range(0, layer_2_subgraphs)],
    'out': tf.Variable(tf.random_normal([int(CLASSES)]))
}


# Construct model
y = multilayer_perceptron(x, weights, biases)   #y contains the predicted outputs
                                            #which will be compared to the 
                                            #ground-truth, y_

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

#get the generator for features and labels
generator = preprocessing.preprocess(input_file, label_file, window_size)
features = []
labels = []
for _ in range(num_examples):
    curr = next(generator)
    #need to convert all to int
    curr_features = curr[0]
    curr_features = list(map(int, curr_features))
    print(curr_features)
    print(curr[1])
    features.append(curr_features)
    labels.append(curr[1])

features = np.asarray(features)
labels = np.asarray(labels)  
print(features.shape)
print(labels.shape)

tf.get_shape
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
     
    # Training cycle
    for epoch in range(ITERATIONS):
        '''avg_cost = 0.''' #removed from example code to simplify
        total_batch = int(num_examples/BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run([optimizer, cost], feed_dict={x: features, y_: labels})
            
            #removed avg_cost tracking for simplicity
            '''# Compute average loss
            avg_cost += int(c / total_batch)''' #c was collected from sess.run
            
        #removed this section from the example code for simplicity    
        '''# Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))''' 
                
    print("Optimization Finished!")


    # Test model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "int32"))
    for j in range(total_batch):
        print("Accuracy:", accuracy.eval({x: features, y: labels}))
