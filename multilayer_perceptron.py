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

if __name__ == "__main__":
    input_file = str(sys.argv[3])
    label_file = str(sys.argv[4])
    window_size = int(sys.argv[1])
    num_examples = int(sys.argv[2])

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 9
display_step = 1

# Network Parameters
n_hidden_1 = 9 # 1st layer number of features
n_hidden_2 = 9 # 2nd layer number of features
n_input =  81 # MNIST data input (img shape: 5064 3240)
n_classes = 3 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# Create fully connected subgraphs from a single weight/bias variable tensor
# Each subgraph is divided into equal groups, and the same number of outputs
# are produced as if the entire layer was fully connected.
def create_subconnected_layer(x, weights, biases, num_subgraphs):
    num_inputs = int(int(x.get_shape()[1]) / num_subgraphs)
    layer_list = []
    for s in range(0, num_subgraphs):
        x_slice = tf.slice(x, [0, s*num_inputs], [-1, num_inputs])
        subgraph = tf.add(tf.matmul(x_slice, weights[s]), biases[s])
        subgraph = tf.nn.relu(subgraph)
        layer_list.append(subgraph)
    return tf.concat(layer_list, 1)

# Create model with subgraphs
layer_1_subgraphs = 3 # How many fully connected subgraphs in layer 1?
layer_2_subgraphs = 1 # How many fully connected subgraphs in layer 2?
assert n_input % layer_1_subgraphs == 0
assert n_hidden_1 % layer_1_subgraphs == 0
assert n_hidden_2 % layer_2_subgraphs == 0
assert n_classes % layer_2_subgraphs == 0
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = create_subconnected_layer(x, weights['h1'], biases['b1'], layer_1_subgraphs)

    # Hidden layer with RELU activation
    layer_2 = create_subconnected_layer(layer_1, weights['h2'], biases['b2'], layer_2_subgraphs)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': [tf.Variable(tf.random_normal([int(n_input/layer_1_subgraphs), int(n_hidden_1/layer_1_subgraphs)])) for s in range(0, layer_1_subgraphs)],
    'h2': [tf.Variable(tf.random_normal([int(n_hidden_1/layer_2_subgraphs), int(n_hidden_2/layer_2_subgraphs)])) for s in range(0, layer_2_subgraphs)],
    'out': tf.Variable(tf.random_normal([int(n_hidden_2), int(n_classes)]))
}
biases = {
    'b1': [tf.Variable(tf.random_normal([int(n_hidden_1/layer_1_subgraphs)])) for s in range(0, layer_1_subgraphs)],
    'b2': [tf.Variable(tf.random_normal([int(n_hidden_2/layer_2_subgraphs)])) for s in range(0, layer_2_subgraphs)],
    'out': tf.Variable(tf.random_normal([int(n_classes)]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

#get the generator for features and labels
generator = preprocessing.preprocess(input_file, label_file, window_size)
features = []
labels = []
features, labels = zip(*generator)
features = np.asarray(features)
labels = np.asarray(labels)   

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
     
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: features,
                                                          y: labels})
            # Compute average loss
            avg_cost += int(c / total_batch)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    #X_train, X_test, Y_train, Y_test = sk.train_test_split(features, labels, test_size=0.33, random_state=42)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "int32"))
    for j in range(total_batch):
        print("Accuracy:", accuracy.eval({x: features, y: labels}))
