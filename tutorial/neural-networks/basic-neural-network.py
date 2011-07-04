################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/basic-neural-network.py
# Basic example of using neural networks
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
from numpy import *
import peach as p


# Creation of a three layer neural network. The first layer is the input layer,
# receives the inputs, and it contains 2 neurons. The second layer is the first
# hidden layer and will contain 2 neurons. The last layer is the output layer
# and will contain only one neuron. We will choose as activation function the
# `Sigmoid` class, and `BackPropagation` class as the learning rule.
nn = p.FeedForward((2, 2, 1), p.Sigmoid, p.BackPropagation)

# We can use the `[]` operator to select a specific layer. Notice that the input
# layer cannot be modified in any way, so `[0]`-th layer is the first hidden
# layer. The `weights` property of a layer is an array containing the synaptic
# weights of those layer -- each line is the weight vector of the corresponding
# neuron.
nn[0].weights = array([[  0.5,  0.5 ],
                       [ -0.5, -0.5 ]], dtype = float)

# We set up the synaptic weights of the neuron on the last layer. Notice that
# this layer could be accessed as `[-1]`, as a FeedForward network is only a
# list of `Layers`.
nn[1].weights = array([ 0.25, -0.25 ], dtype = float)

# This is an example that will be shown to the network for learning.
x = array([ 0.8, 0.2 ], dtype = float)   # Input vector
d = 0.9                                  # Desired response

# We feed the network the input by calling the network as a function. The
# argument to the function is the input vector. The function returns the output
# of the network.
y = nn(x)

# The method below tells the network to learn the example. The specified
# learning rule, in this case the BackPropagation, will be used to adapt the
# synaptic weights of the network.
nn.feed(x, d)

# The code below shows the results
print "Peach tutorial on neural network basics"
print
print "Input to the network:"
print x
print "Network output:"
print y
print
print "Error: %7.4f" % (d - y,)
print
print "Updated weights in the first hidden layer:"
print nn[0].weights
print
print "Updated weights in the output layer:"
print nn[1].weights
print
print "Network output with updated weights:"
print nn(x)
print
print "Updated error: %7.4f" % (d - nn(x),)
print