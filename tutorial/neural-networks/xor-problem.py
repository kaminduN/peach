################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/xor-problem.py
# Solving the exclusive-or problem
################################################################################

# Please, for more information on this demo, see the tutorial documentation.


# First, we import the needed modules
from numpy import *
import peach as p


# The network to solve the exclusive-or problem must have two layers: two input
# neurons and one output neuron. The neurons should be biased and the activation
# function should be sigmoidal. The learning rule is backpropagation.
nn = p.FeedForward((2, 2, 1), p.TanH, p.BackPropagation(0.2), True)

# This is the training set. A training set is a list of tuples with two elements
# each. The first element is the input vector, the second element is the output
# vector. In this case, the output is just a number.
train_set = [ ( array(( -1., -1.)), -1. ),
              ( array(( -1.,  1.)),  1. ),
              ( array((  1., -1.)),  1. ),
              ( array((  1.,  1.)), -1. ) ]

# This shows the training set to the network.
nn.train(train_set)

# Testing the results:
print nn[0].weights
print nn[1].weights
for x, _ in train_set:
    print x, " => ", nn(x)