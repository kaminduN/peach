################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/basic-som.py
# Basic self-organizing maps
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
from numpy import *
import peach as p

# A self-organizing map is a neural network that can be used to solve a number
# of problems including clustering and classification. The goal of this tutorial
# is to show how to use Peach to deal with self-organizing maps. We start by
# creating a small map with only five neurons, with two inputs each. To inspect
# the behaviour of the map, we will set the weights to known values.
nn = p.SOM((5, 2))
nn.weights = array([
    [ 0.5, 0.0 ],
    [ -0.5, 0.0 ],
    [ 0.0, 0.0 ],
    [ 0.5, 0.5 ],
    [-0.5, 0.5 ] ], dtype=float)

# We want to see how the map behaves, so we feed the network with one vector.
# This is it.
x = array([ 0.0, -0.5 ], dtype=float)

# It is expected that the winning neuron is the one represented in the third
# line. We will check to see if that's the case.
y = nn(x)

# We update the winning neuron. Notice that the self-organizing map retains the
# information about the winner, so all we need to do is to pass the input vector
# to calculate the updating.
nn.learn(x)

# Checking the results:
print "The winning neuron was %d" % y
print "Its updated synaptic weights are:"
print nn.weights

