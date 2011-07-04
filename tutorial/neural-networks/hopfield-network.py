# -*- coding: utf-8 -*-
################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/hopfield-network.py
# Hopfield neural networks for recovering patterns
################################################################################


# We import here the needed libraries.
import numpy as p
from peach import *
from random import shuffle


# This function will be used to show the patterns in a way that is easier to
# the eyes. The patterns are represented by numbers -1 and 1. This function
# converts a array in a 7x5 representation, substituting a blank space for -1
# and an asterisk for 1.
def show(x):
    n = len(x)
    for i in range(0, n):
        if i%5 == 0:
            print
        if x[i] == 1:
            print '*',
        else:
            print ' ',


# This is the training set. We will be recognizing vowels in a 7x5 pattern,
# defined here. A Hopfield network doesn't have a very good storage capacity
# that allows recovering of patterns without a big probability of error.
# 5 in 35 patterns is a good enough number that allows a good demonstration.
training_set = [
    array([ -1, -1,  1, -1, -1,     # A
            -1,  1, -1,  1, -1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1,  1,  1,  1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1 ]),
    array([  1,  1,  1,  1,  1,     # E
             1, -1, -1, -1, -1,
             1, -1, -1, -1, -1,
             1,  1,  1,  1, -1,
             1, -1, -1, -1, -1,
             1, -1, -1, -1, -1,
             1,  1,  1,  1,  1 ]),
    array([ -1,  1,  1,  1, -1,     # I
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1,  1,  1,  1, -1 ]),
    array([ -1, -1,  1, -1, -1,     # O
            -1,  1, -1,  1, -1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
            -1,  1, -1,  1, -1,
            -1, -1,  1, -1, -1 ]),
    array([  1, -1, -1, -1,  1,     # U
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
            -1,  1,  1,  1, -1 ])
]


# We define here a test pattern, from one of the patterns of the training set,
# and adding noise. Noise is added by randomly choosing a position in the
# pattern and inverting it.
x = array(training_set[0])
n = len(x)
noise_position = range(n)
shuffle(noise_position)
for k in noise_position[:8]:   # We invert as much as 8 points in the pattern
    x[k] = -x[k]
x = x.reshape((n, 1))


# Here we create the Hopfield network. The Hopfield instantiation needs only the
# size of the network. The default activation function is the Signum, but it can
# be changed by passing any activation function as the second argument of the
# class instantiation.
nn = Hopfield(n)
nn.train(training_set)


# If we call the network, we can retrieve the best pattern automatically. But we
# want to see the progress of the network, so we will step by the algorithm to
# show partial results. To perform a step of the convergence, use the ``step``
# method
i = 0
xx = array(x)
while i < 100:
    xx = nn.step(xx)
    show(xx)
    #raw_input()      # Uncomment this line if you want to see step-by-step.
    i = i + 1         # You will need to press return to perform the next step.


# Shows the initial and the last states. Notice: errors might happen. Since the
# amount of noise is relatively high, the network can converge to a final state
# that is not one of the stored patterns
print "\n\n"+"-"*40
print "\n\nInitial state:"
show(x)
print "\n\nFinal state:"
show(xx)