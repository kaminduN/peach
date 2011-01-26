################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/self-organizing-maps.py
# Extended example on self-organizing maps
################################################################################

# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace. We will
# also need the random module:
from numpy import *
import random
import peach as p


# A self-organizing map has the ability to automatically recognize and classify
# patterns. This tutorial shows graphically how this happens. We have a set of
# points in the cartesian plane, each coordinate obtained from a central point
# plus a random (gaussian, average 0, small variance) shift in some direction.
# We use this set to build the network.

# First, we create the training set:
train_size = 300
centers = [ array([ 1.0, 0.0 ], dtype=float),
            array([ 1.0, 1.0 ], dtype=float),
            array([ 0.0, 1.0 ], dtype=float),
            array([-1.0, 1.0 ], dtype=float),
            array([-1.0, 0.0 ], dtype=float) ]
xs = [ ]
for i in range(train_size):
    x1 = random.gauss(0.0, 0.1)
    x2 = random.gauss(0.0, 0.1)
    xs.append(centers[i%5] + array([ x1, x2 ], dtype=float))

# Since we are working on the plane, each example and each neuron will have two
# coordinates. We will use five neurons (since we have five centers). The
# self-organizing map is created by the line below. Our goal is to show how the
# weights converge to the mass center of the point clouds, so we initialize the
# weights to show it:
nn = p.SOM((5, 2))
for i in range(5):
    nn.weights[i, 0] = 0.3 * cos(i*pi/4)
    nn.weights[i, 1] = 0.3 * sin(i*pi/4)


# We use these lists to track the variation of each neuron:
wlog = [ [ nn.weights[0] ],
         [ nn.weights[1] ],
         [ nn.weights[2] ],
         [ nn.weights[3] ],
         [ nn.weights[4] ] ]

# Here we feed and update the network. We could use the ``train`` method, but
# we want to track the weights:
for x in xs:
    y = nn(x)
    nn.learn(x)
    wlog[y].append(array(nn.weights[y]))


# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``self-organizing-maps.png``.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8, 4)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    for x in xs:
        plot( [x[0]], [x[1]], 'ko')
    for w in wlog:
        w = array(w[1:])
        plot( w[:, 0], w[:, 1], '-x')
    savefig("self-organizing-maps.png")

except ImportError:
    print "After %d iterations:" % (train_size,)
    print nn.weights