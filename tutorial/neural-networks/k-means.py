################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/k-means.py
# Example of using K-Means implementation
################################################################################

# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace. We will
# also need the random module:
from numpy import *
import random
import peach as p


# In this tutorial, we reproduce the behaviour we seen in the self-organizing
# maps tutorial (please, refer to that tutorial for more information). The
# K-Means algorithm has the ability to find the clusters that partition a given
# set of points. This tutorial shows graphically how this happens. We have a set
# of points in the cartesian plane, each coordinate obtained from a central
# point plus a random (gaussian, average 0, small variance) shift in some
# direction.

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

# Since we are working on the plane, each example and each cluster will have two
# components. We will have five clusters, since we have five centers. The
# K-Means instance is created below.
km = p.KMeans(xs, 5)
for i in range(5):
    km.c[i, 0] = 0.3 * cos(i*pi/4)
    km.c[i, 1] = 0.3 * sin(i*pi/4)

# The __call__ interface runs the algorithm till completion. It returns the
# centers of the classification. We might pass the parameter imax to the
# algorithm. This is the maximum number of passes. In general, K-Means will
# converge very fastly and with little error. The default value for this
# parameter is 20. Notice, however, that the algorithm automatically stops if
# there are no more changes in the clusters.
c = km()
print "The algorithm converged to the centers:"
print c
print

# If the system has the plot package matplotlib, this tutorial tries to plot
# the training set and the clustered centers. The plot is saved in the file
# ``k-means.png``.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    xs = array(xs)
    figure(1).set_size_inches(8, 4)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])
    a1.hold(True)
    print xs
    print c
    a1.scatter(xs[:, 0], xs[:, 1], color='black', marker='x')
    a1.scatter(c[:, 0], c[:, 1], color='red', marker='o')
    savefig("k-means.png")
except ImportError:
    pass
