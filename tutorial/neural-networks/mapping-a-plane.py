################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/mapping-a-plane.py
# Using a neuron to map a plane
################################################################################

# Please, for more information on this demo, see the tutorial documentation.


# We import numpy, random and peach, as those are the libraries we will be
# using.
from numpy import *
import random
import peach as p


# Here, we create a FeedForward network with only one layer, with two inputs and
# one output. Since it is only one output, there is only one neuron in the
# layer. We use LMS as the learning algorithm, and the neuron must be biased.
# Notice that we use 0.02 as the learning rate for the algorithm.
nn = p.FeedForward((2, 1), lrule=p.LMS(0.02), bias=True)

# These lists will track the values of the synaptic weights and the error. We
# will use it later to plot the convergence, if the matplotlib module is
# available
w0 = [ ]
w1 = [ ]
w2 = [ ]
elog = [ ]

# We start by setting the error to 1, so we can enter the looping:
error = 1
while abs(error) > 1e-7:                      # Max error is 1e-7
    x1 = random.uniform(-10, 10)              # Generating an example
    x2 = random.uniform(-10, 10)
    x = array([ x1, x2 ], dtype = float)
    d = -1 - 3*x1 + 2*x2                      # Plane equation
    error = nn.feed(x, d)
    w0.append(nn[0].weights[0][0])            # Tracking error and weights.
    w1.append(nn[0].weights[0][1])
    w2.append(nn[0].weights[0][2])
    elog.append(d - nn(x)[0, 0])

print "After %d iterations, we get as synaptic weights:" % (len(w0),)
print nn[0].weights


# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``mapping-a-plane.png``.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    vsize = 4
    figure(1).set_size_inches(8, 4)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.grid(True)
    a1.plot(array(w0))
    a1.plot(array(w1))
    a1.plot(array(w2))
    a1.plot(array(elog))
    a1.set_ylim([-10, 10])
    a1.legend([ "$w_0$", "$w_1$", "$w_2$", "$error$" ])
    savefig("mapping-a-plane.png")

except ImportError:
    pass
