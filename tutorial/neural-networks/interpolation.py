################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/linear-prediction.py
# Using neural networks to predict number sequences
################################################################################


# A neural network can also be used to interpolate values of a sequence or
# function of which little is known. Typically, the structure used is a double
# layer neural network. In the first layer, neurons with sigmoid activation to
# map the nonlinearities in the function, and in the second layer a linear
# activated neuron, to combine the inputs. This structure is commonly known as
# MADALINE (Multiple Adaptive Linear Element). The goal of this tutorial is to
# show how to use Peach to do this.


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:from numpy import *
from numpy import *
import peach as p
from random import randrange


# This is the sequence that we will be interpolating, consisting of twenty
# samples evenly distributed over the interval from -pi/2 to pi/2. While with
# this data simpler methods could be used (since we will be interpolating a
# sinus), our goal is to show how to do that with neural networks
t = linspace(-pi/2., pi/2., 20)
x = sin(t)


# We create the neural network with the command below. It should be a network
# with one input neuron, and one output neuron. The hidden layer must have
# enough neurons to map the variations in the function. We will map part of a
# sinus, so 10 neurons should be enough. We must make the neurons biased. The
# reason for this is that the sigmoids of the first layer must be shifted to the
# position of the variation it will map. The second layer does not need to be
# biased, in general, but there is no harm in letting it be.
nn = p.FeedForward((1, 10, 1), phi=(p.Sigmoid, p.Identity),
       lrule=p.BackPropagation(0.05), bias=True)


# We will use this list to track the error for posterior plotting
elog = [ ]
error = 1.

# The learning loop will be executed at most 5000 times. Most of the time, this
# is an overkill, but given the stochastic nature of the learning, sometimes it
# is needed. Anyways, we put a stop trigger -- when the error reaches 1e-5, the
# algorithm stops
i = 0
while i < 5000 and error > 1.e-5:

    # The training sequence is a list of samples. We could shuffle the list and
    # present them in the same order for many epochs. However, it can be useful
    # randomly choose a sample everytime, since the randomness can help
    # convergence.
    index = randrange(20)
    xx = t[index]
    dd = x[index]

    # Here, the network is fed, the error is collected and logged, and the
    # learning process takes place.
    y = nn(xx)[0, 0]
    error = abs(dd - y)
    nn.learn(xx, dd)
    elog.append(error)
    i = i + 1


# We will now plot the response of the network for more than twenty samples,
# using 500 samples in the same interval.
ty = linspace(-pi/2, pi/2, 500)
y = [ ]
for tt in ty:
    yy = nn(tt)[0, 0]
    y.append(yy)

print nn[0].weights
print error


# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``linear-prediction.png``.
try:
    import pylab

    pylab.subplot(211)
    pylab.hold(True)
    pylab.grid(True)
    pylab.plot(array(elog), 'b')
    pylab.xlabel("Error")

    pylab.subplot(212)
    pylab.grid(True)
    pylab.stem(t, x, "k-", "ko", "k-")
    pylab.plot(ty, y)
    pylab.xlim([ -pi/2, pi/2 ])
    pylab.ylim([ amin(y)-0.1, amax(y)+0.1 ])
    pylab.xticks([ -pi/2, -pi/4, 0., pi/4, pi/2 ])
    pylab.gca().set_xticklabels([ r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$' ])
    pylab.savefig("interpolation.png")

except ImportError:
    print "After %d iterations:" % (len(elog),)
    print nn[0].weights
    print nn[1].weights