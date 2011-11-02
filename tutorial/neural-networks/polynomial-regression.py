################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/polynomial-regression.py
# Using neural networks to approximate functions by polynomials.
################################################################################


# The learning algorithm of neural networks are based, mainly, in the mean
# squared error of the output, considering the desired output of the network.
# The same criterium is used for a lot of other types of approximation. The most
# used, and one of the first, is the linear regression, where the relation of a
# set of points is approximated by a straight line. The theory for the linear
# regression can be easily expanded to approximate functions by polynomials, but
# in general, the equations are not simple.
#
# A neuron can be used to make this approximation simpler. This tutorial shows
# how to do it.


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
from numpy import *
import random
import peach as p


# We create here the neural network. To make the polynomial regression, instead
# of supplying the neuron with the value of the independent variable, we supply
# its integer powers. The number of inputs will be, thus, the order of the
# polynomial used for approximation. With this approach, our neural network will
# be very simple: a single neuron with N+1 inputs, one output, and linear
# activation. The learning algorithm will be the LMS algorithm.
N = 10
nn = p.FeedForward((N, 1), phi=p.Identity, lrule=p.LMS(0.05), bias=True)

# We will map a period of a sinus. It is not expected that the coefficients
# found here will be the same of the Taylor Series, since the optimization
# criterium is diferent.
error = 1
i = 0
powers = arange(N, dtype=float)  # This vector will be used to calculate the powers
while i < 2000:

    # Here, we generate one value in the interval e calculate the desired
    # response. We raise ``x`` to ``powers`` to generate the inputs. It is easy
    # to see that the polynomial regression is a linear combination of the
    # powers of a variable.
    x = random.uniform(-0.5, 0.5)
    d = sin(pi*x)
    xo = x ** powers

    # We feed the network, calculate the error and teach the network
    y = nn(xo)
    error = abs(d - y)
    nn.learn(xo, d)

    i = i + 1


print "Coefficients: "
for i in range(N):
    print "%d -> %10.7f" % (i, nn[0].weights[0][i])


# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``polynomial-regression.png``.
try:
    import pylab

    x = linspace(-0.5, 0.5, 200)
    y = sin(pi*x)
    ye = [ ]
    for xo in x:
        ye.append(nn(xo**powers)[0, 0])
    ye = array(ye)

    pylab.subplot(211)
    pylab.hold(True)
    pylab.grid(True)
    pylab.plot(x, y, 'b--')
    pylab.plot(x, ye, 'g')
    pylab.xlim([ -0.5, 0.5 ])
    pylab.legend([ "$y$", "$\hat{y}$" ])

    pylab.subplot(212)
    pylab.grid(True)
    pylab.stem(arange(0, N+1), reshape(nn[0].weights, (N+1,)), "k-", "ko", "k-")
    pylab.xlim([0, N])
    pylab.savefig("polynomial-regression.png")

except ImportError:
    pass

