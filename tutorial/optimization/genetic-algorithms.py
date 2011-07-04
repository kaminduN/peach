################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: particle-swarm-optimization.py
# Optimization of functions by particle swarms
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
from numpy import *
from numpy.random import random
import peach as p


# This is a simplified version of the Rosenbrock function, to demonstrante
# how the optimizers work.
def f(xy):
    x, y = xy
    return (1.-x)**2. + (y-x*x)**2.


# Genetic algorithms are designed to maximize functions, not to minimize them.
# To minimize a function, we just negate it.
def J(x):
    return -f(x)


# We will allow no more than 500 iterations. For the simplified Rosenbrock
# function, no more than that will be needed.
iMax = 500


# We need to create a population of estimates. In algorithms based on
# populations, such as this or Genetic Algorithms, a list of estimates should
# be created. To this end, we will specify the ranges of the variables in the
# interval from 0. to 2., and randomly choose from this.
# Ranges are specified as a list of tuples, where each element is the allowed
# range for the corresponding variable. In each tuple, the first value is the
# lower limit of the intervalo, and the second value is its upper limit.
ranges = [ ( 0., 2. ), ( 0., 2. ) ]

# We use the numpy.random function to generate a population of five particles,
# randomly positioned in a circle around the point (1., 1.), with radius 1. We
# will do this to better observe the behaviour of the algorithm. In general,
# swarms should have more than five particles, but this is enough for this
# case. This line will be more effective if we convert ranges to a numpy array:
ranges = array(ranges)
theta = random((25, 1)) * 2. * pi
x0 = c_[ 1. + cos(theta), 1. + sin(theta) ]

# Here we create the optimizer. There is not much difference in how an
# stochastic optmizer is created, comparing to deterministic ones. However,
# since this is a population based algorithm, we will track every particle and
# the global best to have a better understanding of the behaviour in the plot.
# (We emphasize, however, that an animation in this case would be a lot better).
ga = p.GeneticAlgorithm(J, x0, ranges, 'ff')
xd = [ ]
yd = [ ]
xx = [ ]
i = 0
while i < iMax:
    x, _ = ga.step()
    x = x.decode()
    xd.append(x[0])
    yd.append(x[1])
    xx.append([ c.decode() for c in ga ])
    i = i + 1
xd = array(xd)
yd = array(yd)
xx = array(xx)


# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``genetic-algorithms.png``.
x = linspace(0., 2., 250)
y = linspace(0., 2., 250)
x, y = meshgrid(x, y)
z = (1-x)**2 + (y-x*x)**2
levels = exp(linspace(0., 2., 10)) - 0.9

try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(6, 6)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.grid(True)
    a1.plot(xx[:, 0, 0], xx[:, 0, 1], 'gray')
    a1.plot(xx[:, 1, 0], xx[:, 1, 1], 'gray')
    a1.plot(xx[:, 2, 0], xx[:, 2, 1], 'gray')
    a1.plot(xx[:, 3, 0], xx[:, 3, 1], 'gray')
    a1.plot(xx[:, 4, 0], xx[:, 4, 1], 'gray')
    a1.plot(xd, yd)
    a1.contour(x, y, z, levels, colors='k', linewidths=0.75)
    a1.set_xlim([ 0., 2. ])
    a1.set_xticks([ 0., 0.5, 1., 1.5, 2. ])
    a1.set_ylim([ 0., 2. ])
    a1.set_yticks([ 0.5, 1., 1.5, 2. ])
    savefig("genetic-algorithms.png")

except ImportError:
    pass

print "Results: ", (xd[-1], yd[-1])
