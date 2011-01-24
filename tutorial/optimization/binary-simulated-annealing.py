################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: binary-simulated-annealing.py
# Optimization of functions by binary simulated annealing
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
from numpy import *
import peach as p


# This is a simplified version of the Rosenbrock function, to demonstrante
# how the optimizers work.
def f(xy):
    x, y = xy
    return (1.-x)**2. + (y-x*x)**2.

# We will allow no more than 1000 iterations. For the simplified Rosenbrock
# function, no more than that will be needed.
iMax = 1000

# Here we create the optimizer. This optimizer is created in quite a different
# way from the others we used. This is because the discrete optimizer deals with
# bit streams, and changing values on bits can be highly unpredictable --
# especially if you are dealing with floating point representations. The first
# two parameters, however, are the same as in other optimizers: the function to
# be optimized, and the first estimate. The next parameter is a string of
# formats that will be used to decode the bit stream: they work exactly as in
# the struct module that is included in every Python distribution -- please,
# consult the official documentation for more information. In this example, we
# use two floating points, thus 'ff'. The next parameter is a list of ranges of
# values that will be allowed for our estimates. This might be needed if you use
# floating points: the algorithm perform a sanity check to guarantee that the
# bitarray really represents a floating point in the allowed range -- in case
# it is not, these are used to random choose new estimates.
bsa = p.BinarySA(f, (0.1, 0.2), [ (0., 2.), (0., 2.) ], 'ff')
xd = [ ]
yd = [ ]
i = 0
while i < iMax:
    x, e = bsa.step()
    xd.append(x[0])
    yd.append(x[1])
    i = i + 1
xd = array(xd)
yd = array(yd)

# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``binary-simulated-annealing.png``.
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
    a1.plot(xd, yd)
    a1.contour(x, y, z, levels, colors='k', linewidths=0.75)
    a1.set_xlim([ 0., 2. ])
    a1.set_xticks([ 0., 0.5, 1., 1.5, 2. ])
    a1.set_ylim([ 0., 2. ])
    a1.set_yticks([ 0.5, 1., 1.5, 2. ])
    savefig("binary-simulated-annealing.png")

except ImportError:
    print "Results: ", (xd[-1], yd[-1])
