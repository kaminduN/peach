################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: continuous-simulated-annealing.py
# Optimization of functions by simulated annealing
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

# Gradient of Rosenbrock function. This is used to enhance the estimate even in
# the case a random change doesn't improve the estimate.
def df(xy):
    x, y = xy
    return array( [ -2.*(1.-x) - 4.*x*(y - x*x), 2.*(y - x*x) ])

# We will allow no more than 500 iterations. For the simplified Rosenbrock
# function, no more than that will be needed.
iMax = 500

# Here we create the optimizer. There is not much difference in how an
# stochastic optmizer is created, comparing to deterministic ones.
csa = p.ContinuousSA(f, (0.1, 0.2), [ (0., 2.), (0., 2.) ], optm=p.Gradient(f, df, h=0.05))
xd = [ ]
yd = [ ]
i = 0
while i < iMax:
    x, e = csa.step()
    xd.append(x[0])
    yd.append(x[1])
    i = i + 1
xd = array(xd)
yd = array(yd)

# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``continuous-simulated-annealing.png``.
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
    savefig("continuous-simulated-annealing.png")

except ImportError:
    print "Results: ", (xd[-1], yd[-1])
