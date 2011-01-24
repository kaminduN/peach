################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: quasi-newton-optimization.py
# Optimization of two-variable functions by quasi-newton methods
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

# Gradient of Rosenbrock function
def df(xy):
    x, y = xy
    return array( [ -2.*(1.-x) - 4.*x*(y - x*x), 2.*(y - x*x) ])

# We will allow no more than 200 iterations. For the simplified Rosenbrock
# function, no more than that will be needed.
iMax = 200

# The first estimate of the minimum is given by the DFP method. Notice that the
# creation of the optimizer is virtually the same as every other. We could, in
# this and in the other optimizers, omit the derivative function and let Peach
# estimate it for us.
dfp = p.DFP(f, (0.1, 0.2), df)
xd = [ 0.1 ]
yd = [ 0.2 ]
i = 0
while i < iMax:
    x, e = dfp.step()
    xd.append(x[0])
    yd.append(x[1])
    i = i + 1
xd = array(xd)
yd = array(yd)

# We now try the BFGS optimizer.
bfgs = p.BFGS(f, (0.1, 0.2), df)
xb = [ 0.1 ]
yb = [ 0.2 ]
i = 0
while i < iMax:
    x, e = bfgs.step()
    xb.append(x[0])
    yb.append(x[1])
    i = i + 1
xb = array(xb)
yb = array(yb)

# Last but not least, the SR1 optimizer
sr1 = p.SR1(f, (0.1, 0.2), df)
xs = [ 0.1 ]
ys = [ 0.2 ]
i = 0
while i < iMax:
    x, e = sr1.step()
    xs.append(x[0])
    ys.append(x[1])
    i = i + 1
xs = array(xs)
ys = array(ys)

# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``quasi-newton-optimization.png``.
# These commands are used to create the functions that will plot the contour
# lines of the Rosenbrock function.
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
    a1.plot(xb, yb)
    a1.plot(xs, ys)
    a1.contour(x, y, z, levels, colors='k', linewidths=0.75)
    a1.legend([ 'DFP', 'BFGS', 'SR1' ])
    a1.set_xlim([ 0., 2. ])
    a1.set_xticks([ 0., 0.5, 1., 1.5, 2. ])
    a1.set_ylim([ 0., 2. ])
    a1.set_yticks([ 0.5, 1., 1.5, 2. ])
    savefig("quasi-newton-optimization.png")

except ImportError:
    print "DFP Optimizer: ", (xd[-1], yd[-1])
    print "BFGS Optimizer: ", (xb[-1], yb[-1])
    print "SR1 Optimizer: ", (xs[-1], ys[-1])
