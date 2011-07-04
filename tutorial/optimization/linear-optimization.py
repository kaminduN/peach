################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: linear-optmization.py
# Simple optimization of one-variable functions
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
from numpy import *
import peach as p

# The Rosenbrock function will be used to test the optimizers. This is a
# simplified version, which allows faster convergence, and it serves only the
# purposes of testing.
def f(x):
    return (1-x)**2 + (1-x*x)**2

# We will allow no more than 100 iterations. For the simplified Rosenbrock
# function, no more than that will be needed.
iMax = 100

# Direct one-dimensional optimizer. To create an optimizer, we declare the
# function to be optimized and the first estimate. Depending on the algorithm,
# other parameters are available. Please, consult the documentation for more
# information.
linear = p.Direct1D(f, 0.75)
xl = [ ]               # These lists will track the progress of the algorithm
i = 0
while i < iMax:
    x, e = linear.step()
    xl.append(x)
    i = i + 1
xl = array(xl)

# Parabolic interpolator optimizer.
interp = p.Interpolation(f, (0., 0.75, 1.5))
xp = [ ]              # These lists will track the progress of the algorithm
i = 0
while i < iMax:
    x, e = interp.step()
    x0, x1, x2 = x        # division by zero. We check for that
    q0 = x0 * (f(x1) - f(x2))
    q1 = x1 * (f(x2) - f(x0))
    q2 = x2 * (f(x0) - f(x1))
    q = q0 + q1 + q2
    if q == 0:            # if q==0, all estimates are identical
        xm = x0
    else:
        xm = 0.5 * (x0*q0 + x1*q1 + x2*q2) / (q0 + q1 + q2)
    xp.append(xm)
    i = i + 1
xp = array(xp)

# Golden Section Optimizer
golden = p.GoldenRule(f, (0.25, 1.25))
xg = [ ]
i = 0
while i < iMax:
    x, e = golden.step()
    xo, xh = x
    xm = 0.5 * (xo+xh)
    xg.append(xm)
    i = i + 1
xg = array(xg)

# Fibonacci optimizer
fib = p.Fibonacci(f, (0.75, 1.4))
xf = [ ]
i = 0
while i < iMax:
    x, e = fib.step()
    xo, xh = x
    xm = 0.5 * (xo+xh)
    xf.append(xm)
    i = i + 1
xf = array(xf)

# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``linear-optimization.png``.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    vsize = 4
    figure(1).set_size_inches(8, 4)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.grid(True)
    a1.plot(xl)
    a1.plot(xp)
    a1.plot(xg)
    a1.plot(xf)
    a1.legend([ "Linear", "Interpolation", "Golden Section", "Fibonacci" ])
    savefig("linear-optimization.png")

except ImportError:
    print "Linear Optimizer: ", xl[-1]
    print "Interpolation Optimizer: ", xp[-1]
    print "Golden Rule Optimizer: ", xg[-1]
    print "Fibonacci Optimizer: ", xf[-1]
