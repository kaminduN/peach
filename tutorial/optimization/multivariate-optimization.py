################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: mutivariate-optmization.py
# Optimization of two-variable functions
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

# Hessian of Rosenbrock function
def hf(xy):
    x, y = xy
    return array([ [ 2. - 4.*(y - 3.*x*x), -4.*x ],
                   [ -4.*x, 2. ] ])

# We will allow no more than 100 iterations. For the simplified Rosenbrock
# function, no more than that will be needed.
iMax = 100

# We first try using the gradient optimizer. To create an optimizer, we declare
# the function to be optimized and the first estimate. Depending on the
# algorithm, other parameters are available. Please, consult the documentation
# for more information.
grad = p.Gradient(f, (0.1, 0.2), df)
xd = [ 0.1 ]         # We use those to keep track of the convergence
yd = [ 0.2 ]
i = 0
while i < iMax:
    x, e = grad.step()
    xd.append(x[0])
    yd.append(x[1])
    i = i + 1
xd = array(xd)
yd = array(yd)

# Gradient optimizer with estimate derivative. To allow the algorithm to
# estimate the derivative, we don't declare a derivative function.
grad2 = p.Gradient(f, (0.1, 0.2))
xe = [ 0.1 ]         # We use those to keep track of the convergence
ye = [ 0.2 ]
i = 0
while i < iMax:
    x, e = grad2.step()
    xe.append(x[0])
    ye.append(x[1])
    i = i + 1
xe = array(xe)
ye = array(ye)

# Newton optimizer, with explicit declaration of the gradient and hessian.
newton = p.Newton(f, (0.1, 0.2), df, hf)
xn = [ 0.1 ]
yn = [ 0.2 ]
i = 0
while i < iMax:
    x, e = newton.step()
    xn.append(x[0])
    yn.append(x[1])
    i = i + 1
xn = array(xn)
yn = array(yn)

# Newton optimizer, with estimated gradient and hessian. We allow the algorithm
# to estimate these functions by not declaring them.
newton2 = p.Newton(f, (0.1, 0.2))
xq = [ 0.1 ]
yq = [ 0.2 ]
i = 0
while i < iMax:
    x, e = newton2.step()
    xq.append(x[0])
    yq.append(x[1])
    i = i + 1
xq = array(xq)
yq = array(yq)

# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``multivariate-optimization.png``.
# These commands are used to create the functions that will plot the contour
# lines of the Rosenbrock function.
x = linspace(0., 2., 250)
y = linspace(0., 2., 250)
x, y = meshgrid(x, y)
z = f((x, y))
levels = exp(linspace(0., 2., 10)) - 0.9

try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8, 8)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.grid(True)
    a1.plot(xd, yd)
    a1.plot(xe, ye)
    a1.plot(xn, yn)
    a1.plot(xq, yq)
    a1.contour(x, y, z, levels, colors='k', linewidths=0.75)
    a1.legend([ 'Gradient', 'Gradient/Estimated', 'Newton',
        'Newton/Estimated' ])
    a1.set_xlim([ 0., 2. ])
    a1.set_xticks([ 0., 0.5, 1., 1.5, 2. ])
    a1.set_ylim([ 0., 2. ])
    a1.set_yticks([ 0.5, 1., 1.5, 2. ])
    savefig("multivariate-optimization.png")

except ImportError:
    print "Gradient Optimizer: ", (xd[-1], yd[-1])
    print "Gradient Optimizer with estimated gradient: ", (xe[-1], ye[-1])
    print "Newton Optimizer: ", (xn[-1], yn[-1])
    print "Newton Optimizer with estimated hessian: ", (xq[-1], yq[-1])
