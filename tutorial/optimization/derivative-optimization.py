################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: derivative-optmization.py
# Simple optimization of one-variable functions by derivative methods
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
from numpy import *
import peach as p

# The Rosenbrock function will be used to test the optimizers. This is a
# simplified version, which allows faster convergence, and it serves only the
# purposes of testing.
def f(x):
    return (1.-x)**2. + (1.-x*x)**2.

# The derivative of the Rosenbrock function. This is used in the Gradient and
# Newton search methods.
def df(x):
    return -2.*(1.-x) - 4.*(1.-x*x)*x

# The second derivative of the Rosenbrock function. Used in Newton method.
def ddf(x):
    return 2. - 4.*(1. - 3.*x*x)

# We will allow no more than 100 iterations. For the simplified Rosenbrock
# function, no more than that will be needed.
iMax = 100

# Gradient optimizer. To create an optimizer, we declare the function to be
# optimized and the first estimate. Depending on the algorithm, other parameters
# are available. Please, consult the documentation for more information.
grad = p.Gradient(f, 0.84, df=df)
xd = [ 0.84 ]
i = 0
while i < iMax:
    x, e = grad.step()
    xd.append(x)
    i = i + 1
xd = array(xd)

# Gradient optimizer with estimated gradient
grad2 = p.Gradient(f, 0.84)
xe = [ 0.84 ]
i = 0
while i < iMax:
    x, e = grad2.step()
    xe.append(x)
    i = i + 1
xe = array(xe)

# Newton optimizer with explicit declaration of derivatives
newton = p.Newton(f, 0.84, df=df, hf=ddf)
xn = [ 0.84 ]
i = 0
while i < iMax:
    x, e = newton.step()
    xn.append(x)
    i = i + 1
xn = array(xn)

# Newton optimizer with estimated gradient and hessian
newton2 = p.Newton(f, 0.84)
xq = [ 0.84 ]
i = 0
while i < iMax:
    x, e = newton2.step()
    xq.append(x)
    i = i + 1
xq = array(xq)

# If the system has the plot package matplotlib, this tutorial tries to plot
# and save the convergence of synaptic weights and error. The plot is saved in
# the file ``derivative-optimization.png``.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    vsize = 4
    figure(1).set_size_inches(8, 4)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.grid(True)
    a1.plot(xd)
    a1.plot(xe)
    a1.plot(xn)
    a1.plot(xq)
    a1.legend([ "Gradient", "Gradient/Estimated", "Newton",
        "Newton/Estimated" ])
    savefig("derivative-optimization.png")

except ImportError:
    print "Gradient Optimizer: ", xd[-1]
    print "Gradient Optimizer with estimated derivatives: ", xe[-1]
    print "Newton Optimizer: ", xn[-1]
    print "Newton Optimizer with estimated derivatives: ", xq[-1]