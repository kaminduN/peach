################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/membership-functions.py
# How to use pre-defined membership functions
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
import numpy
from peach.fuzzy import *


# Membership functions for representing fuzzy sets are available as classes in
# Peach. This way, by instantiating a membership function, you can configure and
# change default parameters to suit it to your needs.

# First, we create the domain in which we will represent the functions.
x = numpy.linspace(-5.0, 5.0, 500)

# Next, we create the functions by instantiating the corresponding classes.
# For more information on parameters of each function, please see the reference
# for the module. The functions thus created can be applied to numbers or arrays
# directly.

# An increasing ramp, starting in x=2 and ending in x=4
increasing_ramp = IncreasingRamp(2.0, 4.0)

# A decreasing ramp, starting in x=-2 and ending in x=-2
decreasing_ramp = DecreasingRamp(-4.0, -2.0)

# A triangle function, starting in x=-3, ending in x=0, with maximum in x=-1.5
triangle = Triangle(-3.0, -1.5, 0.0)

# A trapezoid, starting in x=-1, ending in x=3, with maximum from x=0 to x=2
trapezoid = Trapezoid(-1.0, 0.0, 2.0, 3.0)

# A gaussian with center x=-1.5 and default variance 1.
gaussian = Gaussian(-1.5)

# An increasing sigmoid, with middle point in x=3 and inclination 2.5
increasing_sigmoid = IncreasingSigmoid(3.0, 2.5)

# A decreasing ramp, with middle point in x=-3 and inclination 2.5
decreasing_sigmoid = DecreasingSigmoid(-3.0, 2.5)

# A generalized bell centered at x=1, with width=1.5 and exponent=4
bell = Bell(1.0, 1.5, 4.0)


# We will use the matplotlib module to plot these functions. Notice how the
# objects we instantiated before are used as functions over an array.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8, 6)
    a1 = axes([ 0.125, 0.555, 0.775, 0.40 ])
    a2 = axes([ 0.125, 0.125, 0.775, 0.40 ])

    a1.hold(True)
    a1.plot(x, decreasing_ramp(x))
    a1.plot(x, triangle(x))
    a1.plot(x, trapezoid(x))
    a1.plot(x, increasing_ramp(x))
    a1.set_xlim([ -5, 5 ])
    a1.set_ylim([ -0.1, 1.1 ])
    a1.set_xticks([])
    a1.set_yticks([ 0.0, 1.0 ])

    a2.hold(True)
    a2.plot(x, decreasing_sigmoid(x))
    a2.plot(x, gaussian(x))
    a2.plot(x, bell(x))
    a2.plot(x, increasing_sigmoid(x))
    #a2.plot(x, rc)
    a2.set_xlim([ -5, 5 ])
    a2.set_ylim([ -0.1, 1.1 ])
    a2.set_xticks([])
    a2.set_yticks([ 0.0, 1.0 ])

    savefig("membership-functions.png")

except ImportError:
    pass
