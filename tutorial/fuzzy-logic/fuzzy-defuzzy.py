################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/fuzzy-defuzzy.py
# Fuzzification and defuzzification are not complementary operations
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
import numpy
from peach.fuzzy import *


# To demonstrate that fuzzifying and subsequently defuzzifying a crisp set does
# not correspond to identity function, we will iterate over an interval to see
# what happens.

# The FlatSaw function is a very handy function that creates a set of membership
# functions equally spaced over an interval. The border functions are ramps, and
# the middle functions are triangles. This distribution of functions is very
# common in fuzzy control.
x = numpy.linspace(-2., 2., 500)
xgn, xpn, xz, xpp, xgp = FlatSaw((-2., 2.), 5)
y = numpy.zeros((500, ), dtype=float)

# We iterate over the domain, first fuzzifying the value of the variable x on
# those membership functions, then defuzzifying it using the Centroid function.
# Here, we use the value of the x variable itself to cut the membership
# functions.
for i in xrange(0, 500):
    yt = xgn(x[i]) & xgn(x) |\
         xpn(x[i]) & xpn(x) |\
         xz(x[i]) & xz(x) |\
         xpp(x[i]) & xpp(x) |\
         xgp(x[i]) & xgp(x)
    y[i] = Centroid(yt, x)

# Just to show what happens we will plot the procedure for a given value of the
# variable x.
x0 = -0.67
y0 = xgn(x0) & xgn(x) |\
     xpn(x0) & xpn(x) |\
     xz(x0) & xz(x) |\
     xpp(x0) & xpp(x) |\
     xgp(x0) & xgp(x)


# We will use the matplotlib module to plot these functions. We save the plot in
# a figure called 'fuzzy-defuzzy.png'.
try:
    import pylab

    pylab.figure(1).set_size_inches(8., 6.)

    pylab.subplot(211)
    pylab.hold(True)
    pylab.plot(x, xgn(x), 'k-')
    pylab.plot(x, xpn(x), 'k-')
    pylab.plot(x, xz(x), 'k-')
    pylab.plot(x, xpp(x), 'k-')
    pylab.plot(x, xgp(x), 'k-')
    pylab.plot(x, y0, 'k-')
    pylab.fill(x, y0, 'gray')
    pylab.plot([ x0, x0 ], [ -0.1, 1.1 ], 'k--')
    pylab.xticks([ x0 ])
    pylab.figure(1).axes[0].set_xticklabels([ r'$x_0$' ])
    pylab.ylim([ -0.1, 1.1 ])
    pylab.yticks([ 0., 0.25, 0.5, 0.75, 1. ])

    pylab.subplot(212)
    pylab.hold(True)
    pylab.plot(x, x, 'k--')
    pylab.plot(x, y, 'k')
    pylab.plot([ x0, x0 ], [ -2., 2. ], 'k--')
    pylab.xticks([ x0 ])
    pylab.figure(1).axes[1].set_xticklabels([ r'$x_0$' ])
    pylab.yticks([ ])

    pylab.savefig('fuzzy-defuzzy.png')

except ImportError:
    pass