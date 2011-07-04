################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/defuzzification.py
# Defuzzification methods
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
import numpy
from peach.fuzzy import *


# The main application of fuzzy logic is in the form of fuzzy controllers. The
# last step on the control is the defuzzification, or returning from fuzzy sets
# to crisp numbers. Peach has a number of ways of dealing with that operation.
# Here we se how to do that.


# Just to illustrate the method, we will create arbitrary fuzzy sets. In a
# controller, these functions would be obtained by fuzzification and a set of
# production rules. But our intent here is to show how to use the
# defuzzification methods. Remember that instantiating Membership functions
# gives us a function, so we must apply it over our domain.
y = numpy.linspace(-30.0, 30.0, 500)
gn = Triangle(-30.0, -20.0, -10.0)(y)
pn = Triangle(-20.0, -10.0, 0.0)(y)
z = Triangle(-10.0, 0.0, 10.0)(y)
pp = Triangle(0.0, 10.0, 20.0)(y)
gp = Triangle(10.0, 20.0, 30.0)(y)

# Here we simulate the response of the production rules of a controller. In it,
# a controller will associate a membership value with every membership function
# of the output variable. Here we do that. You will notice that no membership
# values are associated with pp and gp functions. That is because we are
# supposing that they are 0, effectivelly eliminating those functions (we plot
# them anyway.
mf = gn & 0.33 | pn & 0.67 | z & 0.25

# Here are the defuzzification methods. Defuzzification methods are functions.
# They receive, as their first parameter, the membership function (or the fuzzy
# set) and as second parameter the domain of the output variable. Every method
# works that way -- and if you want to implement your own, use this signature.
# Notice that it is a simple function, not a class that is instantiated.
centroid = Centroid(mf, y)                 # Centroid method
bisec = Bisector(mf, y)                    # Bissection method
som = SmallestOfMaxima(mf, y)              # Smallest of Maxima
lom = LargestOfMaxima(mf, y)               # Largest of Maxima
mom = MeanOfMaxima(mf, y)                  # Mean of Maxima

# We will use the matplotlib module to plot these functions. We save the plot in
# a figure called 'defuzzification.png'.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8., 4.)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])
    ll = [ 0.0, 1.0 ]

    a1.hold(True)
    a1.plot([ centroid, centroid ], ll, linewidth = 1)
    a1.plot([ bisec, bisec ], ll, linewidth = 1)
    a1.plot([ som, som ], ll, linewidth = 1)
    a1.plot([ lom, lom ], ll, linewidth = 1)
    a1.plot([ mom, mom ], ll, linewidth = 1)
    a1.plot(y, gn, 'k--')
    a1.plot(y, pn, 'k--')
    a1.plot(y, z, 'k--')
    a1.plot(y, pp, 'k--')
    a1.plot(y, gp, 'k--')
    a1.fill(y, mf, 'gray')
    a1.set_xlim([ -30, 30 ])
    a1.set_ylim([ -0.1, 1.1 ])
    a1.set_xticks(linspace(-30, 30, 7.0))
    a1.set_yticks([ 0.0, 1.0 ])
    a1.legend([ 'Centroid = %7.4f' % centroid,
                'Bisector = %7.4f' % bisec,
                'SOM = %7.4f' % som,
                'LOM = %7.4f' % lom,
                'MOM = %7.4f' % mom ])
    savefig("defuzzification.png")

except ImportError:
    print "Defuzzification results:"
    print "  Centroid = %7.4f" % centroid
    print "  Bisector = %7.4f" % bisec
    print "  SOM = %7.4f" % som
    print "  LOM = %7.4f" % lom
    print "  MOM = %7.4f" % mom