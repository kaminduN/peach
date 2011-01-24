################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/norms-conorms.py
# How to use t-norms and s-norms (norms and conorms)
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
import numpy
from peach.fuzzy import *
from peach.fuzzy.norms import *


# The standard operations with sets -- and thus fuzzy sets -- are intersection,
# union and complement. Fuzzy sets, however, are an extension to classical sets,
# and there are infinite ways to extend those operations. Thus the existence of
# norms, conorms and negations. We show here how to use them in Peach.

# First, remember that we must create the sets. A FuzzySet instance is returned
# when you apply a membership function over a domain. It is, in fact, a
# standard array, but making it a new class allow us to redefine operations.
# Here we create the sets:
x = numpy.linspace(-5.0, 5.0, 500)
a = Triangle(-3.0, -1.0, 1.0)(x)
b = Triangle(-1.0, 1.0, 3.0)(x)

# To set norms, conorms and negations, we use, respectively, the methods
# set_norm, set_conorm and set_negation. Notice that those are class methods, so
# if you change the norm for one instance of a set, you change for them all! So,
# it is better to use the class name to select the methods. Here, we will use
# Zadeh norms, that are already defined in Peach. Notice that we use the
# standard operators for and, or and not operations (respectively, &, | e ~):
FuzzySet.set_norm(ZadehAnd)
FuzzySet.set_conorm(ZadehOr)
aandb_zadeh = a & b             # A and B
aorb_zadeh = a | b              # A or B

# Probabilistic norms are based on the corresponding operations in probability.
# Here we use them
FuzzySet.set_norm(ProbabilisticAnd)
FuzzySet.set_conorm(ProbabilisticOr)
aandb_prob = a & b
aorb_prob = a | b

# There are other norms that we could use. Please, check the documentation for
# a complete list. Here are some of them:
# Norms: ZadehAnd, ProbabilisticAnd, DrasticProduct, EinsteinProduct
# Conorms: ZadehOr, ProbabilisticOr, DrasticSum, EinsteinSum

# We will use the matplotlib module to plot these functions. We save the plot in
# a figure called 'norms-conorms.png'.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8, 6)
    a1 = axes([ 0.125, 0.555, 0.775, 0.40 ])
    a2 = axes([ 0.125, 0.125, 0.775, 0.40 ])

    a1.hold(True)
    a1.plot(x, a, 'k:')
    a1.plot(x, b, 'k:')
    a1.plot(x, aandb_zadeh, 'k')
    a1.plot(x, aandb_prob, 'k-.')
    a1.set_xlim([ -5, 5 ])
    a1.set_ylim([ -0.1, 1.1 ])
    a1.set_xticks([])
    a1.set_yticks([ 0.0, 1.0 ])
    a1.legend((r'$A$', r'$B$', 'Zadeh AND', 'Prob. AND'))

    a2.hold(True)
    a2.plot(x, a, 'k:')
    a2.plot(x, b, 'k:')
    a2.plot(x, aorb_zadeh, 'k')
    a2.plot(x, aorb_prob, 'k-.')
    a2.set_xlim([ -5, 5 ])
    a2.set_ylim([ -0.1, 1.1 ])
    a2.set_xticks([])
    a2.set_yticks([ 0.0, 1.0 ])
    a2.legend((r'$A$', r'$B$', 'Zadeh OR', 'Prob. OR'))

    savefig("norms-conorms.png")

except ImportError:
    pass
