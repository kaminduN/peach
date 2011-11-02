################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/simple-controller.py
# A simgle-input-single-output Mamdani controller
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
import numpy
from peach.fuzzy import *
import pylab as p


# This tutorial shows how to work with a fuzzy-based controller. It is really
# easy to build a standard controller using Peach. We won't go into details of
# how a controller should work -- please, consult the literature on the subject,
# as it is very rich and explains the topic a lot better than we could do here.
#
# We will show how to build a simple single-input single-output controller for
# no specific plant -- it will be completelly abstract. The goal is to show how
# to work with the capabilities built in Peach for dealing with it. A Mamdani
# controller has, typically, three steps: fuzzification, in which numerical
# values are converted to the fuzzy domain; decision rules, where the
# relationship between controlled variable and manipulated variable are
# stablished; and defuzzification, where we travel back from fuzzified domain to
# crisp numerical values.
#
# To build a controller, thus, we need to specify the membership functions of
# the controlled variable. There are a number of ways of doing that (please, see
# the tutorial on membership functions for more detail): we could use built-in
# membership functions; define our own membership functions; or use a support
# function, such as the one below.
#
# Suppose we wanted to use three membership functions to fuzzify our input
# variable: a decreasing ramp from -1 to 0, a triangle ramp from -1 to 0 to 1,
# and an increasing ramp from 0 to 1. We could define these functions as:
#
# i_neg = DecreasingRamp(-1, 0)
# i_zero = Triangle(-1, 0, 1)
# i_pos = IncreasingRamp(0, 1)
#
# Nothing wrong with this method. But, since sequences of triangles are so usual
# in fuzzy controllers, Peach has two methods to create them in a batch. The
# first one is the ``Saw`` function: given an interval and a number of
# functions, it splits the interval in equally spaced triangles. The second one
# is the ``FlatSaw`` function: it also creates a sequence of equally spaced
# triangles, but use a decreasing ramp as the first function, and an increasing
# function as the last one. Both of them return a tuple containing the functions
# in order. The same functions above could be created with the command:

i_neg, i_zero, i_pos = FlatSaw((-2, 2), 3)

# assuming, that is, that the input variable will range from -2 to 2. Notice
# that if we don't use the correct interval, the starts and ends of the
# functions won't fall where we want them. Notice, also, that we are here using
# membership functions, not fuzzy sets!

# We will also need to create membership functions for the output variable.
# Let's assume we need three functions as above, in the range from -10 to 10. We
# do:

o_neg, o_zero, o_pos = FlatSaw((-10, 10), 3)

# The control will be done following the decision rules:
#
# IF input is negative THEN output is positive
# IF input is zero THEN output is zero
# IF input is positive THEN output is negative
#
# We will create now the controller that will implement these rules. Here is
# what we do:

Points = 100
yrange = numpy.linspace(-10., 10., 500)
c = Controller(yrange)

# Here, ``yrange`` is the interval in which the output variable is defined. Our
# controlled doesn't have any rules, so we must add them. To add rules to a
# controller, we use the ``add_rule`` method. A rule is a tuple with the
# following format:
#
# ((input_mf, ), output_mf)
#
# where ``input_mf`` is the condition, and ``output_mf`` is the consequence.
# This format can be used to control multiple variables. For instance, if you
# wanted to control three variables, a rule would have the form:
#
# ((input1_mf, input2_mf, input3_mf), output_mf)
#
# Notice that the conditions are wrapped in a tuple themselves. We will add the
# rules of our controller now:

c.add_rule(((i_neg,), o_pos))
c.add_rule(((i_zero,), o_zero))
c.add_rule(((i_pos,), o_neg))

# The controller is ready to run. We use the ``__call__`` interface to pass to
# the controller the values of the variables (in the form of a n-dimension
# array), and it returns us the result. We will iterate over the domain of the
# input variable to plot the transfer function:

x = numpy.linspace(-2., 2., Points)
y = [ ]
for x0 in x:
    y.append(c(x0))
y = numpy.array(y)

# We will use the matplotlib module to plot these functions. We save the plot in
# a figure called 'simple-controller-mf.png', containing the membership
# functions, and another called 'simple-controller.png', containing the transfer
# function.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8., 4.)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.plot(x, i_neg(x))
    a1.plot(x, i_zero(x))
    a1.plot(x, i_pos(x))
    a1.set_xlim([ -2., 2. ])
    a1.set_ylim([ -0.1, 1.1 ])
    a1.legend([ 'Negative', 'Zero', 'Positive' ])
    savefig("simple-controller-mf.png")

    clf()
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])
    a1.plot(x, y, 'k-')
    a1.set_xlim([ -2., 2. ])
    a1.set_ylim([ -10., 10. ])
    savefig("simple-controller.png")

except ImportError:
    pass