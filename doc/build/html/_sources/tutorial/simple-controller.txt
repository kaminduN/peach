Simple Controller
=================

This tutorial shows how to work with a fuzzy-based controller. It is really easy
to build a standard controller using Peach. We won't go into details of how a
controller should work -- please, consult the literature on the subject, as it
is very rich and explains the topic a lot better than we could do here.

We will show how to build a simple single-input single-output controller for no
specific plant -- it will be completelly abstract. The goal is to show how to
work with the capabilities built in Peach for dealing with it. A Mamdani
controller has, typically, three steps: fuzzification, in which numerical values
are converted to the fuzzy domain; decision rules, where the relationship
between controlled variable and manipulated variable are stablished; and
defuzzification, where we travel back from fuzzified domain to crisp numerical
values.

To build a controller, thus, we need to specify the membership functions of the
controlled variable. There are a number of ways of doing that (please, see the
tutorial on membership functions for more details): we could use built-in
membership functions; define our own membership functions; or use a support
function, such as the one below.

Suppose we wanted to use three membership functions to fuzzify our input
variable: a decreasing ramp from -1 to 0, a triangle ramp from -1 to 0 to 1, and
an increasing ramp from 0 to 1. We could define these functions as::

    i_neg = DecreasingRamp(-1, 0)
    i_zero = Triangle(-1, 0, 1)
    i_pos = IncreasingRamp(0, 1)

There is nothing wrong with this method. But, since sequences of triangles are
so usual in fuzzy controllers, Peach has two methods to create them in a batch.
The first one is the ``Saw`` function: given an interval and a number of
functions, it splits the interval in equally spaced triangles. The second one is
the ``FlatSaw`` function: it also creates a sequence of equally spaced
triangles, but use a decreasing ramp as the first function, and an increasing
function as the last one. Both of them return a tuple containing the functions
in order. The same functions above could be created with the command::

    i_neg, i_zero, i_pos = FlatSaw((-2, 2), 3)

assuming, that is, that the input variable will range from -2 to 2. Notice that
if we don't use the correct interval, the starts and ends of the functions won't
fall where we want them. Notice, also, that we are here using membership
functions, not fuzzy sets! If we iterate these functions over the given interval
and plot the results, we will get something similar to the figure below:

.. image:: figs/simple-controller-mf.png
   :align: center

We will also need to create membership functions for the output variable. Let's
assume we need three functions as above, in the range from -10 to 10. We do::

    o_neg, o_zero, o_pos = FlatSaw((-10, 10), 3)

The control will be done following the decision rules:

IF *input* is *negative* THEN *output* is *positive*

IF *input* is *zero* THEN *output* is *zero*

IF *input* is *positive* THEN *output* is *negative*

We will now create the controller that will implement these rules. Here is what
we do::

    Points = 100
    yrange = numpy.linspace(-10., 10., 500)
    c = Controller(yrange)

Here, ``yrange`` is the interval in which the output variable is defined, and it
is the only mandatory parameter in the creation of the controller. There are
some other parameters that we can use to customize how it works. To create a
controller, we instantiate the ``Controller`` class with the following
parameters::

    c = Controller(yrange, rules=[], defuzzy=Centroid, norm=ZadehAnd,
                   conorm=ZadehOr, negation=ZadehNot, imply=MamdaniImplication,
                   aglutinate=MamdaniAglutination):

Here is what means these parameters:

    yrange
        The range of the output variable. This must be given as a set of points
        belonging to the interval where the output variable is defined, not only
        the start and end points. It is strongly suggested that the interval is
        divided in some (eg.: 100) points equally spaced;

    rules
        The set of decision rules, as defined below. This must be given as a
        list of rules. If none is given, an empty set of rules is assumed;

    defuzzy
        The defuzzification method to be used. If none is given, the Centroid
        method is used;

    norm
        The norm (``and`` operation) to be used. Defaults to Zadeh and. The norm
        is used join the conditions in every rule

    conorm
        The conorm (``or`` operation) to be used. Defaults to Zadeh or.

    negation
        The negation (``not`` operation) to be used. Defaults to Zadeh not.

    imply
        The implication method to be used. Defaults to Mamdani implication.

    aglutinate
        The aglutination method to be used. Defaults to Mamdani aglutination.

So, as it is easy to see, this is a standard Mamdani controller. As created, our
controller doesn't have any rules, so we must add them. To add rules to a
controller, we use the ``add_rule`` method. A rule is a tuple with the following
format::

    ((input_mf, ), output_mf)

where ``input_mf`` is the condition, and ``output_mf`` is the consequence. This
format can be used to control multiple variables. For instance, if you wanted to
control three variables, a rule would have the form::

    ((input1_mf, input2_mf, input3_mf), output_mf)

Notice that the conditions are wrapped in a tuple themselves. We will add the
rules of our controller now::

    c.add_rule(((i_neg,), o_pos))
    c.add_rule(((i_zero,), o_zero))
    c.add_rule(((i_pos,), o_neg))

Besides the ``add_rule`` method, the controller has some other methods to
perform other tasks. Please, consult the documentation on the ``Controller`` for
more information. Of these methods, the most important is the ``__call__``
interface, that we use to pass to the controller the values of the variables (in
the form of a n-dimension array), and it returns us the result. So, if we want
to know what the result of the controller would be for (say) input 0.23, we just
issue the command::

    >> c(0.23)
    -1.53472428319

In this tutorial, we will iterate over the domain of the input variable to plot
the transfer function::

    x = numpy.linspace(-2., 2., Points)
    y = [ ]
    for x0 in x:
        y.append(c(x0))
    y = numpy.array(y)

By using the ``matplotlib`` module, we can plot this function to obtain the
transfer function of this controller. This is a very simple controller, so we
don't expect this transfer function to represent much, but it is interesting to
notice how a very simple controller can give a nice non-linear response:

.. image:: figs/simple-controller.png
   :align: center