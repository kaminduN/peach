Defuzzification
===============

The main application of fuzzy logic is in the form of fuzzy controllers. Fuzzy
controllers do their job in three steps: *fuzzification*, where crisp values
(taken from sensors, for example) are converted to membership values;
*production rules*, that stablish the relationship between input variables and
output variables; and *defuzzification*, returnin from fuzzy sets to crisp
numbers.

Defuzzification is usually a simple step -- for a computer, that is. In general,
there are some computation to be done, and, while the operations are very easy
to understand, they're very ellaborate to do manually. In this tutorial, we show
how to use Peach to perform defuzzification. Notice that defuzzification is part
of a process (there is another tutorial covering controllers in a more complete
way), and we will only simulate here the first two steps.

We import ``numpy`` for arrays and ``peach`` for the library::

    import numpy
    from peach import *

Just to illustrate the method, we will create arbitrary fuzzy sets. In a
controller, these functions would be obtained by fuzzification and a set of
production rules. But our intent here is to show how to use the defuzzification
methods. Remember that instantiating ``Membership`` functions gives a function,
so we must apply it over our domain. Remember, also, that these functions return
``FuzzySet`` instances::

    y = numpy.linspace(-30.0, 30.0, 500)
    gn = Triangle(-30.0, -20.0, -10.0)(y)
    pn = Triangle(-20.0, -10.0, 0.0)(y)
    z = Triangle(-10.0, 0.0, 10.0)(y)
    pp = Triangle(0.0, 10.0, 20.0)(y)
    gp = Triangle(10.0, 20.0, 30.0)(y)

Here, ``y`` is the domain of the output variable. We simulate now the response
of the production rules of a controller. In it, a controller will associate a
membership value with every membership function of the output variable. You will
notice that no membership values are associated with ``pp`` and ``gp``
functions. That is because we are assuming that the results of the corresponding
production rules are 0, effectivelly eliminating those functions::

    mf = gn & 0.33 | pn & 0.67 | z & 0.25

If you use expressions like this, it is extremely easy to program a controller.
Just think of the ``&`` operator as implication, and ``|`` as aglutination. But
Peach has better ways to deal with that.

Here are the defuzzification methods -- if you need more information on them,
consult the literature on the subject. Notice that it is a simple function call,
not a class instantiation::

    centroid = Centroid(mf, y)                 # Centroid method
    bisec = Bisector(mf, y)                    # Bissection method
    som = SmallestOfMaxima(mf, y)              # Smallest of Maxima
    lom = LargestOfMaxima(mf, y)               # Largest of Maxima
    mom = MeanOfMaxima(mf, y)                  # Mean of Maxima

If you want to try your own defuzzification method, creating one is very easy:
just program it as a function. There is no need to instantiate or create
objects. Defuzzification methods receive, as their first parameter, the fuzzy
set to be defuzzified, and as second parameter the domain of the output
variable. It should return the defuzzified value. Every method works that way --
and that's what Peach expects when it needs one.

The figure below shows the results of the defuzzification.

.. image:: figs/defuzzification.png
   :align: center


