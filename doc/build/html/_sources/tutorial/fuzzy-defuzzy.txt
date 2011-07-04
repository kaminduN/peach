Defuzzification is not the Complementary Operation of Fuzzification
===================================================================

The process of applying fuzzy logic in a problema consists, typically, of three
steps: fuzzify an input variable, apply production rules to give the relation
between the input and the output variables, and defuzzify the output. Of course,
there is a lot of detail to that, but people tend to think: if the production
rules are such that the fuzzified output is equal to the fuzzified input, you
have an identity -- that is, their crips values are the same.

Unfortunatelly, that is not so. The defuzzification is, at best, a method to
estimate what was the value of the variable that is in accordance to the
corresponding fuzzy set. As an estimate, it is prone to errors. This tutorial
shows the fact. We will iterate over a variable, fuzzifying and subsequently
defuzzifying it.

We import ``numpy`` for arrays and ``peach`` for the library::

    import numpy
    from peach import *

Next, we create the membership functions to the input variable. We will use the
``FlatSaw`` function, which is a very handy function that creates a set of
membership functions equally spaced over an interval. The border functions are
ramps, and the middle functions are triangles. This distribution of functions is
very common in fuzzy control. All you need to do is pass the interval as a tuple
and the number of membership functions. If you don't want the border functions
to be ramps, you could use the ``Saw`` function -- it does basically the same,
but returns only triangle functions::

    x = numpy.linspace(-2., 2., 500)
    xgn, xpn, xz, xpp, xgp = FlatSaw((-2., 2.), 5)
    y = numpy.zeros((500, ), dtype=float)

Here, ``y`` will hold the response of the processing we do below. We apply the
value of the ``x`` variable itself to generate the fuzzified value of the output
variable, and defuzzify it::

    for i in xrange(0, 500):
        yt = xgn(x[i]) & xgn(x) |\
            xpn(x[i]) & xpn(x) |\
            xz(x[i]) & xz(x) |\
            xpp(x[i]) & xpp(x) |\
            xgp(x[i]) & xgp(x)
        y[i] = Centroid(yt, x)

The figure below shows the results of the process. Notice that, although there
is some errors, they're not big -- if they're acceptable will depend on the
application. Also, notice that the big errors are in the border of the interval.
We would get bigger errors there if we used triangle functions in the border. We
might get better results if we used bell membership functions or more functions.
But perfection is unatanainable for every value of *x*, except, maybe, with an
infinite number of membership functions.

.. image:: figs/fuzzy-defuzzy.png
   :align: center


