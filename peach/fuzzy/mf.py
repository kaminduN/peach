################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/mf.py
# Membership functions for fuzzy logic
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Membership functions

Membership functions are actually subclasses of a main class called Membership,
see below. Instantiate a class to generate a function, optional arguments can be
specified to configure the function as needed. For example, to create a triangle
function starting at 0, with peak in 3, and ending in 4, use::

    mu = Triangle(0, 3, 4)

Please notice that the return value is a *function*. To use it, apply it as a
normal function. For example, the function above, applied to the value 1.5
should return 0.5::

    >>> print mu(1.5)
    0.5

"""


################################################################################
import numpy
from numpy import exp, cos, pi
import types

from base import *


################################################################################
# Membership functions
################################################################################
class Membership(object):
    '''
    Base class of all membership functions.

    This class is used as base of the implemented membership functions, and can
    also be used to transform a regular function in a membership function that
    can be used with the fuzzy logic package.

    To create a membership function from a regular function ``f``, use::

        mf = Membership(f)

    A function this converted can be used with vectors and matrices and always
    return a FuzzySet object. Notice that the value range is not verified so
    that it fits in the range [ 0, 1 ]. It is responsibility of the programmer
    to warrant that.

    To subclass Membership, just use it as a base class. It is suggested that
    the ``__init__`` method of the derived class allows configuration, and the
    ``__call__`` method is used to apply the function over its arguments.
    '''
    def __init__(self, f):
        '''
        Builds a membership function from a regular function

        :Parameters:
          f
            Function to be transformed into a membership function. It must be
            given, and it must be a ``FunctionType`` object, otherwise, a
            ``ValueError`` is raised.
        '''
        if isinstance(f, types.FunctionType):
            self.__f = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'


    def __call__(self, x):
        '''
        Maps the function on a vector

        :Parameters:
          x
            A value, vector or matrix over which the function is evaluated.

        :Returns:
          A ``FuzzySet`` object containing the evaluation of the function over
          each of the components of the input.
        '''
        return FuzzySet(self.__f(x))


################################################################################
class IncreasingRamp(Membership):
    '''
    Increasing ramp.

    Given two points, ``x0`` and ``x1``, with ``x0 < x1``, creates a function
    which returns:

       0, if ``x <= x0``;

       ``(x - x0) / (x1 - x0)``, if ``x0 < x <= x1``;

       1, if ``x > x1``.
    '''
    def __init__(self, x0, x1):
        '''
        Initializes the function.

        :Parameters:
          x0
            Start of the ramp;
          x1
            End of the ramp.
        '''
        self.__x0 = float(x0)
        self.__x1 = float(x1)
        self.__a = 1.0 / (self.__x1 - self.__x0)

    def __call__(self, x):
        y = numpy.select([ x < self.__x0, x < self.__x1 ],
                         [ 0.0, self.__a * (x - self.__x0) ], 1.0)
        return FuzzySet(y)


################################################################################
class DecreasingRamp(Membership):
    '''
    Decreasing ramp.

    Given two points, ``x0`` and ``x1``, with ``x0 < x1``, creates a function
    which returns:

       1, if ``x <= x0``;

       ``(x1 - x) / (x1 - x0)``, if ``x0 < x <= x1``;

       0, if ``x > x1``.
    '''
    def __init__(self, x0, x1):
        '''
        Initializes the function.

        :Parameters:
          x0
            Start of the ramp;
          x1
            End of the ramp.
        '''
        self.__x0 = float(x0)
        self.__x1 = float(x1)
        self.__a = 1.0 / (self.__x1 - self.__x0)

    def __call__(self, x):
        y = numpy.select([ x < self.__x0, x < self.__x1 ],
                         [ 1.0, self.__a * (self.__x1 - x) ], 0.0)
        return FuzzySet(y)


################################################################################
class Triangle(Membership):
    '''
    Triangle function.

    Given three points, ``x0``, ``x1`` and ``x2``, with ``x0 < x1 < x2``,
    creates a function which returns:

      0, if ``x <= x0`` or ``x > x2``;

      ``(x - x0) / (x1 - x0)``, if ``x0 < x <= x1``;

      ``(x2 - x) / (x2 - x1)``, if ``x1 < x <= x2``.
    '''
    def __init__(self, x0, x1, x2):
        '''
        Initializes the function.

        :Parameters:
          x0
            Start of the triangle;
          x1
            Peak of the triangle;
          x2
            End of triangle.
        '''
        self.__x0 = float(x0)
        self.__x1 = float(x1)
        self.__x2 = float(x2)
        self.__a0 = 1.0 / (self.__x1 - self.__x0)
        self.__a1 = 1.0 / (self.__x2 - self.__x1)

    def __call__(self, x):
        y = numpy.select([ x < self.__x0, x < self.__x1, x < self.__x2 ],
                         [ 0.0, self.__a0 * (x - self.__x0),
                                self.__a1 * (self.__x2 - x)], 0.0)
        return FuzzySet(y)


################################################################################
class Trapezoid(Membership):
    '''
    Trapezoid function.

    Given four points, ``x0``, ``x1``, ``x2`` and ``x3``, with
    ``x0 < x1 < x2 < x3``, creates a function which returns:

       0, if ``x <= x0`` or ``x > x3``;

       ``(x - x0)/(x1 - x0)``, if ``x0 <= x < x1``;

       1, if ``x1 <= x < x2``;

       ``(x3 - x)/(x3 - x2)``, if ``x2 <= x < x3``.
    '''
    def __init__(self, x0, x1, x2, x3):
        '''
        Initializes the function.

        :Parameters:
          x0
            Start of the trapezoid;
          x1
            First peak of the trapezoid;
          x2
            Last peak of the trapezoid;
          x3
            End of trapezoid.
        '''
        self.__x0 = float(x0)
        self.__x1 = float(x1)
        self.__x2 = float(x2)
        self.__x3 = float(x3)
        self.__a0 = 1.0 / (self.__x1 - self.__x0)
        self.__a1 = 1.0 / (self.__x3 - self.__x2)

    def __call__(self, x):
        y = numpy.select([ x < self.__x0, x < self.__x1,
                           x < self.__x2, x < self.__x3 ],
                         [ 0.0, self.__a0 * (x - self.__x0), 1.0,
                                self.__a1 * (self.__x3 - x) ], 0.0)
        return FuzzySet(y)


################################################################################
class Gaussian(Membership):
    '''
    Gaussian function.

    Given the center and the width, creates a function which returns a gaussian
    fit to these parameters, that is:

       ``exp(-(x - x0)**2)/a``
    '''
    def __init__(self, x0=0.0, a=1.0):
        '''
        Initializes the function.

        :Parameters:
          x0
            Center of the gaussian. Default value ``0.0``;
          a
            Width of the gaussian. Default value ``1.0``.
        '''
        self.__x0 = float(x0)
        self.__a = 1./float(a)

    def __call__(self, x):
        return FuzzySet(exp(- self.__a*(x - self.__x0)**2))


################################################################################
class IncreasingSigmoid(Membership):
    '''
    Increasing Sigmoid function.

    Given the center and the slope, creates an increasing sigmoidal function.
    It goes to ``0`` as ``x`` approaches to -infinity, and goes to ``1`` as
    ``x`` approaches infinity, that is:

        ``1 / (1 + exp(-a*(x - x0))``
    '''
    def __init__(self, x0=0.0, a=1.0):
        '''
        Initializes the function.

        :Parameters:
          x0
            Center of the sigmoid. Default value ``0.0``. The function evaluates
            to ``0.5`` if ``x = x0``;
          a
            Slope of the sigmoid. Default value ``1.0``.
        '''
        self.__x0 = float(x0)
        self.__a = float(a)

    def __call__(self, x):
        return FuzzySet(1.0 / (1.0 + exp(- self.__a*(x - self.__x0))))


################################################################################
class DecreasingSigmoid(Membership):
    '''
    Decreasing Sigmoid function.

    Given the center and the slope, creates an decreasing sigmoidal function.
    It goes to ``1`` as ``x`` approaches to -infinity, and goes to ``0`` as
    ``x`` approaches infinity, that is:

        ``1 / (1 + exp(a*(x - x0))``
    '''
    def __init__(self, x0=0.0, a=1.0):
        '''
        Initializes the function.

        :Parameters:
          x0
            Center of the sigmoid. Default value ``0.0``. The function evaluates
            to ``0.5`` if ``x = x0``;
          a
            Slope of the sigmoid. Default value ``1.0``.
        '''
        self.__x0 = float(x0)
        self.__a = float(a)

    def __call__(self, x):
        return FuzzySet(1.0 / (1.0 + exp(self.__a*(x - self.__x0))))


################################################################################
class RaisedCosine(Membership):
    '''
    Raised Cosine function.

    Given the center and the frequency, creates a function that is a period of
    a raised cosine, that is:

       0, if ``x <= xm - pi/w`` or ``x > xm + pi/w``;

       ``0.5 + 0.5 * cos(w*(x - xm))``, if ``xm - pi/w <= x < xm + pi/w``;
    '''
    def __init__(self, xm=0.0, w=1.0):
        '''
        Initializes the function.

        :Parameters:
          xm
            Center of the cosine. Default value ``0.0``. The function evaluates
            to ``1`` if ``x = xm``;
          w
            Frequency of the cosine. Default value ``1.0``.
        '''
        self.__xm = float(xm)
        self.__w = float(w)
        self.__x0 = self.__xm - pi / self.__w
        self.__x1 = self.__xm + pi / self.__w

    def __call__(self, x):
        y = numpy.select([ x < self.__x0, x < self.__x1 ],
                         [ 0.0, 0.5*cos(self.__w*(x - self.__xm)) + 0.5 ],
                           0.0)
        return FuzzySet(y)


################################################################################
class Bell(Membership):
    '''
    Generalized Bell function.

    A generalized bell is a symmetric function with its peak in its center and
    fast decreasing to ``0`` outside a given interval, that is:

      ``1 / (1 + ((x - x0)/a)**(2*b))``
    '''
    def __init__(self, x0=0.0, a=1.0, b=1.0):
        '''
        Initializes the function.

        :Parameters:
          x0
            Center of the bell. Default value ``0.0``. The function evaluates to
            ``1`` if ``x = xm``;
          a
            Size of the interval. Default value ``1.0``. A generalized bell
            evaluates to ``0.5`` if ``x = -a`` or ``x = a``;
          b
            Measure of *flatness* of the bell. The bigger the value of ``b``,
            the flatter is the resulting function. Default value ``1.0``.
        '''
        self.__x0 = float(x0)
        self.__a = float(a)
        self.__b = 2 * float(b)

    def __call__(self, x):
        return FuzzySet(1.0 / (1.0 + ((x - self.__x0)/self.__a)**self.__b))


################################################################################
class Smf(Membership):
    '''
    Increasing smooth curve with 0 and 1 minimum and maximum values outside a
    given range.
    '''
    def __init__(self, x0, x1):
        '''
        Initializes the function.

        :Parameters:
          x0
            Start of the curve. For every value below this, the function returns
            0;
          x1
            End of the curve. For every value above this, the function returns
            1;
        '''
        self.__x0 = x0
        self.__x1 = x1
        self.__xm = (x0 + x1) / 2.
        self.__xr = x1 - x0

    def __call__(self, x):
        xa = (x - self.__x0) / self.__xr
        xb = (x - self.__x1) / self.__xr
        y = numpy.select([ x < self.__x0, x < self.__xm, x < self.__x1 ],
                         [ 0., 2.*xa*xa, 1. - 2.*xb*xb ], 1.)
        return FuzzySet(y)


################################################################################
class Zmf(Membership):
    '''
    Decreasing smooth curve with 0 and 1 minimum and maximum values outside a
    given range.
    '''
    def __init__(self, x0, x1):
        '''
        Initializes the function.

        :Parameters:
          x0
            Start of the curve. For every value below this, the function returns
            1;
          x1
            End of the curve. For every value above this, the function returns
            0;
        '''
        self.__x0 = x0
        self.__x1 = x1
        self.__xm = (x0 + x1) / 2.
        self.__xr = x1 - x0

    def __call__(self, x):
        xa = (x - self.__x0) / self.__xr
        xb = (x - self.__x1) / self.__xr
        y = numpy.select([ x < self.__x0, x < self.__xm, x < self.__x1 ],
                         [ 1., 1. - 2.*xa*xa, 2.*xb*xb ], 0.)
        return FuzzySet(y)


################################################################################
# Auxiliary functions
################################################################################
def Saw(interval, n):
    '''
    Splits an ``interval`` into ``n`` triangle functions.

    Given an interval in any domain, this function will create ``n`` triangle
    functions of the same size equally spaced in the interval. It is very
    useful to create membership functions for controllers. The command below
    will create 3 triangle functions equally spaced in the interval (0, 4)::

        mf1, mf2, mf3 = Saw((0, 4), 3)

    This is the same as the following commands::

        mf1 = Triangle(0, 1, 2)
        mf2 = Triangle(1, 2, 3)
        mf3 = Triangle(2, 3, 4)

    :Parameters:
      interval
        A tuple containing the start and the end of the interval, in the format
        ``(start, end)``;
      n
        The number of functions in which the interval must be split.

    :Returns:
      A list of triangle membership functions, in order.
    '''
    xo, xf = interval
    dx = float(xf - xo)/float(n+1)
    mfs = [ ]
    for i in range(n):
        mfs.append(Triangle(xo, xo+dx, xo+2*dx))
        xo = xo + dx
    return mfs


################################################################################
def FlatSaw(interval, n):
    '''
    Splits an ``interval`` into a decreasing ramp, ``n-2`` triangle functions
    and an increasing ramp.

    Given an interval in any domain, this function will create a decreasing ramp
    in the start of the interval, ``n-2`` triangle functions of the same size
    equally spaced in the interval, and a increasing ramp in the end of the
    interval. It is very useful to create membership functions for controllers.
    The command below will create a decreasing ramp, a triangle function and an
    increasing ramp equally spaced in the interval (0, 2)::

        mf1, mf2, mf3 = FlatSaw((0, 2), 3)

    This is the same as the following commands::

        mf1 = DecreasingRamp(0, 1)
        mf2 = Triangle(0, 1, 2)
        mf3 = Increasingramp(1, 2)

    :Parameters:
      interval
        A tuple containing the start and the end of the interval, in the format
        ``(start, end)``;
      n
        The number of functions in which the interval must be split.

    :Returns:
      A list of corresponding functions, in order.
    '''
    xo, xf = interval
    dx = float(xf - xo)/float(n+1)
    mf1 = DecreasingRamp(xo+dx, xo+2*dx)
    mfs = Saw((xo+dx, xf-dx), n-2)
    mf2 = IncreasingRamp(xf-2*dx, xf-dx)
    return [ mf1 ] + mfs + [ mf2 ]


################################################################################
# Test.

if __name__ == "__main__":
    pass