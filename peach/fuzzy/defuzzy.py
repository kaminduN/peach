################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/defuzzy.py
# Defuzzification methods
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements defuzzification methods for use with fuzzy controllers.

Defuzzification methods take a set of numerical values, their corresponding
fuzzy membership values and calculate a defuzzified value for them. They're
implemented as functions, not as classes. So, to implement your own, use the
directions below.

These methods are implemented as functions with the signature ``(mf, y)``, where
``mf`` is the fuzzy set, and ``y`` is an array of values. That is, ``mf`` is a
fuzzy set containing the membership values of each one in the ``y`` array, in
the respective order. Both arrays should have the same dimensions, or else the
methods won't work.

See the example::

    >>> import numpy
    >>> from peach import *
    >>> y = numpy.linspace(0., 5., 100)
    >>> m_y = Triangle(1., 2., 3.)
    >>> Centroid(m_y(y), y)
    2.0001030715316435

The methods defined here are the most commonly used.
"""


################################################################################
import numpy
import types

from base import *


################################################################################
# Defuzzification methods
################################################################################

def Centroid(mf, y):
    '''
    Center of gravity method.

    The center of gravity is calculate using the standard formula found in any
    calculus book. The integrals are calculated using the trapezoid method.

    :Parameters:
      mf
        Fuzzy set containing the membership values of the elements in the
        vector given in sequence
      y
        Array of domain values of the defuzzified variable.

    :Returns:
      The center of gravity of the fuzzy set.
    '''
    return numpy.trapz(mf*y, y) / numpy.trapz(mf, y)


def Bissector(mf, y):
    '''
    Bissection method

    The bissection method finds a coordinate ``y`` in domain that divides the
    fuzzy set in two subsets with the same area. Integrals are calculated using
    the trapezoid method. This method only works if the values in ``y`` are
    equally spaced, otherwise, the method will fail.

    :Parameters:
      mf
        Fuzzy set containing the membership values of the elements in the
        vector given in sequence
      y
        Array of domain values of the defuzzified variable.

    :Returns:
      Defuzzified value by the bissection method.
    '''
    a2 = numpy.trapz(mf, y) / 2.0
    dy = y[1] - y[0]
    b = 0
    i = 0.0
    while i < a2:
        b = b + 1
        i = i + 0.5 * (mf[b] + mf[b-1])*dy
    return y[b]


def SmallestOfMaxima(mf, y):
    '''
    Smallest of maxima method.

    This method finds all the points in the domain which have maximum membership
    value in the fuzzy set, and returns the smallest of them.

    :Parameters:
      mf
        Fuzzy set containing the membership values of the elements in the
        vector given in sequence
      y
        Array of domain values of the defuzzified variable.

    :Returns:
      Defuzzified value by the smallest of maxima method.
    '''
    return y[numpy.argmax(mf)]


def LargestOfMaxima(mf, y):
    '''
    Largest of maxima method.

    This method finds all the points in the domain which have maximum membership
    value in the fuzzy set, and returns the largest of them.

    :Parameters:
      mf
        Fuzzy set containing the membership values of the elements in the
        vector given in sequence
      y
        Array of domain values of the defuzzified variable.

    :Returns:
      Defuzzified value by the largest of maxima method.
    '''
    return y[::-1][numpy.argmax(mf[::-1])]


def MeanOfMaxima(mf, y):
    '''
    Mean of maxima method.

    This method finds the smallest and largest of maxima, and returns their
    average.

    :Parameters:
      mf
        Fuzzy set containing the membership values of the elements in the
        vector given in sequence
      y
        Array of domain values of the defuzzified variable.

    :Returns:
      Defuzzified value by the  of maxima method.
    '''
    mn = y[numpy.argmax(mf)]
    mx = y[::-1][numpy.argmax(mf[::-1])]
    return 0.5*(mn + mx)


################################################################################
# Test
if __name__ == "__main__":
    pass