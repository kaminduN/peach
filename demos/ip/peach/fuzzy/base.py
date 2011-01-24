################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/fuzzy.py
# Fuzzy logic basic definitions
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements basic definitions for fuzzy logic
"""


################################################################################
import numpy
import types

import norms


################################################################################
# Classes
################################################################################
class FuzzySet(numpy.ndarray):
    '''
    Array containing fuzzy values for a set.

    This class defines the behavior of a fuzzy set. It is an array of values in
    the range from 0 to 1, and the basic operations of the logic -- and (using
    the ``&`` operator); or (using the ``|`` operator); not (using ``~``
    operator) -- can be defined according to a set of norms. The norms can be
    redefined using the appropriated methods.

    To create a FuzzySet, instantiate this class with a sequence as argument,
    for example::

        fuzzy_set = FuzzySet([ 0., 0.25, 0.5, 0.75, 1.0 ])
    '''
    __AND__ = norms.ZadehAnd
    'Class variable to hold the *and* method'

    __OR__ = norms.ZadehOr
    'Class variable to hold the *or* method'

    __NOT__ = norms.ZadehNot
    'Class variable to hold the *not* method'

    def __new__(cls, data):
        '''
        Allocates space for the array.

        A fuzzy set is derived from the basic NumPy array, so the appropriate
        functions and methods are called to allocate the space. In theory, the
        values for a fuzzy set should be in the range ``0.0 <= x <= 1.0``, but
        to increase efficiency, no verification is made.

        :Returns:
          A new array object with the fuzzy set definitions.
        '''
        data = numpy.array(data, dtype=float)
        shape = data.shape
        data = numpy.ndarray.__new__(cls, shape=shape, buffer=data,
                                          dtype=float, order=False)
        return data.copy()

    def __init__(self, data=[]):
        '''
        Initializes the object.

        Operations are defaulted to Zadeh norms ``(max, min, 1-x)``
        '''
        pass

    def __and__(self, a):
        '''
        Fuzzy and (``&``) operation.
        '''
        return FuzzySet(FuzzySet.__AND__(self, a))

    def __or__(self, a):
        '''
        Fuzzy or (``|``) operation.
        '''
        return FuzzySet(FuzzySet.__OR__(self, a))

    def __invert__(self):
        '''
        Fuzzy not (``~``) operation.
        '''
        return FuzzySet(FuzzySet.__NOT__(self))

    @classmethod
    def set_norm(cls, f):
        '''
        Selects a t-norm (and operation)

        Use this method to change the behaviour of the and operation.

        :Parameters:
          f
            A function of two parameters which must return the ``and`` of the
            values.
        '''
        if isinstance(f, numpy.vectorize):
            cls.__AND__ = f
        elif isinstance(f, types.FunctionType):
            cls.__AND__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'

    @classmethod
    def set_conorm(cls, f):
        '''
        Selects a t-conorm (or operation)

        Use this method to change the behaviour of the or operation.

        :Parameters:
          f
            A function of two parameters which must return the ``or`` of the
            values.
        '''
        if isinstance(f, numpy.vectorize):
            cls.__OR__ = f
        elif isinstance(f, types.FunctionType):
            cls.__OR__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'

    @classmethod
    def set_negation(cls, f):
        '''
        Selects a negation (not operation)

        Use this method to change the behaviour of the not operation.

        :Parameters:
          f
            A function of one parameter which must return the ``not`` of the
            value.
        '''
        if isinstance(f, numpy.vectorize):
            cls.__NOT__ = f
        elif isinstance(f, types.FunctionType):
            cls.__NOT__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'


################################################################################
# Test
if __name__ == "__main__":
    pass