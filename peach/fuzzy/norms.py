################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/norms.py
# Norms, conorms and negations
################################################################################


# Doc string, reStructuredText formatted:
__doc__ = """
This package implements operations of fuzzy logic.

Basic operations are ``and (&)``, ``or (|)`` and ``not (~)``. Those are
implemented as functions of, respectively, two, two and one values. The ``and``
is the t-norm of the fuzzy logic, and it is a function that takes two values and
returns the result of the ``and`` operation. The ``or`` is a function that takes
two values and returns the result of the ``or`` operation. the ``not`` is a
function that takes one value and returns the result of the ``not`` operation.
To implement your own operations there is no need to subclass -- just create the
functions and use them where appropriate.

Also, implication and aglutination functions are defined here. Implication is
the result of the generalized modus ponens used in fuzzy inference systems.
Aglutination is the generalization from two different conclusions used in fuzzy
inference systems. Both are implemented as functions that take two values and
return the result of the operation. As above, to implement your own operations,
there is no need to subclass -- just create the functions and use them where
appropriate.

The functions here are provided as convenience.
"""


################################################################################
import numpy


################################################################################
# Lofti Zadeh's basic operations
################################################################################
def ZadehAnd(x, y):
    '''
    And operation as defined by Lofti Zadeh.

    And operation is the minimum of the two values.

    :Returns:
      The result of the and operation.
    '''
    return numpy.minimum(x, y)

def ZadehOr(x, y):
    '''
    Or operation as defined by Lofti Zadeh.

    Or operation is the maximum of the two values.

    :Returns:
      The result of the or operation.
    '''
    return numpy.maximum(x, y)

def ZadehNot(x):
    '''
    Not operation as defined by Lofti Zadeh.

    Not operation is the complement to 1 of the given value, that is, ``1 - x``.

    :Returns:
      The result of the not operation.
    '''
    return 1 - x

def ZadehImplication(x, y):
    '''
    Implication operation as defined by Zadeh.

    :Returns:
      The result of the implication.
    '''
    return numpy.maximum(numpy.minimum(x, y), 1. - x)

ZADEH_NORMS = (ZadehAnd, ZadehOr, ZadehNot)
'Tuple containing, in order, Zadeh and, or and not operations'


################################################################################
# Drastic product and sum
################################################################################
def DrasticProduct(x, y):
    '''
    Drastic product that can be used as and operation

    :Returns:
      The result of the and operation
    '''
    return numpy.select([ x == 1., y == 1. ], [ y, x ], 0.)

def DrasticSum(x, y):
    '''
    Drastic sum that can be used as or operation

    :Returns:
      The result of the or operation
    '''
    return numpy.select([ x == 0., y == 0. ], [ y, x ], 1.)

DRASTIC_NORMS = (DrasticProduct, DrasticSum, ZadehNot)
'''Tuple containing, in order, Drastic product (and), Drastic sum (or) and Zadeh
not operations'''


################################################################################
# Einstein product and sum
################################################################################
def EinsteinProduct(x, y):
    '''
    Einstein product that can be used as and operation.

    :Returns:
      The result of the and operation.
    '''
    return (x*y) / (2. - (x + y - x*y))

def EinsteinSum(x, y):
    '''
    Einstein sum that can be used as or operation.

    :Returns:
      The result of the or operation.
    '''
    return (x + y) / (1. + x*y)

EINSTEIN_NORMS = (EinsteinProduct, EinsteinSum, ZadehNot)
'''Tuple containing, in order, Einstein product (and), Einstein sum (or) and
Zadeh not operations'''


################################################################################
# Mamdani's basic operations
################################################################################
def MamdaniImplication(x, y):
    '''
    Implication operation as defined by Mamdani.

    Implication is the minimum of the two values.

    :Returns:
      The result of the implication.
    '''
    return numpy.minimum(x, y)

def MamdaniAglutination(x, y):
    '''
    Aglutination as defined by Mamdani.

    Aglutination is the maximum of the two values.

    :Returns:
      The result of the aglutination.
    '''
    return numpy.maximum(x, y)

MAMDANI_INFERENCE = (MamdaniImplication, MamdaniAglutination)
'Tuple containing, in order, Mamdani implication and algutination'


################################################################################
# Probabilistic operations
################################################################################
def ProbabilisticAnd(x, y):
    '''
    And operation as a probabilistic operation.

    And operation is the product of the two values.

    :Returns:
      The result of the and operation.
    '''
    return x*y

def ProbabilisticOr(x, y):
    '''
    Or operation as a probabilistic operation.

    Or operation is given as the probability of the intersection of two events,
    that is, x + y - xy.

    :Returns:
      The result of the or operation.
    '''
    return x + y - x*y

def ProbabilisticNot(x):
    '''
    Not operation as a probabilistic operation.

    Not operation is the complement to 1 of the given value, that is, ``1 - x``.

    :Returns:
      The result of the not operation.
    '''
    return 1 - x

def ProbabilisticImplication(x, y):
    '''
    Implication as a probabilistic operation.

    Implication is the product of the two values.

    :Returns:
      The result of the and implication.
    '''
    return x*y

def ProbabilisticAglutination(x, y):
    '''
    Implication as a probabilistic operation.

    Implication is given as the probability of the intersection of two events,
    that is, x + y - xy.

    :Returns:
      The result of the and algutination.
    '''
    return x + y - x*y

PROB_NORMS = (ProbabilisticAnd, ProbabilisticOr, ProbabilisticNot)
'Tuple containing, in order, probabilistic and, or and not operations'

PROB_INFERENCE = (ProbabilisticImplication, ProbabilisticAglutination)
'Tuple containing, in order, probabilistic implication and algutination'


################################################################################
# Other implications
################################################################################
def DienesRescherImplication(x, y):
    '''
    Natural implication as in truth table, defined by Dienes-Rescher

    :Returns:
      The result of the implication.
    '''
    return numpy.maximum(1.-x, y)

def LukasiewiczImplication(x, y):
    '''
    Implication of the Lukasiewicz three-valued logic.

    :Returns:
      The result of the implication.
    '''
    return numpy.minimum(1., 1. - x + y)

def GodelImplication(x, y):
    '''
    Implication as defined by Godel.

    :Returns:
      The result of the implication.
    '''
    return numpy.select([ x < y ], [ 1. ], y)


################################################################################
# Test
if __name__ == "__main__":
    pass
