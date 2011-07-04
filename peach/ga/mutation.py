################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: ga/mutation.py
# Basic definitions for mutation on chromosomes
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic definitions and classes for operating mutation on chromosomes.

The mutation operator changes selected bits in the array corresponding to the
chromosome. This operation is not as common as the others, but some genetic
algorithms still implement it.
"""

################################################################################
from numpy.random import uniform


################################################################################
# Classes
################################################################################
class Mutation(object):
    '''
    Base class for mutation operators.

    This class should be subclassed if you want to create your own mutation
    operator. The base class doesn't do much, it is only a prototype. As is done
    with all the base classes within this library, use the ``__init__`` method
    to configure your mutation behaviour -- if needed -- and the ``__call__``
    method to operate over a population.

    A class derived from this one should implement at least 2 methods, defined
    below:

      __init__(self, *cnf, **kw)
        Initializes the object. There is no mandatory arguments, but any
        parameters can be used here to configure the operator. For example, a
        class can define a mutation rate -- this should be defined here::

          __init__(self, rate=0.75)

        A default value should always be offered, if possible.

      __call__(self, population)
        The ``__call__`` implementation should receive a population and operate
        over it. Please, consult the ``ga`` module to see more information on
        populations. It should return the processed population. No recomendation
        on the internals of the method is made.

    Please, note that the GA implementations relies on this behaviour: it will
    pass a population to your ``__call__`` method and expects to received the
    result back.
    '''
    pass


################################################################################
class BitToBit(Mutation):
    '''
    A simple bit-to-bit mutation operator.

    This operator scans every individual in the population, in a bit-to-bit
    fashion. If a uniformly random number is less than the mutation rate (see
    below), then the bit is inverted. The mutation should be made very small,
    since large populations will represent a big number of bits; it should never
    be more than 0.5.
    '''
    def __init__(self, rate=0.05):
        '''
        Initialize the mutation operator.

        :Parameters:
          rate
            Probability that a single bit in an individual will be inverted.
        '''
        self.rate = rate
        '''Property that contains the mutation rate.'''


    def __call__(self, population):
        '''
        Applies the operator over a population.

        The behaviour of this operator is as described above: it scans every bit
        in every individual, and if a random number is less than the mutation
        rate, the bit is inverted.

        :Parameters:
          population
            A list of ``Chromosomes`` containing the present population of the
            algorithm. It is processed and the results of the exchange are
            returned to the caller.

        :Returns:
          The processed population, a list of ``Chromosomes``.
        '''
        rate = self.rate
        for c in population:
            for j in xrange(population.chromosome_size):
                if uniform(0., 1.) < rate:
                    c[j] = ~c[j]
        return population

################################################################################
