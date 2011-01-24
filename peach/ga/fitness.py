################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: ga/fitness.py
# Basic definitions for declaring fitness functions
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic definitions and base classes for definition of fitness functions for use
with genetic algorithms.

Fitness is a function that rates higher the chromosomes that perform better
according to the objective function. For example, if the minimum of a function
needs to be found, then the fitness function should rate better the chromosomes
that correspond to lower values of the objective function. This module gives
support to use common Python functions as fitness functions in genetic
algorithms.

The classes defined in this sub-module take a function and use some algorithm to
rank a population. There are some different ranking functions, some are provided
in this module. There is also a class that can be subclassed to generate other
fitness methods. See the documentation of the corresponding class for more
information.
"""

################################################################################
from numpy import min, sum, argsort, zeros


################################################################################
# Classes
################################################################################
class Fitness(object):
    '''
    Base class for fitness function classifiers.

    This class is used as the base of all fitness functions. However, even if
    it is intended to be used as a base class, it also provides some
    functionality, described below.

    A subclass of this class should implement at least 2 methods:

      __init__(self, *args, **kwargs)
        Initialization method. The initialization procedure doesn't need to take
        any parameters, but if any configuration must be done, it should be
        passed as an argument to the ``__init__`` function. The genetic
        algorithm, however, does not expect parameters in the instantiation, so
        you should provide sensible defaults.

      __call__(self, fx)
        This method is called to calculate population fitness. There is no
        recomendation about the internals of the method, but its signature is
        expected as defined above. This method receives the values of the
        objective function applied over a population -- please, consult the
        ``ga`` module for more information on populations -- and should return a
        vector or list with the fitness value for each chromosome in the same
        order that they appear in the population.

      This class implements the standard normalization fitness, as described in
      every book and article about GAs. The rank given to a chromosome is
      proportional to its objective function value.
    '''
    def __init__(self):
        '''
        Initializes the operator.
        '''
        pass


    def __call__(self, fx):
        '''
        Calculates the fitness for all individuals in the population.

        :Parameters:
          fx
            The values of the objective function for every individual on the
            population to be processed. Please, consult the ``ga`` module for
            more information on populations. This method calculates the fitness
            according to the traditional normalization technique.

        :Returns:
          A vector containing the fitness value for every individual in the
          population, in the same order that they appear there.
        '''
        fx = fx - min(fx)
        return fx / sum(fx)


################################################################################
class Ranking(Fitness):
    '''
    Ranking fitness for a population

    Ranking gives fitness values equally spaced between 0 and 1. The fittest
    individual receives fitness equals to 1, the second best equals to 1 - 1/N,
    the third best 1 - 2/N, and so on, where N is the size of the population.
    It is important to note that the worst fit individual receives a fitness
    value of 1/N, not 0. That allows that no individuals are excluded from the
    selection operator.
    '''
    def __init__(self):
        '''
        Initializes the operator.
        '''
        Fitness.__init__(self)


    def __call__(self, fx):
        '''
        Calculates the fitness for all individuals in the population.

        :Parameters:
          fx
            The values of the objective function for every individual on the
            population to be processed. Please, consult the ``ga`` module for
            more information on populations. This method calculates the fitness
            according to the equally spaced ranking technique.

        :Returns:
          A vector containing the fitness value for every individual in the
          population, in the same order that they appear there.
        '''
        fx = fx - min(fx)
        fx = (argsort(fx) + 1.) / len(fx)
        return fx / sum(fx)

################################################################################