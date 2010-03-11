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
from numpy import min, sum, argsort


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

      __init__(self, f)
        Initialization method. The initialization procedure should take at least
        one parameter, ``f``, which is the function to be maximized. Other
        parameters could be used, but sensible default values should be
        available.

      __call__(self, population)
        This method is called to calculate population fitness. There is no
        recomendation about the internals of the method, but its signature is
        expected as defined above. This method receives a population -- please,
        consult the ``ga`` module for more information on populations -- and
        should return a vector or list with the fitness value for each
        chromosome in the same order that they appear in the population.

      This class implements the standard normalization fitness, as described in
      every book and article about GAs. The rank given to a chromosome is
      proportional to its objective function value.
    '''
    def __init__(self, f):
        '''
        Initializes the operator.

        :Parameters:
          f
            The cost function to be maximized. If you need to minimize a
            function, negate the return value.
        '''
        self.f = f
        '''Objective function to be maximized. Handle with care -- although it
        can be changed, it might cause trouble if you do so.'''


    def __call__(self, population):
        '''
        Calculates the fitness for all individuals in the population.

        :Parameters:
          population
            The population to be processed. Please, consult the ``ga`` module
            for more information on populations. This method calculates the
            fitness according to the traditional normalization technique.

        :Returns:
          A vector containing the fitness value for every individual in the
          population, in the same order that they appear there.
        '''
        f = population.fitness
        for j, c in enumerate(population):
            f[j] = self.f(c.decode())
        f = f - min(f)
        population.fitness = f / sum(f)
        return population.fitness


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
    def __init__(self, f):
        '''
        Initializes the operator.

        :Parameters:
          f
            The cost function to be maximized. If you need to minimize a
            function, negate the return value.
        '''
        self.f = f
        '''Objective function to be maximized. Handle with care -- although it
        can be changed, it might cause trouble if you do so.'''


    def __call__(self, population):
        '''
        Calculates the fitness for all individuals in the population.

        :Parameters:
          population
            The population to be processed. Please, consult the ``ga`` module
            for more information on populations. This method calculates the
            fitness according to the equally spaced ranking technique.

        :Returns:
          A vector containing the fitness value for every individual in the
          population, in the same order that they appear there.
        '''
        f = population.fitness
        for j, c in enumerate(population):
            f[j] = self.f(c.decode())
        f = f - min(f)
        f = (argsort(f) + 1.) / len(population)
        population.fitness = f / sum(f)
        return population.fitness

################################################################################