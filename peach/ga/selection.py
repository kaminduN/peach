################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: ga/chrossover.py
# Basic definitions for crossover among chromosomes
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic classes and definitions for selection operator.

The first step in a genetic algorithm is the selection of the fittest
individuals. The selection method typically uses the fitness of the population
to compute which individuals are closer to the best solution. However, instead
of deterministically deciding which individuals continue to the next generation,
they are randomly choosen, the chances of an individual being choosen given by
its fitness value. This sub-module implements selection methods.
"""


################################################################################
from numpy import add
from numpy.random import uniform
from random import randrange
from chromosome import *


################################################################################
# Classes
################################################################################
class Selection(object):
    '''
    Base class for selection operators.

    This class should be subclassed if you want to create your own selection
    operator. The base class doesn't do much, it is only a prototype. As is done
    with all the base classes within this library, use the ``__init__`` method
    to configure your selection behaviour -- if needed -- and the ``__call__``
    method to operate over a population.

    A class derived from this one should implement at least 2 methods, defined
    below:

      __init__(self, *cnf, **kw)
        Initializes the object. There is no mandatory arguments, but any
        parameters can be used here to configure the operator. A default value
        should always be offered, if possible.

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
class RouletteWheel(Selection):
    '''
    The Roulette Wheel selection method.

    This method randomly chooses a new population with the same size of the
    original population. An individual is choosen with a probability
    proportional to its fitness value, independent of what fitness method was
    used. This is usually abstracted as a roulette wheel in texts about the
    subject. Please, note that the selection is done *in loco*, that is,
    although the new population is returned, it is not a new list -- it is the
    same list as before, but with values changed.
    '''
    def __call__(self, population):
        '''
        Selects the population.

        :Parameters:
          population
            The list of chromosomes that should be operated over. The given list
            is modified, so be aware that the old generation will not be
            available after stepping the GA.

        :Returns:
          The new population.
        '''
        facc = add.accumulate(population.fitness)
        newp = [ ]
        for j in xrange(len(population)):
            rs = uniform(0., 1.)
            si = 0
            while rs > facc[si]:
                si = si + 1
            newp.append(Chromosome(population[si]))
        population[:] = newp
        return population


################################################################################
class BinaryTournament(Selection):
    '''
    The Binary Tournament selection method.

    This method randomly chooses a new population with the same size of the
    original population. Two individuals are choosen at random and they
    "battle", the fittest surviving for the next generation. Please, note that
    the selection is done *in loco*, that is, although the new population is
    returned, it is not a new list -- it is the same list as before, but with
    values changed.
    '''
    def __call__(self, population):
        '''
        Selects the population.

        :Parameters:
          population
            The list of chromosomes that should be operated over. The given list
            is modified, so be aware that the old generation will not be
            available after stepping the GA.

        :Returns:
          The new population.
        '''
        facc = add.accumulate(population.fitness)
        newp = [ ]
        l = len(population)
        for j in xrange(l):
            m = randrange(l)
            n = randrange(l)
            if facc[m] > facc[n]:
                newp.append(Chromosome(population[m]))
            else:
                newp.append(Chromosome(population[n]))
        population[:] = newp
        return population


################################################################################
class Baker(Selection):
    '''
    The Baker selection method.

    This method is very similar to the Roulette Wheel, but instead or randomly
    choosing every new member on the next generation, only the first probability
    is randomized. The others are determined as equally spaced numbers from 0 to
    1, from this number. Please, note that the selection is done *in loco*, that
    is, although the new population is returned, it is not a new list -- it is
    the same list as before, but with values changed.
    '''
    def __call__(self, population):
        '''
        Selects the population.

        :Parameters:
          population
            The list of chromosomes that should be operated over. The given list
            is modified, so be aware that the old generation will not be
            available after stepping the GA.

        :Returns:
          The new population.
        '''
        facc =  add.accumulate(population.fitness)
        newp = [ ]
        cs = uniform(0., 1.)
        si = 0
        l = len(population)
        l1 = 1. / l
        for j in xrange(l):
            while facc[si] < cs:
                si = (si + 1) % l
            newp.append(Chromosome(population[si]))
            cs = (cs + l1) % 1
        population[:] = newp
        return population

################################################################################
