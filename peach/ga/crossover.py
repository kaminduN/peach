################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: ga/chrossover.py
# Basic definitions for crossover among chromosomes
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic definitions for crossover operations and base classes.

Crossover is a very basic and important operation in genetic algorithms. It is
by means of crossover among the chromosomes that population gains diversity,
thus exploring more completelly the solution space and giving better answers.
This sub-module provides definitions of the most common crossover operations,
and provides a class that can be subclassed to construct different types of
crossover for experimentation.
"""


################################################################################
from numpy.random import uniform
from random import randrange


################################################################################
# Classes
################################################################################
class Crossover(object):
    '''
    Base class for crossover operators.

    This class should be subclassed if you want to create your own crossover
    operator. The base class doesn't do much, it is only a prototype. As is done
    with all the base classes within this library, use the ``__init__`` method
    to configure your crossover behaviour -- if needed -- and the ``__call__``
    method to operate over a population.

    A class derived from this one should implement at least 2 methods, defined
    below:

      __init__(self, *cnf, **kw)
        Initializes the object. There are no mandatory arguments, but any
        parameters can be used here to configure the operator. For example, a
        class can define a crossover rate -- this should be defined here::

          __init__(self, rate=0.75)

        A default value should always be offered, if possible.

      __call__(self, population)
        The ``__call__`` implementation should receive a population and operate
        over it. Please, consult the ``ga`` module to see more information on
        populations. It should return the processed population. No recomendation
        on the internals of the method is made. That being said, in general the
        crossover operators pairs chromosomes and swap bits among them (but
        there is nothing to say that you can't do it differently).

    Please, note that the GA implementations relies on this behaviour: it will
    pass a population to your ``__call__`` method and expects to received the
    result back.
    '''
    pass


################################################################################
class OnePoint(Crossover):
    '''
    A one-point crossover operator.

    A one-point crossover randomly selects a single point in two chromosomes and
    swaps the bits among them from that point until the end of the bit stream.
    The crossover rate is the probability that two paired chromosomes will
    exchange bits.
    '''
    def __init__(self, rate=0.75):
        '''
        Initialize the crossover operator.

        :Parameters:
          rate
            Probability that two paired chromosomes will exchange bits.
        '''
        self.rate = rate
        '''Property that contains the crossover rate.'''


    def __call__(self, population):
        '''
        Proceeds the crossover over a population.

        In one-point crossover, chromosomes from a population are randomly
        paired. If a uniform random number is below the ``rate`` given in the
        instantiation of the operator, then a random point is selected and bits
        from that point until the end of the chromosomes are exchanged.

        :Parameters:
          population
            A list of ``Chromosomes`` containing the present population of the
            algorithm. It is processed and the results of the exchange are
            returned to the caller.

        :Returns:
          The processed population, a list of ``Chromosomes``.
        '''
        rate = self.rate
        chromosize = population.chromosome_size
        size = len(population)
        for j in xrange(int((size-1)/2)*2, 2):
            if uniform(0., 1.) <= rate:
                pos = randrange(chromosize)
                tmp = population[j][pos:]
                population[j][pos:] = population[j+1][pos:]
                population[j+1][pos:] = tmp
        return population


################################################################################
class TwoPoint(Crossover):
    '''
    A two-point crossover operator.

    A two-point crossover randomly selects two points in two chromosomes and
    swaps the bits among them between these points. The crossover rate is the
    probability that two paired chromosomes will exchange bits.
    '''
    def __init__(self, rate=0.75):
        '''
        Initialize the crossover operator.

        :Parameters:
          rate
            Probability that two paired chromosomes will exchange bits.
        '''
        self.rate = rate
        '''Property that contains the crossover rate.'''


    def __call__(self, population):
        '''
        Proceeds the crossover over a population.

        In two-point crossover, chromosomes from a population are randomly
        paired. If a uniform random number is below the ``rate`` given in the
        instantiation of the operator, then random points are selected and bits
        between those points are exchanged.

        :Parameters:
          population
            A list of ``Chromosomes`` containing the present population of the
            algorithm. It is processed and the results of the exchange are
            returned to the caller.

        :Returns:
          The processed population, a list of ``Chromosomes``.
        '''
        rate = self.rate
        chromosize = population.chromosome_size
        size = len(population)
        for j in xrange(int((size-1)/2)*2, 2):
            if uniform(0., 1.) <= rate:
                ipos = randrange(chromosize)
                epos = randrange(chromosize)
                if epos < ipos:
                    ipos, epos = epos, ipos
                tmp = population[j][ipos:epos]
                population[j][ipos:epos] = population[j+1][ipos:epos]
                population[j+1][ipos:epos] = tmp
        return population


################################################################################
class Uniform(Crossover):
    '''
    A uniform crossover operator.

    A uniform crossover scans two chromosomes in a bit-to-bit fashion. According
    to a given crossover rate, the corresponding bits are exchanged. The
    crossover rate is the probability that two bits will be exchanged.
    '''
    def __init__(self, rate=0.75):
        '''
        Initialize the crossover operator.

        :Parameters:
          rate
            Probability that bits from two paired chromosomes will be exchanged.
        '''
        self.rate = rate
        '''Property that contains the crossover rate.'''


    def __call__(self, population):
        '''
        Proceeds the crossover over a population.

        In uniform crossover, chromosomes from a population are randomly paired,
        and scaned in a bit-to-bit fashion. If a uniform random number is below
        the ``rate`` given in the instantiation of the operator, then the bits
        under scan will be exchanged in the chromosomes.

        :Parameters:
          population
            A list of ``Chromosomes`` containing the present population of the
            algorithm. It is processed and the results of the exchange are
            returned to the caller.

        :Returns:
          The processed population, a list of ``Chromosomes``.
        '''
        rate = self.rate
        chromosize = population.chromosome_size
        size = len(population)
        for j in xrange(int((size-1)/2)*2, 2):
            for k in xrange(chromosize):
                if uniform(0., 1.) <= rate:
                    tmp = population[j][k]
                    population[j][k] = population[j+1][k]
                    population[j+1][k] = tmp
        return population

################################################################################
