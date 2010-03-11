################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: ga/ga.py
# Basic genetic algorithm
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic Genetic Algorithm (GA)

This sub-package implements a traditional genetic algorithm as described in
books and papers. It consists of selecting, breeding and mutating a population
of chromosomes (arrays of bits) and reinserting the fittest individual from the
previous generation if the GA is elitist. Please, consult a good reference on
the subject, for the subject is way too complicated to be explained here.

Within the algorithm implemented here, it is possible to specify and configure
the selection, crossover and mutation methods using the classes in the
respective sub-modules and custom methods can be implemented (check
``selection``, ``crossover`` and ``mutation`` modules).

A GA object is actually a list of chromosomes. Please, refer to the
documentation of the class below for more information.
"""


################################################################################
from numpy import zeros, argmax, any, isnan, array
from numpy.random import uniform
import types

from chromosome import *
from fitness import *
from selection import *
from crossover import *
from mutation import *


################################################################################
# Classes
################################################################################
class GA(list):
    '''
    A standard Genetic Algorithm

    This class implements the methods to generate, initialize and evolve a
    population of chromosomes according to a given fitness function. A standard
    GA implements, in this order:

      - A selection method, to choose, from this generation, which individuals
        will be present in the next generation;
      - A crossover method, to exchange information between selected individuals
        to add diversity to the population;
      - A mutation method, to change information in a selected individual, also
        to add diversity to the population;
      - The reinsertion of the fittest individual, if the population is elitist
        (which is almost always the case).

    A population is actually a list of chromosomes, and individuals can be
    read and set as in a normal list. Use the ``[ ]`` operators to access
    individual chromosomes but please be aware that modifying the information on
    the list before the end of convergence can cause unpredictable results. The
    population and the algorithm have also other properties, check below to see
    more information on them.    '''
    def __init__(self, fitness, fmt, ranges=[ ], size=50,
                 selection=RouletteWheel, crossover=TwoPoint,
                 mutation=BitToBit, elitist=True):
        '''
        Initializes the population and the algorithm.

        On the initialization of the population, a lot of parameters can be set.
        Those will deeply affect the results. The parameters are:

        :Parameters:
          fitness
            A fitness function to serve as an objective function. In general, a
            GA is used for maximizing a function. This parameter can be a
            standard Python function or a ``Fitness`` instance.

            In the first case, the GA will convert the function in a
            ``Fitness`` instance and call it internally when needed. The
            function should receive a tuple or vector of data according to the
            given ``Chromosome`` format (see below) and return a numeric value.

            In the second case, you can use any of the fitness methods of the
            ``fitness`` sub-module, or create your own. If you want to use your
            own fitness method (for experimentation or simulation, for example),
            it must be an instance of a ``Fitness`` or of a subclass, or an
            exception will be raised. Please consult the documentation on the
            ``fitness`` sub-module.

          fmt
            A ``struct``-format string. The ``struct`` module is a standard
            Python module that packs and unpacks informations in bits. These
            are used to inform the algorithm what types of data are to be used.
            For example, if you are maximizing a function of three real
            variables, the format should be something like ``"fff"``. Any type
            supported by the ``struct`` module can be used. The GA will decode
            the bit array according to this format and send it as is to your 
            fitness function -- your function *must* know what to do with them.

            Alternatively, the format can be an integer. In that case, the GA
            will not try to decode the bit sequence. Instead, the bits are
            passed without modification to the objective function, which must
            deal with them. Notice that, if this is used this way, the
            ``ranges`` property (see below) makes no sense, so it is set to
            ``None``. Also, no sanity checks will be performed.

          ranges
            Since messing with the bits can change substantially the values
            obtained can diverge a lot from the maximum point. To avoid this,
            you can specify a range for each of the variables. ``range``
            defaults to ``[ ]``, this means that no range checkin will be done.
            If given, then every variable will be checked. There are two ways to
            specify the ranges.

            It might be a tuple of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. If ``range`` is given in this way, then this
            range will be used for every variable.

            If can be specified as a list of tuples with the same format as
            given above. In that case, the list must have one range for every
            variable specified in the format and the ranges must appear in the
            same order as there. That is, every variable must have a range
            associated to it.

          size
            This is the size of the population. It defaults to 50.

          selection
            This specifies the selection method. You can use one given in the
            ``selection`` sub-module, or you can implement your own. In any
            case, the ``selection`` parameter must be an instance of
            ``Selection`` or of a subclass. Please, see the documentation on the
            ``selection`` module for more information. Defaults to
            ``RouletteWheel``. If made ``None``, then selection will not be
            present in the GA.

          crossover
            This specifies the crossover method. You can use one given in the
            ``crossover`` sub-module, or you can implement your own. In any
            case, the ``crossover`` parameter must be an instance of
            ``Crossover`` or of a subclass. Please, see the documentation on the
            ``crossover`` module for more information. Defaults to
            ``TwoPoint``. If made ``None``, then crossover will not be
            present in the GA.

          mutation
            This specifies the mutation method. You can use one given in the
            ``mutation`` sub-module, or you can implement your own. In any
            case, the ``mutation`` parameter must be an instance of ``Mutation``
            or of a subclass. Please, see the documentation on the ``mutation``
            module for more information. Defaults to ``BitToBit``.  If made
            ``None``, then mutation will not be present in the GA.

          elitist
            Defines if the population is elitist or not. An elitist population
            will never discard the fittest individual when a new generation is
            computed. Defaults to ``True``.
        '''
        list.__init__(self, [ ])
        for i in xrange(size):
            self.append(Chromosome(fmt))
        if self[0].format is None:
            self.__nargs = 1
            self.ranges = None
        else:
            self.__nargs = len(self[0].decode())
            if not ranges:
                self.ranges = None
            elif len(ranges) == 1:
                self.ranges = array(ranges * self.__nargs)
            else:
                self.ranges = array(ranges)
                '''Holds the ranges for every variable. Although it is a writable
                property, care should be taken in changing parameters before ending
                the convergence.'''
        self.__csize = self[0].size
        self.elitist = elitist
        '''If ``True``, then the population is elitist.'''
        self.fitness = zeros((len(self),), dtype=float)
        '''Vector containing the computed fitness value for every individual.'''

        # Sanitizes the generated values randomly created for the chromosomes.
        if self.ranges is not None: self.sanity()

        # Verifies the validity of the fitness method
        if isinstance(fitness, types.FunctionType):
            fitness = Fitness(fitness)
        if not isinstance(fitness, Fitness):
            raise TypeError, 'not a valid fitness function'
        else:
            self.__fit = fitness
        self.__fit(self)

        # Verifies the validity of the selection method
        try:
            issubclass(selection, Selection)
            selection = selection()
        except TypeError:
            pass
        if not isinstance(selection, Selection):
            raise TypeError, 'not a valid selection method'
        else:
            self.__select = selection

        # Verifies the validity of the crossover method
        try:
            issubclass(crossover, Crossover)
            crossover = crossover()
        except TypeError:
            pass
        if not isinstance(crossover, Crossover) and crossover is not None:
            raise TypeError, 'not a valid crossover method'
        else:
            self.__crossover = crossover

        # Verifies the validity of the mutation method
        try:
            issubclass(mutation, Mutation)
            mutation = mutation()
        except TypeError:
            pass
        if not isinstance(mutation, Mutation) and mutation is not None:
            raise TypeError, 'not a valid mutation method'
        else:
            self.__mutate = mutation


    def __get_csize(self):
        return self.__csize
    chromosome_size = property(__get_csize, None)
    '''This property hold the chromosome size for the population. Not
    writable.'''


    def sanity(self):
        '''
        Sanitizes the chromosomes in the population.

        Since not every individual generated by the crossover and mutation
        operations might be a valid result, this method verifies if they are
        inside the allowed ranges (or if it is a number at all). Each invalid
        individual is discarded and a new one is generated.

        This method has no parameters and returns no values.
        '''
        r = self.ranges
        x0 = r[:, 0]
        x1 = r[:, 1]
        for c in self:
            xs = c.decode()
            if any(xs < x0) or any(xs > x1) or any(isnan(xs)):
                xs = [ uniform(r0, r1) for r0, r1 in r ]
                c.encode(tuple(xs))


    def fit(self):
        '''
        Computes the fitness for each individual of the population.

        This method is only an interface to the ``fitness`` function passed in
        the initialization. It calls the ``Fitness`` instance.

        This method has no parameters and returns no values.
        '''
        return self.__fit(self)


    def step(self):
        '''
        Computes a new generation of the population, a step of the adaptation.

        This method goes through all the steps of the GA, as described above. If
        the selection, crossover and mutation operators are defined, they are
        applied over the population. If the population is elitist, then the
        fittest individual of the past generation is reinserted.

        This method has no parameters and returns no values. The GA itself can
        be consulted (using ``[ ]``) to find the fittest individual which is the
        result of the process.
        '''
        if self.elitist:
            m = argmax(self.fitness)
            max_fit = Chromosome(self[m])
        self.__select(self)
        if self.__crossover is not None: self.__crossover(self)
        if self.__mutate is not None: self.__mutate(self)
        if self.ranges is not None: self.sanity()
        if self.elitist:
            self[0] = max_fit
        self.__fit(self)

################################################################################
