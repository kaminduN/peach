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
class GeneticAlgorithm(list):
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
    def __init__(self, f, x0, ranges=[ ], fmt='f', fitness=Fitness,
                 selection=RouletteWheel, crossover=TwoPoint,
                 mutation=BitToBit, elitist=True):
        '''
        Initializes the population and the algorithm.

        On the initialization of the population, a lot of parameters can be set.
        Those will deeply affect the results. The parameters are:

        :Parameters:
          f
            A multivariable function to be evaluated. The nature of the
            parameters in the objective function will depend of the way you want
            the genetic algorithm to process. It can be a standard function that
            receives a one-dimensional array of values and computes the value of
            the function. In this case, the values will be passed as a tuple,
            instead of an array. This is so that integer, floats and other types
            of values can be passed and processed. In this case, the values will
            depend of the format string (see below)

            If you don't supply a format, your objective function will receive a
            ``Chromosome`` instance, and it is the responsability of the
            function to decode the array of bits in any way. Notice that, while
            it is more flexible, it is certainly more difficult to deal with.
            Your function should process the bits and compute the return value
            which, in any case, should be a scalar.

            Please, note that genetic algorithms maximize functions, so project
            your objective function accordingly. If you want to minimize a
            function, return its negated value.

          x0
            A population of first estimates. This is a list, array or tuple of
            one-dimension arrays, each one corresponding to an estimate of the
            position of the minimum. The population size of the algorithm will
            be the same as the number of estimates in this list. Each component
            of the vectors in this list are one of the variables in the function
            to be optimized.

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

            It can be specified as a list of tuples with the same format as
            given above. In that case, the list must have one range for every
            variable specified in the format and the ranges must appear in the
            same order as there. That is, every variable must have a range
            associated to it.

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

            It defaults to `"f"`, that is, a single floating point variable.

          fitness
            A fitness method to be applied over the objective function. This
            parameter must be a ``Fitness`` instance or subclass. It will be
            applied over the objective function to compute the fitness of every
            individual in the population. Please, see the documentation on the
            ``Fitness`` class.

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
        self.__fx = [ ]
        for x in x0:
            x = array(x).ravel()
            c = Chromosome(fmt)
            c.encode(tuple(x))
            self.append(c)
            self.__fx.append(f(x))
        self.__f = f
        self.__csize = self[0].size
        self.elitist = elitist
        '''If ``True``, then the population is elitist.'''

        if type(fmt) == int:
            self.ranges = None
        elif ranges is None:
            self.ranges = zip(amin(self, axis=0), amax(self, axis=1))
        else:
            ranges = list(ranges)
            if len(ranges) == 1:
                self.ranges = array(ranges * len(x0[0]))
            else:
                self.ranges = array(ranges)
                '''Holds the ranges for every variable. Although it is a
                writable property, care should be taken in changing parameters
                before ending the convergence.'''

        # Sanitizes the first estimate. It is not expected that the values
        # received as first estimates are outside the ranges, but a check is
        # made anyway. If any estimate is outside the bounds, a new random
        # vector is choosen.
        if self.ranges is not None: self.sanity()

        # Verifies the validity of the fitness method
        try:
            issubclass(fitness, Fitness)
            fitness = fitness()
        except TypeError:
            pass
        if not isinstance(fitness, Fitness):
            raise TypeError, 'not a valid fitness function'
        else:
            self.__fit = fitness
        self.__fitness = self.__fit(self.__fx)

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


    def __get_fx(self):
        return self.__fx
    fx = property(__get_fx, None)
    '''Array containing the fitness value for every estimate in the
    population. Not writeable.'''


    def __get_best(self):
        m = argmax(self.__fx)
        return self[m]
    best = property(__get_best, None)
    '''Single vector containing the position of the best point found by all the
    individuals. Not writeable.'''


    def __get_fbest(self):
        m = argmax(self.__fx)
        return self.__fx[m]
    fbest = property(__get_fbest, None)
    '''Single scalar value containing the function value of the best point by
    all the individuals. Not writeable.'''


    def __get_fit(self):
        return self.__fitness
    fitness = property(__get_fit, None)
    '''Vector containing the fitness value for every individual in the
    population. This is not the same as the objective function value. Not
    writeable.'''


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


    def restart(self, x0):
        '''
        Resets the optimizer, allowing the use of a new set of estimates. This
        can be used to avoid stagnation.

        :Parameters:
          x0
            A new set of estimates. It doesn't need to have the same size of the
            original population, but it must be a list of estimates in the same
            format as in the object instantiation. Please, see the documentation
            on the instantiation of the class.
        '''
        self.__fx = [ ]
        for x in x0:
            x = array(x).ravel()
            c = Chromosome(fmt)
            c.encode(tuple(x))
            self.append(c)
            self.__fx.append(f(x))


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
        f = self.__f
        if self.elitist:
            max_fit = Chromosome(self.best)
        self.__select(self)
        if self.__crossover is not None: self.__crossover(self)
        if self.__mutate is not None: self.__mutate(self)
        if self.ranges is not None: self.sanity()
        if self.elitist:
            self[0] = max_fit
        self.__fx = [ f(c.decode()) for c in self ]
        self.__fitness = self.__fit(self.__fx)
        return self.best, self.fbest


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            x, e = self.step()
            i = i + 1
        return x, e


class GA(GeneticAlgorithm):
    '''
    GA is an alias to ``GeneticAlgorithm``
    '''
    pass

################################################################################
