################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: pso/base.py
# Basic particle swarm optimization
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements the simple continuous version of the particle swarm
optimizer. In this implementation, it is possible to specify, besides the
objective function and the first estimates, the ranges of search, which will
influence the max velocity of the particles, and the population size. Other
parameters are available too, please refer to the rest of this documentation for
further details.
"""


from numpy import array, argmin, amin, amax, where
from numpy.random import random, uniform
from acc import *


################################################################################
# Classes
################################################################################
class ParticleSwarmOptimizer(list):
    '''
    A standard Particle Swarm Optimizer

    This class implements a particle swarm optimization (PSO) procedure. A
    swarm is a list of estimates, and should answer to every ``list`` method. A
    population of particles is created to travel through the search domain with
    a certain velocity. At each point, the objective function is evaluated for
    each particle, and the positions are adjusted correspondingly. The velocity
    is then modified (ie, the particles are accelerated) towards its 'personal'
    best (the best value found by that particle at the moment) and a global best
    (the best value found overall at the moment).
    '''
    def __init__(self, f, x0, ranges=None, accelerator=StandardPSO, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        :Parameters:
          f
            A multivariable function to be evaluated. It must receive only one
            parameter, a multidimensional line-vector with the same dimensions
            of the range list (see below) and return a real value, a scalar.

          x0
            A population of first estimates. This is a list, array or tuple of
            one-dimension arrays, each one corresponding to an estimate of the
            position of the minimum. The population size of the algorithm will
            be the same as the number of estimates in this list. Each component
            of the vectors in this list are one of the variables in the function
            to be optimized.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If this parameter is not supplied, then the ranges will
            be computed from the estimates, but be aware that this might not
            represent the complete search space. If supplied, this parameter
            should be a list of ranges for each variable of the objective
            function. It is specified as a list of tuples of two values,
            ``(x0, x1)``, where ``x0`` is the start of the interval, and ``x1``
            its end. Obviously, ``x0`` should be smaller than ``x1``. It can
            also be given as a list with a simple tuple in the same format. In
            that case, the same range will be applied for every variable in the
            optimization.

          accelerator
            An acceleration method, please consult the documentation on ``acc``
            module. Defaults to StandardPSO, that is, velocities change based on
            local and global bests.

          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.

          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        list.__init__(self, [ ])
        self.__fx = [ ]
        for x in x0:
            x = array(x).ravel()
            self.append(x)
            self.__fx.append(f(x))
        self.__f = f

        # Determine ranges of the variables
        if ranges is None:
            ranges = zip(amin(self, axis=0), amax(self, axis=1))
        else:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        # Randomly computes the initial velocities
        s = len(self)
        d = len(x0[0])
        r = self.ranges
        self.__v = (random((s, d)) - 0.5) * (r[:, 1] - r[:, 0])/10.

        # Verifies the validity of the acceleration method
        try:
            issubclass(accelerator, Accelerator)
            accelerator = accelerator(self)
        except TypeError:
            pass
        if not isinstance(accelerator, Accelerator):
            raise TypeError, 'not a valid acceleration method'
        else:
            self.__acc = accelerator

        self.__emax = emax
        self.__imax = imax


    def __get_fx(self):
        return self.__fx
    fx = property(__get_fx, None)
    '''Array containing the objective function values for each estimate in the
    swarm.'''


    def __get_best(self):
        m = argmin(self.__fx)
        return self[m]
    best = property(__get_best, None)
    '''Single vector containing the position of the best point found by all the
    particles. Not writeable.'''


    def __get_fbest(self):
        m = argmin(self.__fx)
        return self.__fx[m]
    fbest = property(__get_fbest, None)
    '''Single scalar value containing the function value of the best point by
    all the particles. Not writeable.'''


    def restart(self, x0):
        '''
        Resets the optimizer, allowing the use of a new set of estimates. This
        can be used to avoid stagnation

        :Parameters:
          x0
            A new set of estimates. It doesn't need to have the same size of the
            original swarm, but it must be a list of estimates in the same
            format as in the object instantiation. Please, see the documentation
            on the instantiation of the class. New velocities will be computed.
        '''
        self[:] = [ ]
        self.__fx = [ ]
        f = self.__f
        for x in x0:
            x = array(x).ravel()
            self.append(x)
            self.__fx.append(f(x))

        # Randomly computes the initial velocities
        s = len(self)
        d = len(x0[0])
        r = self.ranges
        self.__v = (random((s, d)) - 0.5) * (r[:, 1] - r[:, 0])/10.


    def step(self):
        '''
        Computes the new positions of the particles, a step of the algorithm.

        This method updates the velocity given the constants associated with the
        particle and global bests; and then updates the positions accordingly.

        This method has no parameters and returns no values. The particles
        positions can be consulted with the ``[]`` interface (as a swarm of
        particles is a list of estimates), ``best`` property, to find the global
        best, and ``fbest`` property to find the minimum (see above).
        '''
        oldbest = self.best
        f = self.__f
        p = array(self)
        v = self.__acc(self.__v)

        # Next estimates
        p = p + v

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            p = where(p < r0, uniform(r0, r1, p.shape), p)
            p = where(p > r1, uniform(r0, r1, p.shape), p)

        # Update state
        self.__v = v
        self[:] = list(p)
        for i in xrange(len(self)):
            self.__fx[i] = f(p[i])
        best = self.best
        return best, abs(best - oldbest)/best


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


class PSO(ParticleSwarmOptimizer):
    '''
    PSO is an alias to ``ParticleSwarmOptimizer``
    '''
    pass


################################################################################
# Test
if __name__ == "__main__":

    def f(xy):
        x, y = xy
        return (1-x)**2 + (y-x*x)**2

    i = 0
    x0 = random((5, 2))*2
    #p = ParticleSwarmOptimizer(f, x0, [ (0., 2.), (0., 2.) ])
    p = ParticleSwarmOptimizer(f, x0)
    while p.fbest > 5e-7:
        print p
        print p.best
        print p.fbest
        p.step()
        i = i + 1
        print '-'*50
    print i, p.best, p.fbest