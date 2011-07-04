################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: pso/acc.py
# Functions to update the velocity of particles in a swarm.
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Functions to update the velocity (ie, accelerate) of the particles in a swarm.

Acceleration of a particle is an important concept in the theory of particle
swarm optimizers. By choosing an adequate acceleration, particle velocity is
changed so that they can search the domain of definition of the objective
function such that there is a greater probability that a global minimum is
found. Since particle swarm optimizers are derived from genetic algorithms, it
can be said that this is what creates diversity in a swarm, such that the space
is more thoroughly searched.
"""


################################################################################
from numpy import array, sqrt, abs, select, sign
from numpy.random import random


################################################################################
# Classes
################################################################################
class Accelerator(object):
    '''
    Base class for accelerators.

    This class should be derived to implement a function which computes the
    acceleration of a vector of particles in a swarm. Every accelerator function
    should implement at least two methods, defined below:

      __init__(self, *cnf, **kw)
        Initializes the object. There are no mandatory arguments, but any
        parameters can be used here to configure the operator. For example, a
        class can define a variance for randomly chose the acceleration -- this
        should be defined here::

          __init__(self, variance=1.0)

        A default value should always be offered, if possible.

      __call__(self, v):
        The ``__call__`` interface should be programmed to actually compute the
        new velocity of a vector of particles. This method should receive a
        velocity in ``v`` and use whatever parameters from the instantiation to
        compute the new velocities. Notice that this function should operate
        over a vector of velocities, not on a single velocity. This class,
        however, can be instantiated with a single function that is adapted to
        perform over a vector.
    '''
    def __init__(self, f):
        '''
        Initializes an accelerator object.

        This method initializes an accelerator. It receives as argument a simple
        function that is adapted to operate over a vector of velocities.

        :Parameters:
          f
            The function to be used as acceleration. This function can be simple
            function that receives a ``n``-dimensional vector representing the
            velocity of a single particle, where ``n`` is the dimensionality of
            the objective function. The object then wraps the function such that
            it can receive a list of velocities and applies the acceleration on
            every one of them.
        '''
        self.__f = f


    def __call__(self, v):
        '''
        Computes new velocities for every particle.

        This method should be overloaded in implementations of different
        accelerators. This method receives the velocities as a list or a vector
        of the velocities (a ``n``-dimensional vector in each line) or each
        particle in a swarm and computes, for each one of them, a new velocity.

        :Parameters:
          v
            A list or a vector of velocities, where each velocity is one line of
            the vector or one element of the list.

        :Returns:
          A vector of the same size as the argument with the updated velocities.
          The returned vector is returned as a bidimensional array.
        '''
        vn = [ self.__f(vi) for vi in list(v) ]
        return array(vn)


################################################################################
class StandardPSO(Accelerator):
    '''
    Standard PSO Accelerator

    This class implements a method for changing the velocities of particles in
    a particle swarm. The standard way is to retain information on local bests
    and the global bests, and update the velocity based on that.
    '''
    def __init__(self, ps, vmax=None, cp=2.05, cg=2.05):
        '''
        Initializes the accelerator.

        :Parameters:
          ps
            A reference to the Particle Swarm that should be updated. This
            class, in instantiation, will assume that the position of the
            particles in the moment of creation are the local best. The
            objective function is computed for all particles, and the values
            saved for reference in the future. Also, at the same time, the
            global best is computed.

          cp
            The velocity adjustment constant associated with the particle best
            values. Defaults to 2.05.

          cg
            The velocity adjustment constant associated with the global best
            values. Defaults to 2.05. The defaults in the ``cp`` and ``cg``
            parameters are such that the inertia weight in the constrition
            method satisfies ``cp + cg > 4``. Please, look in the bibliography
            for more information.
        '''
        self.__ps = ps
        self.__pbest = ps[:]
        self.__fpbest = ps.fx[:]
        self.__gbest = ps.best
        self.__fgbest = ps.fbest
        if vmax is None:
            self.__vmax = vmax
        else:
            vmax = array(vmax).ravel()
            if len(vmax) == 1:
                self.__vmax = vmax * ones((len(ps.x[0]), ))
            else:
                self.__vmax = vmax
        self.__vmax = 0.15 * (ps.ranges[:, 1] - ps.ranges[:, 0])
        self.cp = cp
        '''Velocity adjustment constant associated with the particle best values.'''
        self.cg = cg
        '''Velocity adjustment constant associated with the global best values.'''
        phi = cp + cg
        self.__k = 2./abs(2. - phi - sqrt((phi - 4.)*phi))


    def __call__(self, v):
        '''
        Computes the new velocities for every particle in the swarm. This method
        receives the velocities as a list or a vector of the velocities (a
        ``n``-dimensional vector in each line) or each particle in a swarm and
        computes, for each one of them, a new velocity.

        :Parameters:
          v
            A list or a vector of velocities, where each velocity is one line of
            the vector or one element of the list.

        :Returns:
          A vector of the same size as the argument with the updated velocities.
          The returned vector is returned as a bidimensional array.
        '''
        ps = self.__ps
        fx = ps.fx[:]

        # Updates local and global best.
        for i in xrange(len(fx)):
            if fx[i] < self.__fpbest[i]:
                self.__pbest[i] = ps[i]
                self.__fpbest[i] = fx[i]
            if fx[i] < self.__fgbest:
                self.__gbest = ps[i]
                self.__fgbest = fx[i]

        # Updates speed.
        ps = array(ps)
        s = ps.shape
        v = self.__k * (v + self.cp * random(s) * (self.__pbest - ps) \
                          + self.cg * random(s) * (self.__gbest - ps))
        vmax = self.__vmax
        if vmax is not None:
            v = select( [ v < vmax ], [ v ], sign(v)*vmax )
        return v
