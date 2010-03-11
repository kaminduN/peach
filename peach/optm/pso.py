################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/pso.py
# Basic particle swarm optimization
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic Particle Swarm Optimization (PSO)

This sub-package implements traditional particle swarm optimizers as described
in literature. It consists of a very simple algorithm emulating the behaviour
of a flock of birds (though in a very simplified way). A population of particles
is created, each particle with its corresponding velocity. They fly towards the
particle and local best, thus exploring the whole domain.

Within this implementation, it is possible to specify the ranges of search,
which will influence the max velocity of the particles, and the population size.
Other parameters are available too, please refer to the rest of this
documentation for further details.

For efficiency purposes, the particles are represented internally as an array of
vectors. The particles can be acessed externally by the corresponding propery.
See below.
"""


from numpy import array, sqrt, abs, reshape, select
from numpy.random import rand


class ParticleSwarmOptimizer(object):
    '''
    A standard Particle Swarm Optimizer

    This class implements a particle swarm optimization (PSO) procedure. A
    population of particles is created to travel through the search domain with
    a certain velocity. At each point, the objective function is evaluated for
    each particle, and the positions are adjusted correspondingly. The velocity
    is then modified (ie, the particles are accelerated) towards its 'personal'
    best (the best value found by that particle at the moment) and a global best
    (the best value found overall at the moment).
    '''
    def __init__(self, f, ranges, size=10, cparticle=2.05, cglobal=2.05):
        '''
        Initializes the optimizer.

        :Parameters:
          f
            A multivariable function to be evaluated. It must receive only one
            parameter, a multidimensional line-vector with the same dimensions
            of the range list (see below) and return a real value, a scalar.
          ranges
            A range of values must be passed to the algorithm, so the max speed
            and the dimensionality of the function can be calculated. This
            parameter is a list of ranges for each variable of the optimized
            function. It is specified as a list of tuples of two values,
            ``(x0, x1)``, where ``x0`` is the start of the interval, and ``x1``
            its end. Obviously, ``x0`` should be smaller than ``x1``.
          size
            The size of the population. It defaults to 10.
          cparticle
            The velocity adjustment constant associated with the particle best
            values. Defaults to 2.05.
          cglobal
            The velocity adjustment constant associated with the global best
            values. Defaults to 2.05. The defaults in the ``cparticle`` and
            ``cglobal`` parameters are such that the inertia weight in the
            constrition method satisfies ``cparticle + cglobal > 4``. Please,
            look in the bibliography for more information.
        '''
        self.__f = f
        self.__size = size
        self.__ranges = array(ranges)
        self.__vmax = 0.15 * (self.__ranges[:, 1] - self.__ranges[:, 0])
        self.__p = self.__gen_particles()
        self.__v = self.__gen_velocities()
        self.__pbest = self.__p
        self.__fpbest = array([ f(x) for x in self.__pbest ]).reshape((size, 1))
        self.__gbest = self.__p[0]
        self.__fgbest = f(self.__gbest)
        self.cp = cparticle
        self.cg = cglobal
        phi = cparticle + cglobal
        self.__k = 2./abs(2. - phi - sqrt((phi - 4.)*phi))

    def __get_p(self):
        return self.__p
    def __set_p(self, p):
        self.__p = array(reshape(p, self.__p.shape))
    particles = property(__get_p, __set_p)
    '''Array containing the vectors representing the particles positions.'''

    def __get_pbest(self):
        return self.__pbest
    pbest = property(__get_pbest, None)
    '''Array containing the positions of the best points found by every particle.
    Not writeable.'''

    def __get_fpbest(self):
        return self.__fpbest
    fpbest = property(__get_fpbest, None)
    '''Array containing the function values of the best points found by every
    particle. Not writeable.'''

    def __get_gbest(self):
        return self.__gbest
    gbest = property(__get_gbest, None)
    '''Single vector containing the position of the best point found by all the
    particles. Not writeable.'''

    def __get_fgbest(self):
        return self.__fgbest
    fgbest = property(__get_fgbest, None)
    '''Single scalar value containing the function value of the best point by
    all the particles. Not writeable.'''

    def __gen_particles(self):
        '''
        Internal method to randomly generate a population of particles. It uses
        the ``ranges`` parameter.
        '''
        r = self.__ranges
        dim, _ = r.shape
        size = self.__size
        p = rand(size, dim) * (r[:, 1] - r[:, 0]) + r[:, 0]
        return p

    def __gen_velocities(self):
        '''
        Internal method to randomly generate the velocities of the particles. It
        uses the ``ranges`` parameter to calculate the maximum velocity.
        '''
        dim, _ = self.__ranges.shape
        size = self.__size
        v = rand(size, dim) * self.__vmax
        return v

    def step(self):
        '''
        Computes the new positions of the particles, a step of the algorithm.

        This method updates the velocity given the constants associated with the
        particle and global bests; and then updates the positions accordingly.
        Then, the particle bests and the global best are calculated and stored
        for future use.

        This method has no parameters and returns no values. The particles
        positions can be consulted with the ``particles``, ``pbest`` and
        ``gbest`` properties (see above).
        '''
        f = self.__f
        p = self.__p
        v = self.__v
        v = self.__k * (v + self.cp * rand() * (self.__pbest - p) \
                          + self.cg * rand() * (self.__gbest - p))
        v = select( [ v < self.__vmax ], [ v ], sign(v)*self.__vmax )
        p = p + v
        fg = self.__fgbest
        for i in xrange(self.__size):
            f0 = f(p[i])
            if (f0 < self.__fpbest[i]):
                self.__pbest[i] = p[i]
                self.__fpbest[i] = f0
            if (f0 < fg):
                self.__gbest = p[i]
                self.__fgbest = f0
        self.__p = p
        self.__v = v

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
    p = ParticleSwarmOptimizer(f, [ (0., 2.), (0., 2.) ], size=5)
    while p.fgbest > 5e-7:
        print p.particles
        print p.pbest
        print p.fpbest
        p.step()
        i = i + 1
        print '-'*50
    print i, p.gbest, p.fgbest