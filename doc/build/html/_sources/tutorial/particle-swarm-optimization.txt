Optimization by Particle Swarms
===============================

Among the stochastic methods available for multivariate optimization, the
Particle Swarm is in general considered one of the most effective in continuous
optimization. It tries to explore the whole search domain by emulating (in a
very simplistic manner) the behaviour of birds in a flock. It is a populations
based method (such as genetic algorithms), where each particle communicate with
other particles to find the direction of the search.

Each particle has a position and a velocity, both multidimensional vectors in
the search space. They are capable of remembering the best point where each of
them wandered, the local best; and the swarm as a whole is capable of
remembering the best overall position found by any particle, the global best. By
communicatint the global best, in what is called the social behaviour of the
flock, particles are able to modify their velocities (ie, accelerate) in
different directions, thus finding a minimum.

In this tutorial, we will use particle swarm optimization to find -- as in many
of the other tutorials -- the minimum of the Rosenbrock function. There is
absolutelly no difference in how a simulated annealing optimizer is created and
used. We start by importing ``peach`` and ``numpy`` in different namespaces::

    from numpy import *
    from numpy.random import random
    import peach as p

We must also create the objective function and the gradient function, as the
algorithms use them. Notice however that, as before, we can omit the gradient
function and let Peach estimate them for us if needed::

    def f(xy):
        x, y = xy
        return (1.-x)**2. + (y-x*x)**2.

We need to create a population of estimates. In algorithms based on populations,
such as this or Genetic Algorithms, a list of estimates should be created. To
this end, we will specify the ranges of the variables in the interval from 0 to
2, and randomly choose from this. Ranges are specified as a list of tuples,
where each element is the allowed range for the corresponding variable. In each
tuple, the first value is the lower limit of the intervalo, and the second value
is its upper limit::

    ranges = [ ( 0., 2. ), ( 0., 2. ) ]

We use the ``numpy.random`` function to generate a population of five particles,
randomly positioned in a circle around the point ``(1., 1.)``, with radius 1. We
will do this to better observe the behaviour of the algorithm. In general,
swarms should have more than five particles, but this is enough for this case.
This line will be more effective if we convert ranges to a numpy array::

    ranges = array(ranges)
    theta = random((5, 1)) * 2. * pi
    x0 = c_[ 1. + cos(theta), 1. + sin(theta) ]

Now we will create the optimizer. We create these optimizers in the same way we
created other optimizers: by instantiating the corresponding class, passing the
function and the first estimate. Notice that the first estimates are given in
the form of a list of tuples, with the first estimate of :math:`x` in the first
place of each tuple, and the first estimate of :math:`y` in the second place.
There is no need to use tuples: lists or arrays will do. To create the
optimizers, we issue::

    pso = p.ParticleSwarmOptimizer(f, x0, ranges)

Notice that we included the ``ranges`` parameter in the creation of the
optimizer. This is not needed -- if ``ranges`` are not included, they will be
automatically computed from the estimates. But notice that, if you randomly
create your population, the ranges for each variable might not reflect the
entire search space.

As we done in the other optimization tutorials, we will execute the algorithm
step by step. We can do this to keep track of the estimates to plot a graphic.
We do this using the commands::

    xs = [ ]
    xd = [ ]
    i = 0
    while i < 500:
        x, e = pso.step()
        xs.append(x)
        xd.append(pso[:])
        i = i + 1

Notice that we used 500 iterations here. In general, stochastic methods pay this
price to be able to find the global minimum: they need more iterations to
converge. That's not a problem, however, since finding the global minimum is
a desired result, and the penalty in the convergence time is not that
significant. However, particle swarms are usually very effective in multivariate
continuous optimization -- and in this specific problem, we do not expect more
that 100 or even 50 iterations to take place.

The ``xs`` variable will hold, in sequence, the estimates. We can plot them to
see the convergence trace. Also, we keep track of every other estimate in the
``xd`` variable. We do this to track the convergence for every other particle in
the swarm. The figure below is a representation of the execution of the method.
The function itself is represented as contour curves in the plane, and the
estimate tracks over them. In blue, we have the track of the best estimate at
each iteration; in gray, the other estimates..

.. image:: figs/particle-swarm-optimization.png
   :align: center
