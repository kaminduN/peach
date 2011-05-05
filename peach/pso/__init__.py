################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: pso/__init__.py
# Makes the pso directory a package and initializes it.
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic Particle Swarm Optimization (PSO)

This sub-package implements traditional particle swarm optimizers as described
in literature. It consists of a very simple algorithm emulating the behaviour
of a flock of birds (though in a very simplified way). A population of particles
is created, each particle with its corresponding velocity. They fly towards the
particle local best and the swarm global best, thus exploring the whole domain.

For consistency purposes, the particles are represented internally as a list of
vectors. The particles can be acessed externally by using the ``[ ]`` interface.
See the rest of the documentation for more information.
"""


# __all__ = [ 'base', 'acc' ]

################################################################################
# Imports sub-packages
from peach.pso.base import *          # Basic definitions
from peach.pso.acc import *           # Acceleration of particles
