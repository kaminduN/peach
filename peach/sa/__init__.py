################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: sa/__init__.py
# Makes the sa directory a package and initializes it.
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements optimization by simulated annealing. Consult:

    base
        Implementation of the basic simulated annealing algorithms;
    neighbor
        Some methods for determining the neighbor of the present estimate;

Simulated Annealing is a meta-heuristic designed for optimization of functions.
It tries to mimic the way that atoms settle in crystal structures of metals. By
slowly cooling the metal, atoms settle in a position of low energy -- thus, it
is a natural optimization method.

Two kinds of optimizer are implemented here. The continuous version of the
algorithm can be used for optimization of continuous objective functions; the
discrete (or binary) one, can be used in combinatorial optimization problems.
"""


# __all__ = [ 'base', 'neighbor' ]

################################################################################
# Imports sub-packages
from peach.sa.base import *          # Basic definitions
from peach.sa.neighbor import *      # Computation of the neighbor
