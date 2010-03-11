################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: ga/__init__.py
# Makes the ga directory a python package and initializes it.
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements genetic algorithms. Consult:

    chromosome
        Basic definitions to work with chromosomes. Defined as arrays of bits;
    crossover
        Defines crossover operators and base classes;
    fitness
        Defines fitness functions and base classes;
    base
        Implementation of the basic genetic algorithm;
    mutation
        Defines mutation operators and base classes;
    selection
        Defines selection operators and base classes;
"""


# __all__ = [ 'chromosome', 'crossover', 'fitness', 'base', 'mutation', 'selection' ]

################################################################################
# Imports sub-packages
from chromosome import *
from crossover import *
from fitness import *
from base import *
from mutation import *
from selection import *
