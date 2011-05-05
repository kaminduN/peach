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

    base
        Implementation of the basic genetic algorithm;
    chromosome
        Basic definitions to work with chromosomes. Defined as arrays of bits;
    crossover
        Defines crossover operators and base classes;
    fitness
        Defines fitness functions and base classes;
    mutation
        Defines mutation operators and base classes;
    selection
        Defines selection operators and base classes;
"""


# __all__ = [ 'base', 'chromosome', 'crossover', 'fitness', 'mutation', 'selection' ]

################################################################################
# Imports sub-packages
from peach.ga.base import *
from peach.ga.chromosome import *
from peach.ga.crossover import *
from peach.ga.fitness import *
from peach.ga.mutation import *
from peach.ga.selection import *
