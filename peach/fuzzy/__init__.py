################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/__init__.py
# Makes the fuzzy directory a python package and initializes it.
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements fuzzy logic. Consult:

    base
        Basic definitions, classes and operations in fuzzy logic;
    mf
        Membership functions;
    defuzzy
        Defuzzification methods;
    control
        Fuzzy controllers (FIS - Fuzzy Inference Systems), for Mamdani- and
        Sugeno-type controllers and others;
    cmeans
        Fuzzy C-Means clustering algorithm;
"""


# __all__ = [ 'base', 'control', 'mf', 'defuzzy', 'cmeans' ]

################################################################################
# Imports sub-packages
from peach.fuzzy.base import *
from peach.fuzzy.control import *
from peach.fuzzy.mf import *
from peach.fuzzy.defuzzy import *
from peach.fuzzy.cmeans import *
