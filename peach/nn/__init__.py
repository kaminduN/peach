################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/__init__.py
# Makes the nn directory a python package and initializes it.
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements support for neural networks. Consult:

    af
      A list of activation functions for use with neurons and a base class to
      implement different activation functions;
    base
      Basic definitions of the objects used with neural networks;
    lrule
      Learning rules;
    nn
      Implementation of different classes of neural networks;
"""


# __all__ = [ 'af', 'base', 'lrules', 'nn' ]

################################################################################
# Imports sub-packages
from af import *
from base import *
from lrules import *
from nn import *
