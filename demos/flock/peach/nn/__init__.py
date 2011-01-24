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

    base
      Basic definitions of the objects used with neural networks;
    af
      A list of activation functions for use with neurons and a base class to
      implement different activation functions;
    lrule
      Learning rules;
    nnet
      Implementation of different classes of neural networks;
"""


# __all__ = [ 'base', 'af', 'lrules', 'nnet' ]

################################################################################
# Imports sub-packages
from base import *
from af import *
from lrules import *
from nnet import *
