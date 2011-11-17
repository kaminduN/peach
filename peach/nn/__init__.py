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
    mem
      Associative memories and Hopfield model;
    kmeans
      K-Means implementation for use with Radial Basis Networks;
    rbfn
      Radial Basis Function Networks;
"""


# __all__ = [ 'base', 'af', 'lrules', 'nnet', 'mem', 'kmeans', 'rbfn' ]

################################################################################
# Imports sub-packages
from peach.nn.base import *
from peach.nn.af import *
from peach.nn.lrules import *
from peach.nn.nnet import *
from peach.nn.mem import *
from peach.nn.kmeans import *
from peach.nn.rbfn import *
