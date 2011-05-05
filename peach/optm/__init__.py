################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/__init__.py
# Makes the optm directory a package and initializes it.
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements deterministic optimization methods. Consult:

    base
        Basic definitions and interface with the optimization methods;
    linear
        Basic methods for one variable optimization;
    multivar
        Gradient, Newton and othe multivariable optimization methods;
    quasinewton
        Quasi-Newton methods;

Every optimizer works in pretty much the same way. Instantiate the respective
class, using as parameter the cost function to be optimized, the first estimate
(a scalar in case of a single variable optimization, and a one-dimensional array
in case of multivariable optimization) and some other parameters. Use ``step()``
to perform one iteration of the method, use the ``__call__()`` method to perform
the search until the stop conditions are met. See each method for details.
"""


# __all__ = [ 'base', 'linear', 'multivar', 'quasinewton' ]

################################################################################
# Imports sub-packages
from peach.optm.base import *          # Basic definitions
from peach.optm.linear import *        # Linear and 1-D optimization
from peach.optm.multivar import *      # Gradient and Newton methods
from peach.optm.quasinewton import *   # Quasi-newton methods
