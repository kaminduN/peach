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

    optm
        Basic definitions and interface with the optimization methods;
    linear
        Basic methods for one variable optimization;
    multivar
        Gradient, Newton and othe multivariable optimization methods;
    quasinewton
        Quasi-Newton methods;
    stochastic
        General stochastic methods;
    sa
        Simulated Annealing methods;
    pso
        Particle Swarm Optimization

Every optimizer works in pretty much the same way. Instantiate the respective
class, using as parameter the cost function to be optimized and some other
parameters. Use ``step()`` to perform one iteration of the method, use the
``__call__()`` method to perform the search until the stop conditions are met.
See each method for details.
"""


# __all__ = [ 'linear', 'multivar', 'quasinewton', 'stochastic', 'sa', 'pso' ]

################################################################################
# Imports sub-packages
from linear import *        # Linear and 1-D optimization
from multivar import *      # Gradient and Newton methods
from quasinewton import *   # Quasi-newton methods
from stochastic import *    # Stochastic methods
from sa import *            # Simulated annealing
from pso import *           # Particle Swarm Optimizer
