################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/sa.py
# Simulated Annealing
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
General methods of stochastic optimization.
"""

################################################################################
# from numpy import 
from optm import Optimizer


################################################################################
# Classes
################################################################################
class CrossEntropy(Optimizer):
    '''
    Multidimensional search based on cross-entropy technique.

    In cross-entropy, a set of N possible solutions is randomly generated at
    each interaction. To converge the solutions, the best M solutions are
    selected and its statistics are calculated. A new set of solutions are
    randomly generated from these statistics.
    '''
    def __init__(self, f, M=30, N=60, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.
          M
            Size of the solution set used to calculate the statistics to
            generate the next set of solutions
          N
            Total size of the solution set.
          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.
          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        self.__f = f
        self.__M = int(M)
        self.__N = int(N)
        self.__emax = float(emax)
        self.__imax = int(imax)
        self.__solutions = [ ]


    def step(self):
        '''
        One step of the search (*NOT IMPLEMENTED YET*)

        In this method, the solution set is searched for the M best solutions.
        Mean and variance of these solutions is calculated, and these values are
        used to randomly generate, from a gaussian distribution, a set of N new
        solutions.
        '''
        pass