################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/quasinewton.py
# Quasi-newton multivariable search methods
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements basic quasi-Newton optimizers. Newton optimizer is very
efficient, except that inverse matrices need to be calculated at each
convergence step. These methods try to estimate the hessian inverse iteratively,
thus increasing performance.
"""

################################################################################
import numpy
from numpy import dot, sum, abs, eye
from numpy.linalg import inv
from optm import Optimizer, gradient, hessian


################################################################################
# Classes
################################################################################
class DFP(Optimizer):
    '''
    DFP (*Davidon-Fletcher-Powell*) search
    '''
    def __init__(self, f, df=None, B=None, h=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.
          df
            A function to calculate the gradient vector of the cost function
            ``f``. Defaults to ``None``, if no gradient is supplied, then it is
            estimated from the cost function using Euler equations.
          B
            A first estimate of the inverse hessian. Note that, differently from
            the Newton method, the elements in this matrix are numbers, not
            functions. So, it is an estimate at a given point, and its values
            *should* be coherent with the first estimate (that is, ``B`` should
            be the inverse of the hessian evaluated at the first estimate), or
            else the algorithm might diverge. Defaults to ``None``, if none is
            given, it is estimated. Note that, given the same reasons as before,
            the estimate of ``B`` is deferred to the first calling of the
            ``step`` method, where it is handled automatically.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.
          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        Optimizer.__init__(self)
        self.__f = f
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__B = B
        self.__h = h
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is dependent of parameters
        calculated before (namely, the estimate of the inverse hessian), so it
        is not recomended that different investigations are used with the same
        optimizer in the same cost function.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        # Initializes the inverse hessian, if needed
        if self.__B is None:
            B = inv(hessian(self.__f)(x))
        else:
            B = self.__B

        # Updates x
        n = x.size
        x = x.reshape((n, 1))          # x as a line-vector
        dfx = self.__df(x).reshape((n, 1))
        dx = - self.__h * dot(B, dfx)
        xn = x + dx

        # Updates B
        y = self.__df(xn) - dfx
        Bty = dot(B.T, y)
        dB = dot(dx, dx.T) / dot(y.T, dx) \
             - dot(Bty, Bty.T) / dot(y.T, Bty)
        self.__B = B + dB

        return xn, sum(abs(xn - x))


    def __call__(self, x):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Parameters:
          x
            The initial triplet of values from where the search must start.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            x, e = self.step(x)
            i = i + 1
        return x, e


################################################################################
class BFGS(Optimizer):
    '''
    BFGS (*Broyden-Fletcher-Goldfarb-Shanno*) search
    '''
    def __init__(self, f, df=None, B=None, h=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.
          df
            A function to calculate the gradient vector of the cost function
            ``f``. Defaults to ``None``, if no gradient is supplied, then it is
            estimated from the cost function using Euler equations.
          B
            A first estimate of the inverse hessian. Note that, differently from
            the Newton method, the elements in this matrix are numbers, not
            functions. So, it is an estimate at a given point, and its values
            *should* be coherent with the first estimate (that is, ``B`` should
            be the inverse of the hessian evaluated at the first estimate), or
            else the algorithm might diverge. Defaults to ``None``, if none is
            given, it is estimated. Note that, given the same reasons as before,
            the estimate of ``B`` is deferred to the first calling of the
            ``step`` method, where it is handled automatically.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.
          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        Optimizer.__init__(self)
        self.__f = f
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__B = B
        self.__h = h
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is dependent of parameters
        calculated before (namely, the estimate of the inverse hessian), so it
        is not recomended that different investigations are used with the same
        optimizer in the same cost function.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        # Initializes the inverse hessian, if needed
        if self.__B is None:
            B = inv(hessian(self.__f)(x))
        else:
            B = self.__B

        # Updates x
        n = x.size
        x = x.reshape((n, 1))          # x as a line-vector
        dfx = self.__df(x).reshape((n, 1))
        dx = - self.__h * dot(B, dfx)
        xn = x + dx

        # Updates B
        dxt = dx.transpose()
        y = self.__df(xn) - dfx
        ytx = dot(y.T, dx)
        M = eye(n) - dot(y, dx.T) / ytx
        self.__B = dot(dot(M.T, B), M) + dot(dx, dx.T) / ytx

        return xn, sum(abs(xn - x))


    def __call__(self, x):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Parameters:
          x
            The initial triplet of values from where the search must start.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            x, e = self.step(x)
            i = i + 1
        return x, e


################################################################################
class SR1(Optimizer):
    '''
    SR1 (*Symmetric Rank 1* )search method
    '''
    def __init__(self, f, df=None, B=None, h=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.
          df
            A function to calculate the gradient vector of the cost function
            ``f``. Defaults to ``None``, if no gradient is supplied, then it is
            estimated from the cost function using Euler equations.
          B
            A first estimate of the inverse hessian. Note that, differently from
            the Newton method, the elements in this matrix are numbers, not
            functions. So, it is an estimate at a given point, and its values
            *should* be coherent with the first estimate (that is, ``B`` should
            be the inverse of the hessian evaluated at the first estimate), or
            else the algorithm might diverge. Defaults to ``None``, if none is
            given, it is estimated. Note that, given the same reasons as before,
            the estimate of ``B`` is deferred to the first calling of the
            ``step`` method, where it is handled automatically.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.
          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        Optimizer.__init__(self)
        self.__f = f
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__B = B
        self.__h = h
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is dependent of parameters
        calculated before (namely, the estimate of the inverse hessian), so it
        is not recomended that different investigations are used with the same
        optimizer in the same cost function.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        # Initializes the inverse hessian, if needed
        if self.__B is None:
            B = inv(hessian(self.__f)(x))
        else:
            B = self.__B

        # Updates x
        n = x.size
        x = x.reshape((n, 1))          # x as a line-vector
        dfx = self.__df(x).reshape((n, 1))
        dx = - self.__h * dot(B, dfx)
        xn = x + dx

        # Updates B
        y = self.__df(xn) - dfx
        M = dx - dot(B, y)
        dB = dot(M, M.T) / dot(M.T, y)
        self.__B = B + dB

        return xn, sum(abs(xn - x))


    def __call__(self, x):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Parameters:
          x
            The initial triplet of values from where the search must start.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            x, e = self.step(x)
            i = i + 1
        return x, e


################################################################################
# Test
if __name__ == "__main__":
    pass