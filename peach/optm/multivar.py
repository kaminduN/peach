################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/multivar.py
# Gradient and multivariable search methods
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements basic multivariable optimizers, including gradient and
Newton searches.
"""

################################################################################
import numpy
from numpy import dot, abs, sum, ravel
from numpy.linalg import inv
from optm import Optimizer, gradient, hessian


################################################################################
# Classes
################################################################################
class Direct(Optimizer):
    '''
    Multidimensional direct search

    This optimization method is a generalization of the 1D method, using
    variable swap as search direction. This results in a very simplistic and
    inefficient method that should be used only when any other method fails.
    '''
    def __init__(self, f, h=0.5, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.
          dx
            The initial step of the search. Defaults to 0.5
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
        self.__dx = None
        self.__h = h
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is highly dependent of the steps
        executed before, as the search step is updated at each call to this
        method.

        One characteristic of this method is that it uses the dimensions of the
        input vector to initialize the updating matrix -- this is necessary to
        mantain coherence with the interface of other methods. The matrix
        dimensions are calculated and used in the first call to this method and
        used in future calls. In general, there is no need to worry about this,
        since everything is taken care of automatically. But, if future calls to
        this method are done with incoherent dimensions, then an exception will
        be raised.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        x = x.ravel()           # Makes x a line-vector.
        if self.__dx is None:
            n = x.size
            self.__dx = self.__h * numpy.eye(n, 1).reshape(x.shape)
            h = numpy.eye(n, n, -1)
            h[0, n-1] = - 0.5
            self.__h = h
        f = self.__f
        fo = f(x)
        x = x + self.__dx
        fn = f(x)
        if sum(fn-fo) > 0:
            print self.__dx
            self.__dx = dot(self.__h, self.__dx)
        return x, abs(self.__dx)


    def __call__(self, x):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Parameters:
          x
            The value from where the search must start.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        i = 0
        while sum(abs(self.__dx)) > emax/2. and i < imax:
            x, e = self.step(x)
            i = i + 1
        return x, e


################################################################################
class Gradient(Optimizer):
    '''
    Gradient search

    This method uses the fact that the gradient of a function points to the
    direction of largest increase in the function (in general called *uphill*
    direction). So, the contrary direction (*downhill*) is used as search
    direction.
    '''
    def __init__(self, f, df=None, h=0.1, emax=1e-5, imax=1000):
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
        self.__h = h
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is dependent only of the given
        estimated, so it can be used for different kind of investigations on the
        same cost function.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        x = ravel(x)           # Makes x a line-vector
        xold = x
        dfx = self.__df(x)
        x = x - self.__h * dfx
        return x, sum(abs(x - xold))


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
class Newton(Optimizer):
    '''
    Newton search

    This is a very effective method to find minimum points in functions. In a
    very basic fashion, this method corresponds to using Newton root finding
    method on f'(x). Converges *very* fast if the cost function is quadratic
    of simmilar to it.
    '''
    def __init__(self, f, df=None, hf=None, h=0.1, emax=1e-5, imax=1000):
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
          hf
            A function to calculate the hessian matrix of the cost function
            ``f``. Defaults to ``None``, if no hessian is supplied, then it is
            estimated from the cost function using Euler equations.
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
        if hf is None:
            self.__hf = hessian(f)
        else:
            self.__hf = hf
        self.__h = h
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is dependent only of the given
        estimated, so it can be used for different kind of investigations on the
        same cost function.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        x = ravel(x)           # Makes x a line-vector
        xold = x
        df = self.__df(x)
        hf = self.__hf(x)
        try:
            x = x - self.__h * dot(inv(hf), df)
        except IndexError:
            x = x - self.__h * df / hf
        return x, sum(abs(x - xold))


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