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
from numpy import dot, sum, abs, eye, array, outer
from numpy.linalg import inv
from base import Optimizer, gradient, hessian


################################################################################
# Classes
################################################################################
class DFP(Optimizer):
    '''
    DFP (*Davidon-Fletcher-Powell*) search
    '''
    def __init__(self, f, x0, ranges=None, df=None, h=0.1, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.

          x0
            First estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a list of ranges
            for each variable of the objective function. It is specified as a
            list of tuples of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. It can also be given as a list with a simple
            tuple in the same format. In that case, the same range will be
            applied for every variable in the optimization.

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
        self.__x = array(x0).ravel()
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__B = inv(hessian(f)(self.__x))
        self.__h = h

        # Determine ranges of the variables
        if ranges is not None:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0, h=None):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
        '''
        self.__x = array(x0).ravel()
        self.__B = inv(hessian(self.__f)(self.__x))
        if h is not None:
            self.__h = h


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent of parameters
        calculated before (namely, the estimate of the inverse hessian), so it
        is not recomended that different investigations are used with the same
        optimizer in the same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        # Updates x
        x = self.__x
        B = self.__B
        dfx = self.__df(x)
        dx = - self.__h * dot(B, dfx)
        xn = x + dx

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            xn = where(xn < r0, r0, xn)
            xn = where(xn > r1, r1, xn)

        # Updates B
        y = self.__df(xn) - dfx
        By = dot(B, y)
        dB = outer(dx, dx) / dot(y, dx) - outer(By, By) / dot(y, By)
        self.__B = B + dB

        # Updates state
        self.__x = xn
        return xn, sum(abs(xn - x))


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
class BFGS(Optimizer):
    '''
    BFGS (*Broyden-Fletcher-Goldfarb-Shanno*) search
    '''
    def __init__(self, f, x0, ranges=None, df=None, h=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.

          x0
            First estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a list of ranges
            for each variable of the objective function. It is specified as a
            list of tuples of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. It can also be given as a list with a simple
            tuple in the same format. In that case, the same range will be
            applied for every variable in the optimization.

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
        self.__x = array(x0).ravel()
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__B = inv(hessian(self.__f)(self.__x))
        self.__h = h

        # Determine ranges of the variables
        if ranges is not None:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        self.__emax = float(emax)
        self.__imax = int(imax)


    def restart(self, x0, h=None):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
        '''
        self.__x = array(x0).ravel()
        self.__B = inv(hessian(self.__f)(self.__x))
        if h is not None:
            self.__h = h


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent of parameters
        calculated before (namely, the estimate of the inverse hessian), so it
        is not recomended that different investigations are used with the same
        optimizer in the same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        # Updates x
        x = self.__x
        n = x.size
        B = self.__B
        dfx = self.__df(x)
        dx = - self.__h * dot(B, dfx)
        xn = x + dx

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            xn = where(xn < r0, r0, xn)
            xn = where(xn > r1, r1, xn)

        # Updates B
        y = self.__df(xn) - dfx
        ytx = dot(y.T, dx)
        M = eye(n) - outer(y, dx.T) / ytx
        self.__B = dot(dot(M.T, B), M) + outer(dx, dx) / ytx

        # Updates state
        self.__x = xn
        return xn, sum(abs(xn - x))


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
class SR1(Optimizer):
    '''
    SR1 (*Symmetric Rank 1* ) search method
    '''
    def __init__(self, f, x0, ranges=None, df=None, h=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.

          x0
            First estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a list of ranges
            for each variable of the objective function. It is specified as a
            list of tuples of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. It can also be given as a list with a simple
            tuple in the same format. In that case, the same range will be
            applied for every variable in the optimization.

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
        self.__x = array(x0).ravel()
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__B = inv(hessian(self.__f)(self.x))
        self.__h = h

        # Determine ranges of the variables
        if ranges is not None:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0, h=None):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
        '''
        self.__x = array(x0).ravel()
        self.__B = inv(hessian(self.__f)(self.x))
        if h is not None:
            self.__h = h


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent of parameters
        calculated before (namely, the estimate of the inverse hessian), so it
        is not recomended that different investigations are used with the same
        optimizer in the same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        # Updates x
        x = self.__x
        B = self.__B
        dfx = self.__df(x)
        dx = - self.__h * dot(B, dfx)
        xn = x + dx

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            xn = where(xn < r0, r0, xn)
            xn = where(xn > r1, r1, xn)

        # Updates B
        y = self.__df(xn) - dfx
        M = dx - dot(B, y)
        dB = outer(M, M) / dot(M, y)
        self.__B = B + dB

        self.__x = xn
        return xn, sum(abs(xn - x))


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
# Test
if __name__ == "__main__":

    # Rosenbrock function
    def f(xy):
        x, y = xy
        return (1.-x)**2. + (y-x*x)**2.

    # Gradient of Rosenbrock function
    def df(xy):
        x, y = xy
        return array( [ -2.*(1.-x) - 4.*x*(y-x*x), 2.*(y-x*x) ])

    dfp = DFP(f, (0., 0.), emax=1e-12)
    print dfp()
    bfgs = BFGS(f, (0., 0.), emax=1e-12)
    print bfgs()
    sr1 = SR1(f, (0., 0.), emax=1e-12)
    print sr1()

