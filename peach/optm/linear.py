################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/linear.py
# 1D search methods
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements basic one variable only optimizers.
"""


################################################################################
from numpy import abs, max, isnan, sqrt
from optm import Optimizer


################################################################################
# Classes
################################################################################
class Direct1D(Optimizer):
    '''
    1-D direct search.

    This methods 'oscilates' around the function minimum, reducing the updating
    step until it achieves the maximum error or the maximum number of steps.
    This is a very inefficient method, and should be used only at times where no
    other methods are able to converge (eg., if a function has a lot of
    discontinuities).
    '''
    def __init__(self, f, dx=0.5, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
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
        self.__dx = dx
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is highly dependent of the steps
        executed before, as the search step is updated at each call to this
        method.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        f = self.__f
        fo = f(x)
        x = x + self.__dx
        fn = f(x)
        if fn > fo:
            self.__dx = - self.__dx / 2.
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
        while abs(self.__dx) > emax/2. and i < imax:
            x, e = self.step(x)
            i = i + 1
        return x, e


################################################################################
class Interpolation(Optimizer):
    '''
    Optimization by quadractic interpolation.

    This methods takes three estimates and finds the parabolic function that
    fits them, and returns as a new estimate the vertex of the parabola. The
    procedure can be repeated until a good approximation is found.
    '''
    def __init__(self, f, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
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
            A triple ``(x0, x1, x2)``, with ``x0 < x1 < x2`` of estimates on the
            cost function. From these values the new estimate is calculated.
            This can of course be the result of a previous iteration of the
            algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          triplet of estimates of the minimum, and ``e`` is the estimated error.
        '''
        x0, x1, x2 = x
        f = self.__f
        q0 = x0 * (f(x1) - f(x2))
        q1 = x1 * (f(x2) - f(x0))
        q2 = x2 * (f(x0) - f(x1))
        xm = 0.5 * (x0*q0 + x1*q1 + x2*q2) / (q0 + q1 + q2)
        if isnan(xm):
            return (x0, x1, x2), max(abs(x1-x0), abs(x2-x1))
        if xm < x0:
            x = (xm, x0, x1)
        elif x0 < xm < x1:
            x = (x0, xm, x1)
        elif x1 < xm < x2:
            x = (x1, xm, x2)
        elif x2 < xm:
            x = (x1, x2,  xm)
        else:
            x = (xm, xm, xm)
        return x, max(abs(xm-x0), abs(x2-xm))


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
            x0, x1, x2 = x
            if x0 == x1:
                return x0, e
            elif x1 == x2:
                return x1, e
            elif x0 == x2:
                return x2, e
            x, e = self.step(x)
            i = i + 1
        f = self.__f
        q0 = x0 * (f(x1) - f(x2))
        q1 = x1 * (f(x2) - f(x0))
        q2 = x2 * (f(x0) - f(x1))
        xm = 0.5 * (x0*q0 + x1*q1 + x2*q2) / (q0 + q1 + q2)
        return xm, e


################################################################################
class GoldenRule(Optimizer):
    '''
    Optimizer by the Golden Section Rule

    This optimizer uses the golden rule to section an interval in search of the
    minimum. Using a simple heuristic, the interval is refined until an interval
    small enough to satisfy the error requirements is found.
    '''
    def __init__(self, f, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
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
        self.__k = (sqrt(5) - 1.) / 2.
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
            A duple ``(x0, x1)``, with ``x0 < x1`` of estimates on the cost
            function. From these values the new estimate is calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          duple of estimates of the minimum, and ``e`` is the estimated error.
        '''
        x0, x1 = x
        f = self.__f
        k = self.__k
        k1 = 1 - k
        xl = k*x0 + k1*x1
        xh = k1*x0 + k*x1
        if f(xl) > f(xh):
            return (xl, x1), abs(x1 - xl)
        else:
            return (x0, xh), abs(xh - x0)


    def __call__(self, x):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Parameters:
          x
            The initial duple of values from where the search must start.

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
        xl, xh = x
        return 0.5 * (xl + xh), e


################################################################################
class Fibonacci(Optimizer):
    '''
    Optimization by the Golden Rule Section, estimated by Fibonacci numbers.

    This optimizer uses the golden rule to section an interval in search of the
    minimum. Using a simple heuristic, the interval is refined until an interval
    small enough to satisfy the error requirements is found. The golden section
    is estimated at each step using Fibonacci numbers. This can be useful in
    situations where only integer numbers should be used.
    '''
    def __init__(self, f, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
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
        self.__k1 = 1.
        self.__k2 = 1.
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, the result of the step is highly dependent of the steps
        executed before, as the estimate of the golden ratio is updated at each
        call to this method.

        :Parameters:
          x
            A duple ``(x0, x1)``, with ``x0 < x1`` of estimates on the cost
            function. From these values the new estimate is calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          duple of estimates of the minimum, and ``e`` is the estimated error.
        '''
        x0, x1 = x
        f = self.__f
        k1 = self.__k1
        k2 = self.__k2
        self.__k1 = k2
        self.__k2 = k1 + k2
        k = k1 / k2
        k1 = 1 - k
        xl = k*x0 + k1*x1
        xh = k1*x0 + k*x1
        if f(xl) > f(xh):
            return (xl, x1), abs(x1 - xl)
        else:
            return (x0, xh), abs(xh - x0)


    def __call__(self, x):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Parameters:
          x
            The initial duple of values from where the search must start.

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
        xl, xh = x
        return 0.5* (xl + xh), e


################################################################################
# Test
if __name__ == "__main__":

    # Rosenbrock function
    def f(x):
        return (1-x)**2 + (1-x*x)**2

    linear = Linear1D(f, emax=1e-12)
    print linear(3.21345)
    interp = Interpolation(f, emax=1e-12)
    print interp((0., 0.75, 1.5))
    golden = GoldenRule(f, emax=1e-12)
    print golden((0.75, 1.4))
    fibo = Fibonacci(f, emax=1e-12)
    print fibo((0.75, 1.4))
