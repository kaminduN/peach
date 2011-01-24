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
from numpy import abs, max
from base import Optimizer


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
    discontinuities, or similar conditions).
    '''
    def __init__(self, f, x0, range=None, h=0.5, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
          x0
            First estimate of the minimum. Since this is a linear method, this
            should be a ``float`` or ``int``.
          range
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a tuples of two
            values, ``(x0, x1)``, where ``x0`` is the start of the interval, and
            ``x1`` its end. Obviously, ``x0`` should be smaller than ``x1``.
            When this parameter is present, the algorithm will not let the
            estimates fall outside the given interval.
          h
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
        self.__x = x0
        self.range = range
        '''Holds the range for the estimates. If this attribute is set, the
        algorithm will never let the estimates fall outside the given
        interval.'''
        self.__h = h
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
            The new initial value of the estimate of the minimum. Since this is
            a linear method, this should be a ``float`` or ``int``.
          h
            The initial step of the search. Defaults to 0.5
        '''
        self.__x = x0
        if h is not None:
            self.__h = h


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is highly dependent of the steps
        executed before, as the search step is updated at each call to this
        method.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        f = self.__f
        x = self.__x
        fo = f(x)

        # Computes next estimate
        x = x + self.__h

         # Sanity check
        if self.range is not None:
            r0, r1 = self.range
            if x < r0: x = r0
            if x > r1: x = r1

        # Update state
        fn = f(x)
        if fn > fo:
            self.__h = - self.__h / 2.
        self.__x = x
        return x, abs(self.__h)


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
        i = 0
        while abs(self.__h) > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
class Interpolation(Optimizer):
    '''
    Optimization by quadractic interpolation.

    This methods takes three estimates and finds the parabolic function that
    fits them, and returns as a new estimate the vertex of the parabola. The
    procedure can be repeated until a good approximation is found.
    '''
    def __init__(self, f, x0, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
          x0
            First estimate of the minimum. The interpolation search needs three
            estimates to approximate the parabolic function. Thus, the first
            estimate must be a triple ``(xl, xm, xh)``, with the property that
            ``xl < xm < xh``. Be aware, however, that no checking is done -- if
            the estimate doesn't correspond to this condition, in some point an
            exception will be raised.

            Notice that, given the nature of the estimate of the interpolation
            method, it is not necessary to have a specific parameter to restrict
            the range of acceptable values -- it is already embedded in the
            estimate. If you need to restrict your estimate between an interval,
            just use its limits as ``xl`` and ``xh`` in the estimate.
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
        self.__x = x0
        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            The new initial value of the estimate of the minimum. The
            interpolation search needs three estimates to approximate the
            parabolic function. Thus, the estimate must be a triple
            ``(xl, xm, xh)``, with the property that ``xl < xm < xh``. Be aware,
            however, that no checking is done -- if the estimate doesn't
            correspond to this condition, in some point an exception will be
            raised.
        '''
        self.__x = x0


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent only of the given
        estimated, so it can be used for different kind of investigations on the
        same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          triplet of estimates of the minimum, and ``e`` is the estimated error.
        '''
        x0, x1, x2 = self.__x
        f = self.__f
        q0 = x0 * (f(x1) - f(x2))
        q1 = x1 * (f(x2) - f(x0))
        q2 = x2 * (f(x0) - f(x1))
        q = q0 + q1 + q2
        if q == 0:
            return (x0, x1, x2), max(abs(x1-x0), abs(x2-x1))
        xm = 0.5 * (x0*q0 + x1*q1 + x2*q2) / (q0 + q1 + q2)
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
        self.__x = x
        return x, max(abs(xm-x0), abs(x2-xm))


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
            x0, x1, x2 = self.__x
            _, e = self.step()
            if x0 == x1:
                return x0, e
            elif x1 == x2:
                return x1, e
            elif x0 == x2:
                return x2, e
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
    def __init__(self, f, x0, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
          x0
            First estimate of the minimum. The golden rule search needs two
            estimates to partition the interval. Thus, the first estimate must
            be a duple ``(xl, xh)``, with the property that ``xl < xh``. Be
            aware, however, that no checking is done -- if the estimate doesn't
            correspond to this condition, in some point an exception will be
            raised.

            Notice that, given the nature of the estimate of the golden rule
            method, it is not necessary to have a specific parameter to restrict
            the range of acceptable values -- it is already embedded in the
            estimate. If you need to restrict your estimate between an interval,
            just use its limits as ``xl`` and ``xh`` in the estimate.
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
        self.__x = x0
        self.__k = 0.6180339887498949       # Golden ratio
        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            The new value of the estimate of the minimum. The golden rule search
            needs two estimates to partition the interval. Thus, the estimate
            must be a duple ``(xl, xh)``, with the property that ``xl < xh``.
        '''
        self.__x = x0


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent only of the given
        estimated, so it can be used for different kind of investigations on the
        same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          duple of estimates of the minimum, and ``e`` is the estimated error.
        '''
        x0, x1 = self.__x
        f = self.__f
        k = self.__k
        k1 = 1 - k
        xl = k*x0 + k1*x1
        xh = k1*x0 + k*x1
        if f(xl) > f(xh):
            x = (xl, x1)
            e = abs(x1 - xl)
        else:
            x = (x0, xh)
            e = abs(xh - x0)
        self.__x = x
        return x, e


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
        xl, xh = self.__x
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
    def __init__(self, f, x0, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A one variable only function to be optimized. The function should
            have only one parameter and return the function value.
          x0
            First estimate of the minimum. The Fibonacci search needs two
            estimates to partition the interval. Thus, the first estimate must
            be a duple ``(xl, xh)``, with the property that ``xl < xh``. Be
            aware, however, that no checking is done -- if the estimate doesn't
            correspond to this condition, in some point an exception will be
            raised.

            Notice that, given the nature of the estimate of the Fibonacci
            method, it is not necessary to have a specific parameter to restrict
            the range of acceptable values -- it is already embedded in the
            estimate. If you need to restrict your estimate between an interval,
            just use its limits as ``xl`` and ``xh`` in the estimate.
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
        self.__x = x0
        self.__k1 = 1.
        self.__k2 = 1.
        self.__emax = float(emax)
        self.__imax = int(imax)


    def restart(self, x0):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            The new value of the estimate of the minimum. The Fibonacci search
            needs two estimates to partition the interval. Thus, the estimate
            must be a duple ``(xl, xh)``, with the property that ``xl < xh``. Be
            aware, however, that no checking is done -- if the estimate doesn't
            correspond to this condition, in some point an exception will be
            raised.
        '''
        self.__x = x0


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is highly dependent of the steps
        executed before, as the estimate of the golden ratio is updated at each
        call to this method.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          duple of estimates of the minimum, and ``e`` is the estimated error.
        '''
        x0, x1 = self.__x
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
            x = (xl, x1)
            e = abs(x1 - xl)
        else:
            x = (x0, xh)
            e = abs(xh - x0)
        self.__x = x
        return x, e


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
        xl, xh = self.__x
        return 0.5* (xl + xh), e


################################################################################
# Test
if __name__ == "__main__":

    # Rosenbrock function
    def f(x):
        return (1.-x)**2. + (1.-x*x)**2.

    linear = Direct1D(f, 3.21345, emax=1e-12)
    print linear()
    interp = Interpolation(f, (0., 0.75, 1.5), emax=1e-12)
    print interp()
    golden = GoldenRule(f, (0.75, 1.4), emax=1e-12)
    print golden()
    fibo = Fibonacci(f, (0.75, 1.4), emax=1e-12)
    print fibo()
