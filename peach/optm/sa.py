################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/sa.py
# Simulated Annealing
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements two versions of simulated annealing optimization. One
works with numeric data, and the other with a codified bit string. This last
method can be used in discrete optimization problems.
"""

################################################################################
from numpy import exp, abs, sum, reshape, array, isnan
from numpy.linalg import inv
from numpy.random import standard_normal, uniform
from random import randrange
from bitarray import bitarray
import struct
import types
from optm import gradient, Optimizer


################################################################################
# Classes
################################################################################
class ContinuousSA(Optimizer):
    '''
    Simulated Annealing continuous optimization.

    This is a simulated annealing optimizer implemented to work with vectors of
    continuous variables (obviouslly, implemented as floating point numbers). In
    general, simulated annealing methods searches for neighbors of one estimate,
    which makes a lot more sense in discrete problems. While in this class the
    method is implemented in a different way (to deal with continuous
    variables), the principle is pretty much the same -- the neighbor is found
    based on a gaussian neighborhood.

    A simulated annealing algorithm adapted to deal with continuous variables
    has an enhancement that can be used: a gradient vector can be given and, in
    case the neighbor is not accepted, the estimate is updated in the downhill
    direction.
    '''
    def __init__(self, f, df=None, T0=1000., rt=0.95, h=0.05, emax=1e-8, imax=1000):
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
          T0
            Initial temperature of the system. The temperature is, of course, an
            analogy. Defaults to 1000.
          rt
            Temperature decreasing rate. The temperature must slowly decrease in
            simulated annealing algorithms. In this implementation, this is
            controlled by this parameter. At each step, the temperature is
            multiplied by this value, so it is necessary that ``0 < rt < 1``.
            Defaults to 0.95, smaller values make the temperature decay faster,
            while larger values make the temperature decay slower.
          h
            Convergence step. In the case that the neighbor estimate is not
            accepted, a simple gradient step is executed. This parameter is the
            convergence step to the gradient step.
          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.
          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        self.__f = f
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__t = float(T0)
        self.__r = float(rt)
        self.__h = float(h)
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, a neighbor of the given estimate is chosen at random,
        using a gaussian neighborhood. It is accepted as a new estimate if it
        performs better in the cost function *or* if the temperature is high
        enough. In case it is not accepted, a gradient step is executed.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        f = self.__f
        xn = x + self.__h*standard_normal(x.shape)
        delta = f(xn) - f(x)
        if delta < 0 or exp(-delta/self.__t) > uniform():
            xr = xn
            er = abs(delta)
        else:
            er = self.__h * self.__df(x)
            xr = x - er
            er = sum(abs(er))
        self.__t = self.__t * self.__r
        return (xr, er)


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
class DiscreteSA(Optimizer):
    '''
    Simulated Annealing discrete optimization.

    This is a simulated annealing optimizer implemented to work with vectors of
    discrete variables, which can be floating point or integer numbers,
    characters or anything allowed by the ``struct`` module of the Python
    standard library. The neighborhood of an estimate is calculated by inverting
    a number of bits randomly according to a given rate. Given the nature of
    this implementation, no alternate convergence can be used in the case of
    rejection of an estimate.
    '''
    def __init__(self, f, fmt, ranges=[ ], T0=1000., rt=0.95, nb=1, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.
          fmt
            A ``struct``-module string with the format of the data used. Please,
            consult the ``struct`` documentation, since what is explained there
            is exactly what is used here. For example, if you are going to use
            the optimizer to deal with three-dimensional vectors of continuous
            variables, the format would be something like::

              fmt = 'fff'

            Default value is an empty string. Notice that this is implemented as
            a ``bitarray``, so this module must be present.

            It is strongly recommended that integer numbers are used! Floating
            point numbers can be simulated with long integers. The reason for
            this is that random bit sequences can have no representation as
            floating point numbers, and that can make the algorithm not perform
            adequatelly.
          ranges
            Ranges of values allowed for each component of the input vector. If
            given, ranges are checked and a new estimate is generated in case
            any of the components fall beyond the value. ``range`` can be a
            tuple containing the inferior and superior limits of the interval;
            in that case, the same range is used for every variable in the input
            vector. ``range`` can also be a list of tuples of the same format,
            inferior and superior limits; in that case, the first tuple is
            assumed as the range allowed for the first variable, the second
            tuple is assumed as the range allowed for the second variable and so
            on.
          T0
            Initial temperature of the system. The temperature is, of course, an
            analogy. Defaults to 1000.
          rt
            Temperature decreasing rate. The temperature must slowly decrease in
            simulated annealing algorithms. In this implementation, this is
            controlled by this parameter. At each step, the temperature is
            multiplied by this value, so it is necessary that ``0 < rt < 1``.
            Defaults to 0.95, smaller values make the temperature decay faster,
            while larger values make the temperature decay slower.
          nb
            The number of bits to be randomly choosen to be inverted in the
            calculation of the neighbor. Be very careful while choosing this
            parameter. While very large optimizations can benefit from a big
            value here, it is not recommended that more than one bit per
            variable is inverted at each step -- otherwise, the neighbor might
            fall very far from the present estimate, which can make the
            algorithm not work accordingly. This defaults to 1, that is, at each
            step, only one bit will be inverted at most.
          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.
          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        self.__f = f
        self.format = fmt
        if not ranges:
            self.ranges = None
        elif len(ranges) == 1:
            self.ranges = array(ranges * len(fmt))
        else:
            self.ranges = array(ranges)
        self.__t = float(T0)
        self.__r = float(rt)
        self.__nb = int(nb)
        self.__xbest = None
        self.__fbest = None
        self.__emax = float(emax)
        self.__imax = int(imax)


    def step(self, x):
        '''
        One step of the search.

        In this method, a neighbor of the given estimate is obtained from the
        present estimate by choosing ``nb`` bits and inverting them. It is
        accepted as a new estimate if it performs better in the cost function
        *or* if the temperature is high enough. In case it is not accepted, the
        previous estimate is mantained.

        :Parameters:
          x
            The value from where the new estimate should be calculated. This can
            of course be the result of a previous iteration of the algorithm.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        f = self.__f
        fx = f(x)
        if self.__xbest is None or self.__fbest > fx:
            self.__xbest = x
            self.__fbest = f(x)
        bits = bitarray()
        bits.fromstring(struct.pack(self.format, *x))
        for i in xrange(self.__nb):
            index = randrange(len(bits))
            bits[index] = 1 - bits[index]
        xn = struct.unpack(self.format, bits.tostring())
        xn = reshape(xn, x.shape)

        # Performs a sanity check in the values.
        r = self.ranges
        if r is not None:
            x0 = r[:, 0]
            x1 = r[:, 1]
            if any(xn < x0) or any(xn > x1) or any(isnan(xn)):
                return (x, 1)

        delta = f(xn) - f(x)
        if delta < 0 or exp(-delta/self.__t) > uniform():
            xr = xn
        else:
            xr = x
        self.__t = self.__t * self.__r
        return (xr, abs(delta))


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
        return self.__xbest, e


################################################################################
# Test
if __name__ == "__main__":
    pass