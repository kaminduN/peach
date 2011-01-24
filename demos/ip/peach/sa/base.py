################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: sa/base.py
# Simulated Annealing
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements two versions of simulated annealing optimization. One
works with numeric data, and the other with a codified bit string. This last
method can be used in discrete optimization problems.
"""

################################################################################
from numpy import exp, abs, array, isnan, where
from numpy.random import uniform
from random import randrange
from bitarray import bitarray
import struct
import types

from neighbor import *


################################################################################
# Classes
################################################################################
class ContinuousSA(object):
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
    def __init__(self, f, x0, ranges=None, neighbor=GaussianNeighbor, optm=None,
                 T0=1000., rt=0.95, emax=1e-8, imax=1000):
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

          neighbor
            Neighbor function. This is a function used to compute the neighbor
            of the present estimate. You can use the ones defined in the
            ``neighbor`` module, or you can implement your own. In any case, the
            ``neighbor`` parameter must be an instance of ``ContinuousNeighbor``
            or of a subclass. Please, see the documentation on the ``neighbor``
            module for more information. The default is ``GaussianNeighbor``,
            which computes the new estimate based on a gaussian distribution
            around the present estimate.

          optm
            A standard optimizer such as gradient or Newton. This is used in
            case the estimate is not accepted by the algorithm -- in this case,
            a new estimate is computed in a standard way, providing a little
            improvement in any case. It defaults to None; in that case, no
            standard optimizatiion will be used. Notice that, if you want to use
            a standard optimizer, you must create it before you instantiate this
            class. By doing it this way, you can configure the optimizer in any
            way you want. Please, consult the documentation in ``Gradient``,
            ``Newton`` and others.

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
        self.__x = array(x0).ravel()
        self.__fx = f(self.__x)

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

        # Verifies the validity of the neighbor method
        try:
            issubclass(neighbor, ContinuousNeighbor)
            neighbor = neighbor()
        except TypeError:
            pass
        if isinstance(neighbor, types.FunctionType):
            neighbor = ContinuousNeighbor(neighbor)
        if not isinstance(neighbor, ContinuousNeighbor):
            raise TypeError, 'not a valid neighbor function'
        else:
            self.__nb = neighbor

        self.__optm = optm
        self.__t = float(T0)
        self.__r = float(rt)
        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def __get_fx(self):
        return self.__fx

    fx = property(__get_fx, None)
    '''The value of the objective function at the present estimate.'''


    def restart(self, x0, T0=1000., rt=0.95, h=0.5):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate. Restartings are essential to the working of
        simulated annealing algorithms, to allow them to leave local minima.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

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
            The initial step of the search. Defaults to 0.5
        '''
        self.__x = array(x0).ravel()
        self.__fx = self.__f(self.__x)
        self.__t = float(T0)
        self.__r = float(rt)
        self.__h = float(h)


    def step(self):
        '''
        One step of the search.

        In this method, a neighbor of the given estimate is chosen at random,
        using a gaussian neighborhood. It is accepted as a new estimate if it
        performs better in the cost function *or* if the temperature is high
        enough. In case it is not accepted, a gradient step is executed.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        f = self.__f
        x = self.__x
        fx = self.__fx

        # Next estimate
        xn = self.__nb(x)
        delta = f(xn) - fx
        if delta < 0 or exp(-delta/self.__t) > uniform():
            xr = xn
            er = abs(delta)
        elif self.__optm is not None:
            self.__optm.restart(x0 = x)
            xr, er = self.__optm.step()
        else:
            xr = x
            er = abs(delta)

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            xr = where(xr < r0, r0, xr)
            xr = where(xr > r1, r1, xr)

        # Update state
        self.__t = self.__t * self.__r
        self.__x = xr
        self.__fx = f(xr)
        return (xr, er)


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
class BinarySA(object):
    '''
    Simulated Annealing binary optimization.

    This is a simulated annealing optimizer implemented to work with vectors of
    bits, which can be floating point or integer numbers, characters or anything
    allowed by the ``struct`` module of the Python standard library. The
    neighborhood of an estimate is calculated by an appropriate method given in
    the class instantiation. Given the nature of this implementation, no
    alternate convergence can be used in the case of rejection of an estimate.
    '''
    def __init__(self, f, x0, ranges=[ ], fmt=None, neighbor=InvertBitsNeighbor,
                 T0=1000., rt=0.95, emax=1e-8, imax=1000):
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

            The default value for this parameter is ``None``, meaning that a
            default format is not supplied. If a format is not supplied, then
            the estimate will be passed as a bitarray to the objective function.
            This means that your function must take care to decode the bit
            stream to extract meaning from it.

          neighbor
            Neighbor function. This is a function used to compute the neighbor
            of the present estimate. You can use the ones defined in the
            ``neighbor`` module, or you can implement your own. In any case, the
            ``neighbor`` parameter must be an instance of ``BinaryNeighbor`` or
            of a subclass. Please, see the documentation on the ``neighbor``
            module for more information. The default is ``InvertBitsNeighbor``,
            which computes the new estimate by inverting some bits in the
            present estimate.

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
        self.__set_x(x0)

        # Determine ranges
        if ranges is None:
            self.ranges = None
        elif len(ranges) == 1:
            self.ranges = array(ranges * len(fmt))
        else:
            self.ranges = array(ranges)

        # Verifies the validity of the neighbor method
        try:
            issubclass(neighbor, BinaryNeighbor)
            neighbor = neighbor()
        except TypeError:
            pass
        if isinstance(neighbor, types.FunctionType):
            neighbor = BinaryNeighbor(neighbor)
        if not isinstance(neighbor, BinaryNeighbor):
            raise TypeError, 'not a valid neighbor function'
        else:
            self.__nb = neighbor

        self.__t = float(T0)
        self.__r = float(rt)
        self.__xbest = self.__x[:]        # Holds the best estimate so far and
        self.__fbest = f(self.__get_x())  # its value on the objective function.
        self.__emax = float(emax)
        self.__imax = int(imax)


    def __encode(self, values):
        '''
        Given the format of the estimate, encode the values. Return the
        corresponding bitarray. The format used is the one defined in the class
        instantiation.

        :Parameters:
          values
            An array, list or tuple of values to be encoded.

        :Returns:
          The encoded bitarray.
        '''
        if self.format is None:
            return values[:]
        else:
            x = bitarray()
            x.fromstring(struct.pack(self.format, *values))
            return x[:]

    def __decode(self, bits):
        '''
        Given the format of the estimate, decode the bitarray. Return the
        corresponding values in the form of a tuple. The format used is the one
        defined in the class instantiation.

        :Parameters:
          bits
            Bitarray containing the bits to be decoded. It must be compatible
            with the informed format.

        :Returns:
          A tuple with the decoded values.
        '''
        if self.format is None:
            return bits[:]
        return struct.unpack(self.format, bits.tostring())

    def __set_x(self, values):
        '''
        Setter for the estimate. The estimate must be given according to the
        format given as parameter in the instantiation of the class, otherwise
        an error will be raised; in that case, the estimate will be converted to
        a bitarray and stored as this. In case that no format was informed, then
        the estimate must be a bitarray and will be stored as such.

        :Parameters:
          x
            New estimate
        '''
        self.__x = self.__encode(values)

    def __get_x(self):
        '''
        Getter for the estimate. The estimate is decoded as the format supplied.
        If no format was supplied, then the estimate is returned as a bitarray.

        :Returns:
          The estimate, decoded as the format.
        '''
        return self.__decode(self.__x)

    x = property(__get_x, __set_x)
    '''The estimate of the minimum'''


    def __get_best(self):
        '''
        Getter for the best value so far. Returns a tuple containing both the
        best estimate and its value.

        :Returns:
          A tuple ``(x, fx)``, where ``x`` is the best estimate so far, and
          ``fx`` is its value on the objective function.
        '''
        return (self.__decode(self.__xbest), self.__fbest)

    best = property(__get_best, None)
    '''A tuple ``(x, fx)``, where ``x`` is the best estimate so far, and
    ``fx`` is its value on the objective function.'''


    def restart(self, x0, ranges=None, T0=1000., rt=0.95, h=0.5):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate. Restartings are essential to the working of
        simulated annealing algorithms, to allow them to leave local minima.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

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
        '''
        self.__set_x(x0)
        if ranges is not None:
            if len(ranges) == 1:
                self.ranges = array(ranges * len(fmt))
            else:
                self.ranges = array(ranges)
        self.__t = float(T0)
        self.__r = float(rt)


    def step(self):
        '''
        One step of the search.

        In this method, a neighbor of the given estimate is obtained from the
        present estimate by choosing ``nb`` bits and inverting them. It is
        accepted as a new estimate if it performs better in the cost function
        *or* if the temperature is high enough. In case it is not accepted, the
        previous estimate is mantained.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        # Keep track of the best result so far.
        f = self.__f
        fx = f(self.__get_x())
        if self.__fbest > fx:
            self.__xbest = self.__x[:]
            self.__fbest = fx

        # Perform computation of neighbor by changing a number of bits in the
        # bitarray representation of the estimate.
        xn = self.__nb(self.__x)

        # Performs a sanity check in the values.
        xs = self.__decode(xn)
        r = self.ranges
        if r is not None:
            x0 = r[:, 0]
            x1 = r[:, 1]
            if any(xs < x0) or any(xs > x1) or any(isnan(xs)):
                xs = [ uniform(r0, r1) for r0, r1 in r ]

        # Update step, using temperature to decide if the new estimate is kept.
        delta = f(xs) - fx
        if delta < 0 or exp(-delta/self.__t) > uniform():
            self.__set_x(xs)
        self.__t = self.__t * self.__r
        return (self.__decode(self.__xbest), abs(delta))


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
        return self.__decode(self.__xbest), e


################################################################################
# Test
if __name__ == "__main__":
    pass