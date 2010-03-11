################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/af.py
# Activation functions and base class
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Base activation functions and base class

Activation functions define if a neuron is activated or not. There are a lot of
different definitions for activation functions in the literature, and this
sub-package implements some of them. An activation function is defined by its
response and its derivative. Being conveniently defined as classes, it is
possible to define a custom derivative method.

In this package, also, there is a base class that should be subclassed if you
want to define your own activation function. This class, however, can be
instantiated with a standard Python function as an initialization parameter, and
it is adjusted to work with the internals of the package.

If the base class is instantiated, then the function should take a real number
as input, and return a real number. The response of the function determines if
the neuron is activated or not.
"""


################################################################################
from numpy import vectorize, array, where, ones, select, exp, pi, arctan, tanh, cosh
import types


################################################################################
# Classes
################################################################################
class Activation(object):
    '''
    Base class for activation functions.

    This class can be used as base for activation functions. A subclass should
    have at least three methods, described below:

      __init__
        This method should be used to configure the function. In general, some
        parameters to change the behaviour of a simple function is passed. In a
        subclass, the ``__init__`` method should call the mother class
        initialization procedure.
      __call__
        The ``__call__`` interface is the function call. It should receive a
        *vector* of real numbers and return a *vector* of real numbers. Using
        the capabilities of the ``numpy`` module will help a lot. In case you
        don't know how to use, maybe instantiating this class instead will work
        better (see below).
      derivative
        This method implements the derivative of the activation function. It is
        used in the learning methods. If one is not provided (but remember to
        call the superclass ``__init__`` so that it is created).
    '''
    def __init__(self, f=None, df=None):
        '''
        Initializes the activation function.

        Instantiating this class creates and adjusts a standard Python function
        to work with layers of neurons.

        :Parameters:
          f
            The activation function. It can be created as a lambda function or
            any other method, but it should take a real value, corresponding to
            the activation potential of a neuron, and return a real value,
            corresponding to its activation. Defaults to ``None``, if none is
            given, the identity function is used.
          df
            The derivative of the above function. It can be defined as above, or
            not given. If not given, an estimate is calculated based on the
            given function. Defaults to ``None``.
        '''
        if isinstance(f, types.FunctionType):
            self.__f = vectorize(f)
        elif f is None:
            self.__f = lambda x: array(x, dtype=float)
        else:
            raise ValueError, 'invalid function'
        if df is None:
            self.d = self.derivative
            '''An alias to the derivative of the function.'''
        else:
            self.d = df
            self.derivative = df


    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return self.__f(x)


    def derivative(self, x, dx=5.0e-5):
        '''
        An estimate of the derivative of the activation function.

        This method estimates the derivative using difference equations. This is
        a simple estimate, but efficient nonetheless.

        :Parameters:
          x
            A real number or vector of real numbers representing the point over
            which the derivative is to be calculated.
          dx
            The value of the interval of the estimate. The smaller this number
            is, the better. However, if made too small, the precision is not
            enough to avoid errors. This defaults to 5e-5, which is the values
            that gives the best results.

        :Returns:
          The value of the derivative over the given point.
        '''
        return (self(x+dx/2.0)-self(x-dx/2.0)) / dx


################################################################################
class Threshold(Activation):
    '''
    Threshold activation function.
    '''
    def __init__(self, threshold=0.0, amplitude=1.0):
        '''
        Initializes the object.

        :Parameters:
          threshold
            The threshold value. If the value of the input is lower than this,
            the function is 0, otherwise, it is the given ``amplitude``.
          amplitude
            The maximum value of the function.
        '''
        self.__t = float(threshold)
        self.__a = float(amplitude)
        self.d = self.derivative

    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return where(x >= self.__t, self.__a, 0.0)

    def derivative(self, x):
        '''
        The function derivative. Technically, this function doesn't have a
        derivative, but making it equals to 1, this can be used in learning
        algorithms.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The derivative of the activation function applied over the input
          vector.
        '''
        try:
            return ones(x.shape)
        except AttributeError:
            return 1.0

Step = Threshold
'''Alias to ``Threshold``'''


################################################################################
class Linear(Activation):
    '''
    Identity activation function
    '''
    def __init__(self):
        '''
        Initializes the function
        '''
        self.d = self.derivative

    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return array(x, dtype = float)

    def derivative(self, x):
        '''
        The function derivative.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The derivative of the activation function applied over the input
          vector.
        '''
        try:
            return ones(x.shape)
        except AttributeError:
            return 1.0

Identity = Linear
'''An alias to ``Linear``'''


################################################################################
class Ramp(Activation):
    '''
    Ramp activation function
    '''
    def __init__(self, p0=(-0.5, 0.0), p1=(0.5, 1.0)):
        '''
        Initializes the object.

        Two points are needed to set this function. They are used to determine
        where the ramp begins and where it ends.

        :Parameters:
          p0
            The starting point, given as a tuple ``(x0, y0)``. For values of the
            input below ``x0``, the function returns ``y0``. Defaults to
            ``(-0.5, 0.0)``.
          p1
            The ending point, given as a tuple ``(x1, y1)``. For values of the
            input above ``x1``, the function returns ``y1``. Defaults to
            ``(0.5, 1.0)``.
         '''
        self.__x0 = float(p0[0])
        self.__y0 = float(p0[1])
        self.__x1 = float(p1[0])
        self.__y1 = float(p1[1])
        self.__a = (self.__y1 - self.__y0) / (self.__x1 - self.__x0)
        self.d = self.derivative

    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return select([ x < self.__x0, x < self.__x1 ],
                      [ self.__y0, self.__a * (x - self.__x0) + self.__y0 ],
                      self.__y1)

    def derivative(self, x):
        '''
        The function derivative.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The derivative of the activation function applied over the input
          vector.
        '''
        return select([ x < self.__x0, x < self.__x1 ],
                      [ 0.0, self.__a ], 0.0)


################################################################################
class Sigmoid(Activation):
    '''
    Sigmoid activation function
    '''
    def __init__(self, a = 1.0, x0 = 0.0):
        '''
        Initializes the object.

        :Parameters:
          a
            The slope of the function in the center ``x0``. Defaults to 1.0.
          x0
            The center of the sigmoid. Defaults to 0.0.
        '''
        self.__a = float(a)
        self.__x0 = float(x0)
        self.d = self.derivative

    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return 1.0 / (1.0 + exp(- self.__a*(x - self.__x0)))

    def derivative(self, x):
        '''
        The function derivative.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The derivative of the activation function applied over the input
          vector.
        '''
        t = exp(-self.__a * (x - self.__x0))
        return self.__a * t / (1 + t)**2

Logistic = Sigmoid
'''An alias to ``Sigmoid``'''


################################################################################
class Signum(Activation):
    '''
    Signum activation function
    '''
    def __init__(self):
        '''
        Initializes the object.
        '''
        self.d = self.derivative

    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return where(x == 0.0, 0.0, x / abs(x))

    def derivative(self, x):
        '''
        The function derivative. Technically, this function doesn't have a
        derivative, but making it equals to 1, this can be used in learning
        algorithms.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The derivative of the activation function applied over the input
          vector.
        '''
        try:
            return ones(x.shape)
        except AttributeError:
            return 1.0


################################################################################
class ArcTan(Activation):
    '''
    Inverse tangent activation function
    '''
    def __init__(self, a = 1.0, x0 = 0.0):
        '''
        Initializes the object

        :Parameters:
          a
            The slope of the function in the center ``x0``. Defaults to 1.0.
          x0
            The center of the sigmoid. Defaults to 0.0.
        '''
        self.__a = float(a)
        self.__x0 = float(x0)
        self.d = self.derivative

    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return self.__a / pi * arctan(x - self.__x0)

    def derivative(self, x):
        '''
        The function derivative.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The derivative of the activation function applied over the input
          vector.
        '''
        return self.__a / pi / (1.0 + (x - self.__x0)**2)


################################################################################
class TanH(Activation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self, a = 1.0, x0 = 0.0):
        '''
        Initializes the object

        :Parameters:
          a
            The slope of the function in the center ``x0``. Defaults to 1.0.
          x0
            The center of the sigmoid. Defaults to 0.0.
        '''
        self.__a = float(a)
        self.__x0 = float(x0)
        self.d = self.derivative

    def __call__(self, x):
        '''
        Call interface to the object.

        This method applies the activation function over a vector of activation
        potentials, and returns the results.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The activation function applied over the input vector.
        '''
        return self.__a * tanh(x - self.__x0)

    def derivative(self, x):
        '''
        The function derivative.

        :Parameters:
          x
            A real number or a vector of real numbers representing the
            activation potential of a neuron or a layer of neurons.

        :Returns:
          The derivative of the activation function applied over the input
          vector.
        '''
        return self.__a / cosh(x - self.__x0)**2


################################################################################
# Test
if __name__ == "__main__":
    pass