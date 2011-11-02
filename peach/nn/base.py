################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/base.py
# Basic definitions for layers of neurons
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic definitions for layers of neurons.

This subpackage implements the basic classes used with neural networks. A neural
network is basically implemented as a layer of neurons. To speed things up, a
layer is implemented as a array, where each line represents the weight vector
of a neuron. Further definitions and algorithms are based on this definition.
"""


################################################################################
from numpy import dot, array, reshape, vstack, ones
from numpy.random import randn
from af import Activation, Linear


_BIAS = ones((1, 1), dtype=float)
"""This constant vector is defined to implement in a fast way the bias of a
neuron, as an input of value 1, stacked over the real input to the neuron."""

################################################################################
# Classes
################################################################################
class Layer(object):
    '''
    Base class for neural networks.

    This class implements a layer of neurons. It is represented by a array of
    real values. Each line of the array represents the weight vector of a
    single neuron. If the neurons on the layer are biased, then the first
    element of the weight vector is the bias weight, and the bias input is
    always valued 1. Also, to each layer is associated an activation function,
    that determines if the neuron is fired or not. Please, consult the module
    ``af`` to see more about activation functions.

    In general, this class shoulb be subclassed if you want to use neural nets.
    But, as neural nets are very different one from the other, check carefully
    the documentation to see if the attributes, properties and methods are
    suited to your task.
    '''
    def __init__(self, shape, phi=Linear, bias=False):
        """
        Initializes the layer.

        A layer is represented by a array where each line is the weight vector
        of a single neuron. The first element of the vector is the bias weight,
        in case the neuron is biased. Associated with the layer is an activation
        function defined in an appropriate way.

        :Parameters:
          shape
            Stablishes the size of the layer. It must be a two-tuple of the
            format ``(m, n)``, where ``m`` is the number of neurons in the
            layer, and ``n`` is the number of inputs of each neuron. The neurons
            in the layer all have the same number of inputs.
          phi
            The activation function. It can be an ``Activation`` object (please,
            consult the ``af`` module) or a standard Python function. In this
            case, it must receive a single real value and return a single real
            value which determines if the neuron is activated or not. Defaults
            to ``Linear``.
          bias
            If ``True``, then the neurons on the layer are biased. That means
            that an additional weight is added to each neuron to represent the
            bias. If ``False``, no modification is made.
        """
        m, n = shape
        if bias:
            n = n + 1
        self.__weights = randn(m, n)
        self.__size = m
        self.__inputs = n

        # The ``phi`` property sets the activation function, see below.
        self.phi = phi

        # Properties.
        self.__v = None
        self.__y = None
        self.__bias = bias


    def __getsize(self):
        return self.__size
    size = property(__getsize, None)
    '''Number of neurons in the layer. Not writable.'''


    def __getinputs(self):
        if self.__bias:
            return self.__inputs - 1
        else:
            return self.__inputs
    inputs = property(__getinputs, None)
    '''Number of inputs for each neuron in the layer. Not writable.'''


    def __getshape(self):
        if self.__bias:
            return (self.__size, self.__inputs - 1)
        else:
            return (self.__size, self.__inputs)
    shape = property(__getshape, None)
    '''Shape of the layer, given in the format of a tuple ``(m, n)``, where
    ``m`` is the number of neurons in the layer, and ``n`` is the number of
    inputs in each neuron. Not writable.'''


    def __getbias(self):
        return self.__bias
    bias = property(__getbias, None)
    '''True if the neuron is biased. Not writable.'''


    def __getweights(self):
        return self.__weights
    def __setweights(self, m):
        self.__weights = array(reshape(m, self.weights.shape))
    weights = property(__getweights, __setweights)
    '''A ``numpy`` array containing the synaptic weights of the network. Each
    line is the weight vector of a neuron. It is writable, but the new weight
    array must be the same shape of the neuron, or an exception is raised.'''


    def __getphi(self):
        return self.__phi
    def __setphi(self, phi):
        try:
            issubclass(phi, Activation)
            phi = phi()
        except TypeError:
            pass
        if isinstance(phi, Activation):
            self.__phi = phi
        else:
            self.__phi = Activation(phi)
    phi = property(__getphi, __setphi)
    '''The activation function. It can be set with an ``Activation`` instance or
    a standard Python function. If a standard function is given, it must receive
    a real value and return a real value that is the activation value of the
    neuron. In that case, it is adjusted to work accordingly with the internals
    of the layer.'''


    def __getv(self):
        if self.__v is None:
            raise ValueError, 'activation potential unavailable'
        else:
            return self.__v
    v = property(__getv, None)
    '''The activation potential of the neuron. Not writable, and only available
    after the neuron is fed some input.'''


    def __gety(self):
        if self.__y is None:
            raise ValueError, 'activation unavailable'
        else:
            return self.__y
    y = property(__gety, None)
    '''The activation value of the neuron. Not writable, and only available
    after the neuron is fed some input.'''


    def __getitem__(self, n):
        '''
        The ``[ ]`` get interface.

        The input to this method is forwarded to the ``weights`` property. That
        means that it will return the respective line/element of the weight
        array.

        :Parameters:
          n
            A slice object containing the elements referenced. Since it is
            forwarded to an array, it behaves exactly as one.

        :Returns:
          The element or elements in the referenced indices.
        '''
        return self.weights[n]


    def __setitem__(self, n, w):
        '''
        The ``[ ]`` set interface.

        The inputs to this method are forwarded to the ``weights`` property.
        That means that it will set the respective line/element of the weight
        array.

        :Parameters:
          n
            A slice object containing the elements referenced. Since it is
            forwarded to an array, it behaves exactly as one.

          w
            A value or array of values to be set in the given indices.
       '''
        self.weights[n] = w


    def __call__(self, x):
        '''
        The feedforward method to the layer.

        The ``__call__`` interface should be called if the answer of the neuron
        to a given input vector ``x`` is desired. *This method has collateral
        effects*, so beware. After the calling of this method, the ``v`` and
        ``y`` properties are set with the activation potential and the answer of
        the neurons, respectivelly.

        :Parameters:
          x
            The input vector to the layer.

        :Returns:
          The vector containing the answer of every neuron in the layer, in the
          respective order.
        '''
        # Adjusts the input vector in case the neuron is biased. Also, the
        # input vector is reshaped as a column-vector.
        x = reshape(x, (self.inputs, 1))
        if self.__bias:
            x = vstack((_BIAS, x))
        self.__v = dot(self.weights, x)
        self.__y = self.phi(self.__v)
        return self.__y


################################################################################
# Test
if __name__ == "__main__":
    pass