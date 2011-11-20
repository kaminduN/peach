################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/lrules.py
# Learning rules for neural networks
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Learning rules for neural networks and base classes for custom learning.

This sub-package implements learning methods commonly used with neural networks.
There are a lot of different topologies and different learning methods for each
one. It is very difficult to find a consistent framework for defining learning
methods, in consequence. This method defines some base classes that are coupled
with the neural networks that they are supposed to work with. Also, based on
these classes, some of the traditional methods are implemented.

If you want to implement a different learning method, you must subclass the
correct base class. Consult the classes below. Also, pay attention to how the
implementation is expected to behave. Since learning algorithms are usually
somewhat complex, care should be taken to make everything work accordingly.
"""


################################################################################
from numpy import ones, hstack, reshape, dot, sum, exp


_BIAS = ones((1, 1), dtype=float)
"""This constant vector is defined to implement in a fast way the bias of a
neuron, as an input of value 1, stacked over the real input to the neuron."""


################################################################################
# Classes
################################################################################
class FFLearning(object):
    '''
    Base class for FeedForwarding Multilayer neural networks.

    As a base class, this class doesn't do anything. You should subclass this
    class if you want to implement a learning method for multilayer networks.

    A learning method for a neural net of this kind must deal with a
    ``FeedForward`` instance. A ``FeedForward`` object is a list of ``Layers``
    (consulting the documentation of these classes is important!). Each layer is
    a bidimensional array, where each line represents the synaptic weights of a
    single neuron. So, a multilayer network is actually a three-dimensional
    array, if you will. Usually, though, learning methods for this kind of net
    propagate some measure of the error from the output back to the input (the
    ``BackPropagation`` method, for instance).

    A class implementing a learning method should have at least two methods:

      __init__
        The ``__init__`` method should initialize the object. It is in general
        used to configure some property of the learning algorithm, such as the
        learning rate.
      __call__
        The ``__call__`` interface is how the method should interact with the
        neural network. It should have the following signature::

          __call__(self, nn, x, d)

        where ``nn`` is the ``FeedForward`` instance to be modified *in loco*,
        ``x`` is the input vector and ``d`` is the desired response of the net
        for that particular input vector. It should return nothing.
    '''
    def __call__(self, nn, x, d):
        '''
        The ``__call__`` interface.

        Read the documentation for this class for more information. A call to
        the class should have the following parameters:

        :Parameters:
          nn
            A ``FeedForward`` neural network instance that is going to be
            modified by the learning algorithm. The modification is made *in
            loco*, that is, the synaptic weights of ``nn`` should be modified
            in place, and not returned from this function.
          x
            The input vector from the training set.
          d
            The desired response for the given input vector.
        '''
        raise NotImplementedError, 'learning rule not defined'


################################################################################
class LMS(FFLearning):
    '''
    The Least-Mean-Square (LMS) learning method.

    The LMS method is a very simple method of learning, thoroughly described in
    virtually every book about the subject. Please, consult a good book on
    neural networks for more information. This implementation tries to use the
    ``numpy`` routines as much as possible for better efficiency.
    '''
    def __init__(self, lrate=0.05):
        '''
        Initializes the object.

        :Parameters:
          lrate
            Learning rate to be used in the algorithm. Defaults to 0.05.
        '''
        self.lrate = lrate
        '''Learning rate used in the algorithm.'''


    def __call__(self, nn, x, d):
        '''
        The ``__call__`` interface.

        The learning implementation. Read the documentation for the base class
        for more information. A call to the class should have the following
        parameters:

        :Parameters:
          nn
            A ``FeedForward`` neural network instance that is going to be
            modified by the learning algorithm. The modification is made *in
            loco*, that is, the synaptic weights of ``nn`` should be modified
            in place, and not returned from this function.
          x
            The input vector from the training set.
          d
            The desired response for the given input vector.
        '''
        # g would be like the local error gradient for each neuron. In LMS, this
        # serves only to propagate the error
        d = reshape(d, (nn.y.shape))
        g = d - nn.y

        # The error is backpropagated, thus the lists are inverted. To combine
        # each layer, we ``zip`` them.
        for w1, w2 in zip(nn[:-1][::-1], nn[1:][::-1]):

            # xs is the input vector, transposed because of backpropagation.
            xs = w1.y.transpose()

            # Adjusts for bias.
            if w2.bias:
                xs = hstack((_BIAS, xs))
                wt = w2.weights[:, 1:].transpose()
            else:
                wt = w2.weights.transpose()

            # Update synaptic weights
            dw = self.lrate * dot(g, xs)
            w2.weights = w2.weights + dw

            # Backpropagate the error.
            g = dot(wt, g)

        # Repeat the procedure for the first layer.
        w = nn[0]
        xs = x.reshape((1, w.inputs))
        if w.bias:
            xs = hstack((_BIAS, xs))
        dw = self.lrate * dot(g, xs)
        w.weights = w.weights + dw


WidrowHoff = LMS
'''Alias for the LMS class'''

DeltaRule = LMS
'''Alias for the LMS class'''


################################################################################
class BackPropagation(FFLearning):
    '''
    The BackPropagation learning method.

    The backpropagation method is a very simple method of learning, thoroughly
    described in virtually every book about the subject. Please, consult a good
    book on neural networks for more information. This implementation tries to
    use the ``numpy`` routines as much as possible for better efficiency.
    '''
    def __init__(self, lrate=0.05):
        '''
        Initializes the object.

        :Parameters:
          lrate
            Learning rate to be used in the algorithm. Defaults to 0.05.
        '''
        self.lrate = lrate
        '''Learning rate used in the algorithm.'''


    def __call__(self, nn, x, d):
        '''
        The ``__call__`` interface.

        The learning implementation. Read the documentation for the base class
        for more information. A call to the class should have the following
        parameters:

        :Parameters:
          nn
            A ``FeedForward`` neural network instance that is going to be
            modified by the learning algorithm. The modification is made *in
            loco*, that is, the synaptic weights of ``nn`` should be modified
            in place, and not returned from this function.
          x
            The input vector from the training set.
          d
            The desired response for the given input vector.
        '''
        # g is the local error gradient for each neuron.
        d = reshape(d, (nn.y.shape))
        g = (d - nn.y) * nn[-1].phi.d(nn[-1].v)

        # The error is backpropagated, thus the lists are inverted. To combine
        # each layer, we ``zip`` them.
        for w1, w2 in zip(nn[:-1][::-1], nn[1:][::-1]):

            # xs is the input vector, transposed because of backpropagation.
            xs = w1.y.transpose()

            # Adjusts for bias.
            if w2.bias:
                xs = hstack((_BIAS, xs))
                wt = w2.weights[:, 1:].transpose()
            else:
                wt = w2.weights.transpose()

            # Update synaptic weights
            dw = self.lrate * dot(g, xs)
            w2.weights = w2.weights + dw

            # Backpropagate the error.
            g = dot(wt, g) * w1.phi.d(w1.v)

        # Repeat the procedure for the first layer.
        w = nn[0]
        xs = x.reshape((1, w.inputs))
        if w.bias:
            xs = hstack((_BIAS, xs))
        dw = self.lrate * dot(g, xs)
        w.weights = w.weights + dw


################################################################################
class SOMLearning(object):
    '''
    Base class for Self-Organizing Maps.

    As a base class, this class doesn't do anything. You should subclass this
    class if you want to implement a learning method for self-organizing maps.

    A learning method for a neural net of this kind must deal with a ``SOM``
    instance. A ``SOM`` object is a ``Layer`` (consulting the documentation of
    these classes is important!).

    A class implementing a learning method should have at least two methods:

      __init__
        The ``__init__`` method should initialize the object. It is in general
        used to configure some property of the learning algorithm, such as the
        learning rate.
      __call__
        The ``__call__`` interface is how the method should interact with the
        neural network. It should have the following signature::

          __call__(self, nn, x)

        where ``nn`` is the ``SOM`` instance to be modified *in loco*, and ``x``
        is the input vector. It should return nothing.
    '''
    def __call__(self, nn, x, d):
        '''
        The ``__call__`` interface.

        Read the documentation for this class for more information. A call to
        the class should have the following parameters:

        :Parameters:
          nn
            A ``SOM`` neural network instance that is going to be modified by
            the learning algorithm. The modification is made *in loco*, that is,
            the synaptic weights of ``nn`` should be modified in place, and not
            returned from this function.
          x
            The input vector from the training set.
        '''
        raise NotImplementedError, 'learning rule not defined'


################################################################################
class WinnerTakesAll(SOMLearning):
    '''
    Purely competitive learning method without learning rate adjust.

    A winner-takes-all strategy detects the winner on the self-organizing map
    and adjusts it in the direction of the input vector, scaled by the learning
    rate. Its tendency is to cluster around the gravity center of the points in
    the training set.
    '''
    def __init__(self, lrate=0.05):
        '''
        Initializes the object.

        :Parameters:
          lrate
            Learning rate to be used in the algorithm. Defaults to 0.05.
        '''
        self.lrate = lrate
        '''Learning rate used with the algorithm.'''


    def __call__(self, nn, x):
        '''
        The ``__call__`` interface.

        The learning implementation. Read the documentation for the base class
        for more information. A call to the class should have the following
        parameters:

        :Parameters:
          nn
            A ``SOM`` neural network instance that is going to be modified by
            the learning algorithm. The modification is made *in loco*, that is,
            the synaptic weights of ``nn`` should be modified in place, and not
            returned from this function.
          x
            The input vector from the training set.
        '''
        xs = x.reshape((nn.inputs, 1))
        i = nn.y
        w = nn.weights
        dw = self.lrate * (x - w[i])
        w[i] = w[i] + dw


WTA = WinnerTakesAll
'''Alias for the ``WinnerTakesAll`` class'''


################################################################################
class Competitive(SOMLearning):
    '''
    Competitive learning with time adjust of the learning rate.

    A competitive strategy detects the winner on the self-organizing map and
    adjusts it in the direction of the input vector, scaled by the learning
    rate. Its tendency is to cluster around the gravity center of the points in
    the training set. As time passes, the learning rate grows smaller, this
    allows for better adjustment of the synaptic weights.
    '''
    def __init__(self, lrate=0.05, tl=1000.):
        '''
        Initializes the object.

        :Parameters:
          lrate
            Learning rate to be used in the algorithm. Defaults to 0.05.
          tl
            Time constant that measures how many iterations will be needed to
            reduce the learning rate to a small value. Defaults to 1000.
        '''
        self.lrate = lrate
        self.__lrate = 1.0
        self.__lrm = exp(-1.0/float(tl))


    def __call__(self, nn, x):
        '''
        The ``__call__`` interface.

        The learning implementation. Read the documentation for the base class
        for more information. A call to the class should have the following
        parameters:

        :Parameters:
          nn
            A ``SOM`` neural network instance that is going to be modified by
            the learning algorithm. The modification is made *in loco*, that is,
            the synaptic weights of ``nn`` should be modified in place, and not
            returned from this function.
          x
            The input vector from the training set.
        '''
        xs = x.reshape((nn.inputs, 1))
        i = nn.y
        w = nn.weights

        # Adjusts the learning rate according to an exponential rule
        lrate = self.lrate * self.__lrate
        self.__lrate = self.__lrate * self.__lrm

        # Updates the weights
        dw = lrate * (x - w[i])
        w[i] = w[i] + dw


################################################################################
class Cooperative(SOMLearning):
    '''
    Cooperative learning with time adjust of the learning rate and neighborhood
    function to propagate cooperation

    A cooperative strategy detects the winner on the self-organizing map and
    adjusts it in the direction of the input vector, scaled by the learning
    rate. Its tendency is to cluster around the gravity center of the points in
    the training set. As time passes, the learning rate grows smaller, this
    allows for better adjustment of the synaptic weights.

    Also, a neighborhood is defined on the winner. Neurons close to the winner
    are also updated in the direction of the input vector, although with a
    smaller scale determined by the neighborhood function. A neighborhood
    function is 1. at 0., and decreases monotonically as the distance increases.

    *There are issues with this class!* -- some of the class capabilities are
    yet to be developed.
    '''
    def __init__(self, lrate=0.05, tl=1000, tn=1000):
        '''
        Initializes the object.

        :Parameters:
          lrate
            Learning rate to be used in the algorithm. Defaults to 0.05.
          tl
            Time constant that measures how many iterations will be needed to
            reduce the learning rate to a small value. Defaults to 1000.
          tn
            Time constant that measures how many iterations will be needed to
            shrink the neighborhood. Defaults to 1000.
        '''
        self.__neighbor = 1.0
        self.__lrate = 1.0
        self.__lrm = exp(-1.0/float(tl))
        self.__nbm = exp(-1.0/float(tn))
        self.__s0 = float(s0)


    def __call__(self, nn, x):
        '''
        The ``__call__`` interface.

        The learning implementation. Read the documentation for the base class
        for more information. A call to the class should have the following
        parameters:

        :Parameters:
          nn
            A ``SOM`` neural network instance that is going to be modified by
            the learning algorithm. The modification is made *in loco*, that is,
            the synaptic weights of ``nn`` should be modified in place, and not
            returned from this function.
          x
            The input vector from the training set.
        '''
        xs = x.reshape((nn.inputs, 1))
        i = nn.y
        w = nn.weights
        wi = nn.weights[i, :]

        # Adjusts the learning rate according to an exponential rule
        lrate = nn.lrate * self.__lrate
        self.__lrate = self.__lrate * self.__lrm

        # Apply neighborhood function.
        s = self.__s0 * self.__neighbor
        self.__neighbor = self.__neighbor * self.__nbm
        d = sum((wi - w)**2, axis=1)
        h = exp(-d/(2*s**2))

        # Updates the weights
        dw = lrate * h * (x - w).transpose()
        w = w + dw.transpose()
        nn.weights = w


################################################################################
# Test
if __name__ == "__main__":
    pass