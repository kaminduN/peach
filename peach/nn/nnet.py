################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/nn.py
# Basic topologies of neural networks
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic topologies of neural networks.

This sub-package implements various neural network topologies, see the complete
list below. These topologies are implemented using the ``Layer`` class of the
``base`` sub-package. Please, consult the documentation of that module for more
information on layers of neurons. The neural nets implemented here don't derive
from the ``Layer`` class, instead, they have instance variables to take control
of them. Thus, there is no base class for networks. While subclassing the
classes of this module is usually safe, it is recomended that a new kind of
net is developed from the ground up.
"""

################################################################################
from numpy import sum, abs, reshape, sqrt, argmin
import random

from base import *
from af import *
from lrules import *


################################################################################
class FeedForward(list):
    '''
    Classic completely connected neural network.

    A feedforward neural network is implemented as a list of layers, each layer
    being a ``Layer`` object (please consult the documentation on the ``base``
    module for more information on layers). The layers are completely connected,
    which means that every neuron in one layers is connected to every other
    neuron in the following layer.

    There is a number of learning methods that are already implemented, but in
    general, any learning class derived from ``FFLearning`` can be used. No
    other kind of learning can be used. Please, consult the documentation on the
    ``lrules`` (*learning rules*) module.
    '''
    def __init__(self, layers, phi=Linear, lrule=BackPropagation, bias=False):
        '''
        Initializes a feedforward neural network.

        A feedforward network is implemented as a list of layers, completely
        connected.

        :Parameters:
          layers
            A list of integers containing the shape of the network. The first
            element of the list is the number of inputs of the network (or, as
            somebody prefer, the number of input neurons); the number of outputs
            is the number of neurons in the last layer. Thus, at least two
            numbers should be given.
          phi
            The activation functions to be used with each layer of the network.
            Please consult the ``Layer`` documentation in the ``base`` module
            for more information. This parameter can be a single function or a
            list of functions. If only one function is given, then the same
            function is used in every layer. If a list of functions is given,
            then the layers use the functions in the sequence given. Note that
            heterogeneous networks can be created that way. Defaults to
            ``Linear``.
          lrule
            The learning rule used. Only ``FFLearning`` objects (instances of
            the class or of the subclasses) are allowed. Defaults to
            ``BackPropagation``. Check the ``lrules`` documentation for more
            information.
          bias
            If ``True``, then the neurons are biased.
        '''
        list.__init__(self, [ ])
        layers = list(layers)
        for n, m in zip(layers[:-1], layers[1:]):
            self.append(Layer((m, n), bias=bias))
        self.phi = phi
        self.__n = len(self)
        self.__lrule = lrule
        if isinstance(lrule, FFLearning):
            self.__lrule = lrule
        else:
            try:
                issubclass(lrule, FFLearning)
                self.__lrule = lrule()
            except TypeError:
                raise ValueError, 'uncompatible learning rule'


    def __getnlayers(self):
        return self.__n
    nlayers = property(__getnlayers, None)
    '''Number of layers of the neural network. Not writable.'''


    def __getbias(self):
        r = [ ]
        for l in self:
            r.append(l.bias)
        return tuple(r)
    bias = property(__getbias, None)
    '''A tuple containing the bias of each layer. Not writable.'''


    def __gety(self):
        return self[-1].y
    y = property(__gety, None)
    '''A list of activation values for each neuron in the last layer of the
    network, ie., the answer of the network. This property is available only
    after the network is fed some input.'''


    def __getphi(self):
        r = [ ]
        for l in self:
            r.append(l.phi)
        return tuple(r)
    def __setphi(self, phis):
        try:
            phis = tuple(phis)
            for w, v in zip(self, phis):
                w.phi = v
        except TypeError:
            for w in self:
                w.phi = phis
    phi = property(__getphi, __setphi)
    '''Activation functions for every layer in the network. It is a list of
    ``Activation`` objects, but can be set with only one function. In this case,
    the same function is used for every layer.'''


    def __call__(self, x):
        '''
        The feedforward method of the network.

        The ``__call__`` interface should be called if the answer of the neuron
        network to a given input vector ``x`` is desired. *This method has
        collateral effects*, so beware. After the calling of this method, the
        ``y`` property is set with the activation potential and the answer of
        the neurons, respectivelly.

        :Parameters:
          x
            The input vector to the network.

        :Returns:
          The vector containing the answer of every neuron in the last layer, in
          the respective order.
        '''
        for w in self:
            x = w(x)
        return self[-1].y


    def learn(self, x, d):
        '''
        Applies one example of the training set to the network.

        Using this method, one iteration of the learning procedure is made with
        the neurons of this network. This method presents one example (not
        necessarilly of a training set) and applies the learning rule over the
        network. The learning rule is defined in the initialization of the
        network, and some are implemented on the ``lrules`` method. New methods
        can be created, consult the ``lrules`` documentation but, for
        ``FeedForward`` instances, only ``FFLearning`` learning is allowed.

        Also, notice that *this method only applies the learning method!* The
        network should be fed with the same input vector before trying to learn
        anything first. Consult the ``feed`` and ``train`` methods below for
        more ways to train a network.

        :Parameters:
          x
            Input vector of the example. It should be a column vector of the
            correct dimension, that is, the number of input neurons.
          d
            The desired answer of the network for this particular input vector.
            Notice that the desired answer should have the same dimension of the
            last layer of the network. This means that a desired answer should
            be given for every output of the network.

        :Returns:
          The error obtained by the network.
        '''
        self.__lrule(self, x, d)
        return sum(abs(d - self.y))


    def feed(self, x, d):
        '''
        Feed the network and applies one example of the training set to the
        network.

        Using this method, one iteration of the learning procedure is made with
        the neurons of this network. This method presents one example (not
        necessarilly of a training set) and applies the learning rule over the
        network. The learning rule is defined in the initialization of the
        network, and some are implemented on the ``lrules`` method. New methods
        can be created, consult the ``lrules`` documentation but, for
        ``FeedForward`` instances, only ``FFLearning`` learning is allowed.

        Also, notice that *this method feeds the network* before applying the
        learning rule. Feeding the network has collateral effects, and some
        properties change when this happens. Namely, the ``y`` property is set.
        Please consult the ``__call__`` interface.

        :Parameters:
          x
            Input vector of the example. It should be a column vector of the
            correct dimension, that is, the number of input neurons.
          d
            The desired answer of the network for this particular input vector.
            Notice that the desired answer should have the same dimension of the
            last layer of the network. This means that a desired answer should
            be given for every output of the network.

        :Returns:
          The error obtained by the network.
        '''
        self(x)
        return self.learn(x, d)


    def train(self, train_set, imax=2000, emax=1e-5, randomize=False):
        '''
        Presents a training set to the network.

        This method automatizes the training of the network. Given a training
        set, the examples are shown to the network (possibly in a randomized
        way). A maximum number of iterations or a maximum admitted error should
        be given as a stop condition.

        :Parameters:
          train_set
            The training set is a list of examples. It can have any size and can
            contain repeated examples. In fact, the definition of the training
            set is open. Each element of the training set, however, should be a
            two-tuple ``(x, d)``, where ``x`` is the input vector, and ``d`` is
            the desired response of the network for this particular input. See
            the ``learn`` and ``feed`` for more information.
          imax
            The maximum number of iterations. Examples from the training set
            will be presented to the network while this limit is not reached.
            Defaults to 2000.
          emax
            The maximum admitted error. Examples from the training set will be
            presented to the network until the error obtained is lower than this
            limit. Defaults to 1e-5.
          randomize
            If this is ``True``, then the examples are shown in a randomized
            order. If ``False``, then the examples are shown in the same order
            that they appear in the ``train_set`` list. Defaults to ``False``.
        '''
        i = 0
        error = 1
        s = len(train_set)
        while i<imax and error>emax:
            if randomize:
                x, d = random.choice(train_set)
            else:
                x, d = train_set[i%s]
            error = self.feed(x, d)
            i = i+1
        return error


################################################################################
class SOM(Layer):
    '''
    A Self-Organizing Map (SOM).

    A self-organizing map is a type of neural network that is trained via
    unsupervised learning. In particular, the self-organizing map finds the
    neuron closest to an input vector -- this neuron is the winning neuron, and
    it is the answer of the network. Thus, the SOM is usually used for
    classification and pattern recognition.

    The SOM is a single-layer network, so this class subclasses the ``Layer``
    class. But some of the properties of a ``Layer`` object are not available or
    make no sense in this context.
    '''
    def __init__(self, shape, lrule=Competitive):
        '''
        Initializes a self-organizing map.

        A self-organizing map is implemented as a layer of neurons. There is no
        connection among the neurons. The answer to a given input is the neuron
        closer to the given input. ``phi`` (the activation function) ``v`` (the
        activation potential) and ``bias`` are not used.

        :Parameters:
          shape
            Stablishes the size of the SOM. It must be a two-tuple of the
            format ``(m, n)``, where ``m`` is the number of neurons in the
            layer, and ``n`` is the number of inputs of each neuron. The neurons
            in the layer all have the same number of inputs.
          lrule
            The learning rule used. Only ``SOMLearning`` objects (instances of
            the class or of the subclasses) are allowed. Defaults to
            ``Competitive``. Check the ``lrules`` documentation for more
            information.
        '''
        Layer.__init__(self, shape, phi=None, bias=False)
        self.__lrule = lrule
        self.__y = None
        self.__phi = None
        if isinstance(lrule, SOMLearning):
            self.__lrule = lrule
        else:
            try:
                issubclass(lrule, SOMLearning)
                self.__lrule = lrule()
            except TypeError:
                raise ValueError, 'uncompatible learning rule'


    def __gety(self):
        if self.__y is None:
            raise ValueError, "activation unavailable"
        else:
            return self.__y
    y = property(__gety, None)
    '''The winning neuron for a given input, the answer of the network. This
    property is available only after the network is fed some input.'''


    def __call__(self, x):
        '''
        The response of the network to a given input.

        The ``__call__`` interface should be called if the answer of the neuron
        network to a given input vector ``x`` is desired. *This method has
        collateral effects*, so beware. After the calling of this method, the
        ``y`` property is set with the activation potential and the answer of
        the neurons, respectivelly.

        :Parameters:
          x
            The input vector to the network.

        :Returns:
          The winning neuron.
        '''
        x = reshape(x, (1, self.inputs))
        dist = sqrt(sum((x - self.weights)**2, axis=1))
        self.__y = argmin(dist)
        return self.y


    def learn(self, x):
        '''
        Applies one example of the training set to the network.

        Using this method, one iteration of the learning procedure is made with
        the neurons of this network. This method presents one example (not
        necessarilly of a training set) and applies the learning rule over the
        network. The learning rule is defined in the initialization of the
        network, and some are implemented on the ``lrules`` method. New methods
        can be created, consult the ``lrules`` documentation but, for
        ``SOM`` instances, only ``SOMLearning`` learning is allowed.

        Also, notice that *this method only applies the learning method!* The
        network should be fed with the same input vector before trying to learn
        anything first. Consult the ``feed`` and ``train`` methods below for
        more ways to train a network.

        :Parameters:
          x
            Input vector of the example. It should be a column vector of the
            correct dimension, that is, the number of input neurons.

        :Returns:
          The error obtained by the network.
        '''
        self.__lrule(self, x)
        return sum(abs(x - self.y))


    def feed(self, x):
        '''
        Feed the network and applies one example of the training set to the
        network.

        Using this method, one iteration of the learning procedure is made with
        the neurons of this network. This method presents one example (not
        necessarilly of a training set) and applies the learning rule over the
        network. The learning rule is defined in the initialization of the
        network, and some are implemented on the ``lrules`` method. New methods
        can be created, consult the ``lrules`` documentation but, for
        ``SOM`` instances, only ``SOMLearning`` learning is allowed.

        Also, notice that *this method feeds the network* before applying the
        learning rule. Feeding the network has collateral effects, and some
        properties change when this happens. Namely, the ``y`` property is set.
        Please consult the ``__call__`` interface.

        :Parameters:
          x
            Input vector of the example. It should be a column vector of the
            correct dimension, that is, the number of input neurons.

        :Returns:
          The error obtained by the network.
        '''
        self(x)
        return self.learn(x)


    def train(self, train_set, imax=2000, emax=1e-5, randomize=False):
        '''
        Presents a training set to the network.

        This method automatizes the training of the network. Given a training
        set, the examples are shown to the network (possibly in a randomized
        way). A maximum number of iterations or a maximum admitted error should
        be given as a stop condition.

        :Parameters:
          train_set
            The training set is a list of examples. It can have any size and can
            contain repeated examples. In fact, the definition of the training
            set is open. Each element of the training set, however, should be a
            input vector of the correct dimensions, See the ``learn`` and
            ``feed`` for more information.
          imax
            The maximum number of iterations. Examples from the training set
            will be presented to the network while this limit is not reached.
            Defaults to 2000.
          emax
            The maximum admitted error. Examples from the training set will be
            presented to the network until the error obtained is lower than this
            limit. Defaults to 1e-5.
          randomize
            If this is ``True``, then the examples are shown in a randomized
            order. If ``False``, then the examples are shown in the same order
            that they appear in the ``train_set`` list. Defaults to ``False``.
        '''
        i = 0
        error = 1
        s = len(train_set)
        while i<imax and error>emax:
            if randomize:
                x = random.choice(train_set)
            else:
                x = train_set[i%s]
            error = self.feed(x)
            i = i+1
        return error


################################################################################
# Test
if __name__ == "__main__":
    pass