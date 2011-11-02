################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/nn.py
# Basic topologies of neural networks
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Radial Basis Function Networks

This sub-package implements the basic behaviour of radial basis function
networks. This is a two-layer neural network that works as a universal function
approximator. The activation functions of the first layer are radial basis
functions (RBFs), that are symmetric around the origin, that is, the value of
this kind of function depends only on the distance of the evaluated point to the
origin. The second layer has only one neuron with linear activation, that is, it
only combines the inputs of the first layer.

The training of this kind of network, while it can be done using a traditional
optimization technique such as gradient descent, is usually made in two steps.
In the first step, the position of the centers and the width of the RBFs are
computed. In the second step, the weights of the second layer are adapted. In
this module, the RBFN architecture is implemented, allowing training of the
second layer. Centers must be supplied, but they can be easily found from the
training set using algorithms such as K-Means (the one traditionally used),
SOMs or Fuzzy C-Means.
"""

################################################################################
from numpy import array, amax, sum
from random import choice
from nnet import *

################################################################################
# Classes
class RBFN(object):

    def __init__(self, c, phi=Gaussian, phi2=Linear):
        '''
        Initializes the radial basis function network.

        A radial basis function is implemented as two layers of neurons, the
        first one with the RBFs, the second one a linear combinator.

        :Parameters:
          c
            Two-dimensional array containing the centers of the radial basis
            functions, where each line is a vector with the components of the
            center. Thus, the number of lines in this array is the number of
            centers of the network.
          phi
            The radial basis function to be used in the first layer. Defaults to
            the gaussian.
          phi2
            The activation function of the second layer. If the network is being
            used to approximate functions, this should be Linear. Since this is
            the most commom situation, it is the default value. In occasions,
            this can be made (say) a sigmoid, for pattern recognition.
        '''
        self.__c = array(c)
        self.__n = len(self.__c)
        wmax = 0.
        for ci in self.__c:
            w = amax(sum((ci - self.__c)**2, axis=1))
            if w > wmax:
                wmax = w
        self.__w = array([ sqrt(wmax) ]*self.__n) / (self.__n - 1)
        self.phi = phi
        self.__l = FeedForward((self.__n, 1), phi=phi2, lrule=BackPropagation)


    def __getwidth(self):
        return self.__w
    def __setwidth(self, w):
        try:
            if len(w) != len(self.__c):
                raise AttributeError('Width array must have the same number of componets as the number of centers')
            else:
                self.__w = array(w)
        except TypeError:
            self.__w = array([ w ]*self.__n)
    width = property(__getwidth, __setwidth)
    '''The computed width of the RBFs. This property can be read and written. If
    a single value is written, then it is used for every center. If a vector of
    values is supplied, then it must be one for each center.'''


    def __getweights(self):
        return self.__l[0].weights
    def __setweights(self, w):
        self.__l[0].weights = w
    weights = property(__getweights, __setweights)
    '''A ``numpy`` array containing the synaptic weights of the second layer of
    the network. It is writable, but the new weight array must be the same shape
    of the neuron, or an exception is raised.'''


    def __gety(self):
        return self.__l.y
    y = property(__gety, None)
    '''The activation value for the second layer of the network, ie., the answer
    of the network. This property is available only after the network is fed
    some input.'''


    def __getphi(self):
        return self.__phi
    def __setphi(self, phi):
        try:
            issubclass(phi, RadialBasis)
            phi = phi()
        except TypeError:
            pass
        if isinstance(phi, RadialBasis):
            self.__phi = phi
        else:
            self.__phi = RadialBasis(phi)
    phi = property(__getphi, __setphi)
    '''The radial basis function. It can be set with a ``RadialBasis`` instance
    or a standard Python function. If a standard function is given, it must
    receive a real value and return a real value that is the activation value of
    the neuron. In that case, it is adjusted to work accordingly with the
    internals of the layer.'''


    def __getphi2(self):
        return self.__phi2
    def __setphi2(self, phi):
        try:
            issubclass(phi, Activation)
            phi = phi()
        except TypeError:
            pass
        if isinstance(phi, Activation):
            self.__phi2 = phi
        else:
            self.__phi2 = Activation(phi)
    phi2 = property(__getphi, __setphi)
    '''The activation function for the second layer. It can be set with an
    ``Activation`` instance or a standard Python function. If a standard
    function is given, it must receive a real value and return a real value that
    is the activation value of the neuron. In that case, it is adjusted to work
    accordingly with the internals of the layer.'''


    def __call__(self, x):
        '''
        Feeds the network and return the result.

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
        x = array([ self.__phi((x-ci)/wi) for ci, wi in zip(self.__c, self.__w) ])
        return self.__l(x)


    def learn(self, x, d):
        '''
        Applies one example of the training set to the network.

        Using this method, one iteration of the learning procedure is executed
        for the second layer of the network. This method presents one example
        (not necessarilly from a training set) and applies the learning rule
        over the layer. The learning rule is defined in the initialization of
        the network, and some are implemented on the ``lrules`` method. New
        methods can be created, consult the ``lrules`` documentation but, for
        the second layer of a ``RBFN'' instance, only ``FFLearning`` learning is
        allowed.

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
        x = array([ self.__phi((x-ci)/wi) for ci, wi in zip(self.__c, self.__w) ])
        return self.__l.learn(x, d)


    def feed(self, x, d):
        '''
        Feed the network and applies one example of the training set to the
        network. This adapts only the synaptic weights in the second layer of
        the RBFN.

        Using this method, one iteration of the learning procedure is made with
        the neurons of this network. This method presents one example (not
        necessarilly from a training set) and applies the learning rule over the
        network. The learning rule is defined in the initialization of the
        network, and some are implemented on the ``lrules`` method. New methods
        can be created, consult the ``lrules`` documentation but, for the second
        layer of a ``RBFN``, only ``FFLearning`` learning is allowed.

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
        x = array([ self.__phi((x-ci)/wi) for ci, wi in zip(self.__c, self.__w) ])
        return self.__l.feed(x, d)


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
