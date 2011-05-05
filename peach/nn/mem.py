################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/mem.py
# Associative memories and Hopfield models
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Associative memories and Hopfield network model.

This sub-package implements associative memories. In associative memories,
information is recovered by supplying not an exact index (such as in their
usual counterparts), but supplying an index simmilar enough that the information
can be deduced from what is stored in its synaptic weights. There are a number
of different memories of this kind.

The most common type is the Hopfield network. A Hopfield network is a recurrent
self-associative memory. Although there are real-valued versions of the network,
the binary type is more common. In it, patterns are recovered from an initial
estimate through an iterative process.
"""

################################################################################
from numpy import zeros, eye, all
from random import randrange

from peach.nn.base import *
from peach.nn.af import *


################################################################################
# Classes
################################################################################
class Hopfield(Layer):
    '''
    Hopfield neural network model

    A Hopfield network is a recurrent network of one layer of neurons. There
    output of every neuron is conected to the inputs of every other neuron, but
    not to itself. This kind of network is autoassociative, or content-based
    memory. That means that, given a noisy version of a pattern stored in it,
    the network is capable of, through an iterative algorithm, recover the
    original pattern, removing the noise. There is a limit in the quantity of
    patterns that can be stored without causing error, and if a pattern is
    stored, its negated form is also stored.

    This is the binary form of the Hopfield network, which is the most common
    form. It implements a ``Layer`` of neurons, without bias, and with the
    Signum as the activation function.
    '''
    def __init__(self, size, phi=Signum):
        '''
        Initializes the Hopfield network.

        The Hopfield network is implemented as a layer of neurons.

        :Parameters:
          size
            The number of neurons in the network. In a Hopfield network, the
            number of neurons is also the number of inputs in each neuron, and
            the dimensionality of the patterns to be stored and recovered.
          phi
            The activation function. Traditionally, the Hopfield network uses
            the signum function as activation. This is the default value.
        '''
        Layer.__init__(self, (size, 1), phi=Signum, bias=False)
        self.__size = size
        self.__weights = zeros((size, size))


    def __getinputs(self):
        return self.__size
    inputs = property(__getinputs, None)
    '''Number of inputs for each neuron in the layer. For a Hopfield model,
       there are as much inputs as there are neurons. Not writable.'''


    def __getweights(self):
        return self.__weights
    def __setweights(self, m):
        self.__weights = array(reshape(m, self.weights.shape))
    weights = property(__getweights, __setweights)
    '''A ``numpy`` array containing the synaptic weights of the network. Each
    line is the weight vector of a neuron. It is writable, but the new weight
    array must be the same shape of the neuron, or an exception is raised.'''


    def learn(self, x):
        '''
        Applies one example of the training set to the network.

        Training a Hopfield network is not exactly an iterative procedure. The
        network usually stores a small number of patterns, and the learning
        procedure consists only in computing the synaptic weight matrix, which
        can be done in very few steps (in fact, just the number of patterns).
        This method is here for consistency with the rest of the library, but
        it works, anyway.

        :Parameters:
          x
            The pattern to be stored. It must be a vector with the same size as
            the network, or else an exception will be raised. The pattern can be
            of any dimensionality, but it will internally be converted to a
            column vector.
        '''
        n = self.size
        print n, len(x)
        x = array(x).reshape((1, n))
        self.weights = self.weights + 1./float(n) * dot(x.T, x)
        self.weights = self.weights * (1 - eye(n))


    def train(self, train_set):
        '''
        Presents a training set to the network

        This method stores all the patterns of the training set in the weight
        matrix. It calls the ``learn`` method for every pattern in the set.

        :Parameters:
          train_set
            A list containing all the patterns to be stored in the network. Each
            pattern is a vector of any dimensions, which are converted
            internally to a column vector.
        '''
        for x in train_set:
            self.learn(x)


    def step(self, x):
        '''
        Performs a step in the recovering procedure

        The algorithm for recovering the patterns stored in a Hopfield network
        is an iterative algorithm which goes from a starting test pattern (a
        stored pattern with noise) and recovers the noiseless version -- if
        possible. This method takes the test pattern and performs one step of
        the convergence

        :Parameters:
          x
            The noisy pattern.

        :Returns:
          The result of one step of the convergence. This might be the same as
          the input pattern, or the pattern with one component inverted.
        '''
        x = reshape(x, (self.inputs, 1))
        k = randrange(self.size)
        y = self.phi(dot(self.weights[:, k], x)[0])
        if y != 0:
            x[k, 0] = y
        return x


    def __call__(self, x, imax=2000, eqmax=100):
        '''
        Recovers a stored pattern

        The ``__call__`` interface should be called if a memory needs to be
        recovered from the network. Given a noisy pattern ``x``, the algorithm
        will be executed until convergence or a maximum number of iterations
        occur. This method repeatedly calls the ``step`` method until a stop
        condition is reached. The stop condition is the maximum number of
        iterations, or a number of iterations where no changes are found in the
        retrieved pattern.

        :Parameters:
          x
            The noisy pattern vector presented to the network.
          imax
            The maximum number of iterations the algorithm is to be repeated.
            When this number of iterations is reached, the algorithm will stop,
            whether the pattern was found or not. Defaults to 2000.
          eqmax
            The maximum number of iterations the algorithm will be repeated if
            no changes occur in the retrieval of the pattern. At each iteration
            of the algorithm, a component might change. It is considered that,
            if a number of iterations are performed and no changes are found in
            the pattern, then the algorithm converged, and it stops. Defaults to
            100.

        :Returns:
          The vector containing the recovered pattern from the stored memories.
        '''
        i = 0
        eq = 0
        while i < imax and eq < eqmax:
            xnew = self.step(x)
            if any(xnew != x):
                x = xnew
                eq = 0
            i = i + 1
            eq = eq + 1
        return x


################################################################################
# Test
if __name__ == "__main__":
    pass