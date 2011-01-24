Interpolation of a Number Sequence
======================================

A neural network can also be used to interpolate values of a sequence or
function of which little is known. The problem of interpolation is the
following: given a limited set of the values of a function in a given interval,
compute the function for any value of the independent variable in the same
interval. There are a lot of techniques to deal with this problem, and most do
well in a given set of conditions. The aim of this tutorial is to do that with
neural networks.

Typically, the structure used is a double layer neural network. In the first
layer, neurons with sigmoid activation to map the nonlinearities in the
function, and in the second layer a linear activated neuron, to combine the
inputs. This structure is commonly known as *MADALINE* (*Multiple Adaptive
Linear Element*). The goal of this tutorial is to show how to use Peach to do
this.

As always, we first import ``numpy`` for arrays and ``peach`` for the library.
Actually, ``peach`` also the ``numpy`` module, but we want it in a separate
namespace. We will also use the ``random`` module to generate noise::

    from numpy import *
    import peach as p
    from random import randrange

We must create the sequence of samples that we will interpolate. For this
tutorial, we will use twenty samples of a sinus function, evenly spaced in the
interval from :math:`-\pi/2` to :math:`\pi/2`::

    t = linspace(-pi/2., pi/2., 20)
    x = sin(t)

We create the neural network with the command below. It should be a network with
one input neuron (since it is a single variable function), and one output
neuron. The hidden layer must have enough neurons to map the variations in the
function. Since the sinus is a very simple function, and our frequency is low,
10 neurons should be enough. We must make the neurons biased. The reason for
this is that the sigmoids of the first layer must be shifted to the position of
the variation it will map. The second layer does not need to be biased, in
general, but there is no harm in letting it be. And, since we are using
activation functions that are not linear, we must use the backpropagation
learning rule::

    nn = p.FeedForward((1, 10, 1), phi=(p.Sigmoid, p.Identity),
                       lrule=p.BackPropagation(0.05), bias=True)

Recalling -- the tuple gives the number of neurons in the input, hidden and
output layers in this order; ``phi`` gives the activation functions for each
layer; ``lrule`` is the learning rule, which here is the backpropagation with a
learning rate of 0.05; ``bias`` indicates that our neurons are biased.

The learning loop will be executed at most 5000 times. Most of the time, this is
an overkill, but given the stochastic nature of the learning, sometimes it is
needed. Anyways, we put a stop trigger -- when the error reaches 1e-5, the
algorithm stops. The training sequence is a list of samples. We could shuffle
the list and present the examples in the same order for many epochs. However, it
can be useful to randomly choose a sample at every step of the training, since
the randomnesse can help the convergence::

    while i < 5000 and error > 1.e-5:

        # Randomly choosing the sample
        index = randrange(20)
        xx = t[index]
        dd = x[index]

        # Here, the network is fed, the error is collected and logged, and the
        # learning process takes place.
        y = nn(xx)[0, 0]
        error = abs(dd - y)
        nn.learn(xx, dd)
        i = i + 1


And in the end of this loop, the network will have converged and the function
can be calculated as correctly as possible for values inside the given interval.
Notice, however, that extrapolating will probably not work well. Using the
``matplotlib`` package we can plot the result of the prediction, the convergence
of the prediction error, and in the second plot, the value of the prediction
coefficients after convergence.

.. image:: figs/interpolation.png
   :align: center
