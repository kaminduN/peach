Basic Neural Network Manipulation
=================================

The aim of this tutorial is to show how to deal with simple neural networks. We
will create a simple multi-layer perceptron (MLP), set its synaptic weights and
show the network an example. To understand this tutorial, you should have some
knowledge of how neural networks, in special MLPs. Please, consult a good text
book on the subject.

We will create the neural network in the figure below. As we can see, it is a
neural network with two inputs, one hidden layer with two neurons and one output
layer with one layer. We will use a sigmoidal function as its activation
function and backpropagation as its learning algorithm. These are, in general,
the default choices.

.. image:: figs/basic-neural-network.png
   :align: center

We will assume that we are using the Python command line to see what we are
doing. So, the first thing we need to do is to import the Peach library. We do
this with the command:

  >>> from peach import *

Since we are using the command line, there is no problem in using this, but
remember that, usually, it is recommended that a module is not imported in the
main namespace.

The network we are trying to build has 2 input neurons, 2 hidden neurons and 1
input neuron. We create a new MLP by instantiating the ``FeedForward`` class, as
below:

  >>> nn = FeedForward((2, 2, 1), Sigmoid, BackPropagation)

In this command line, ``(2, 2, 1)`` are the dimensions of each layer, in the
sequence above. The synaptic weights are randomly created and stored in a
NumPy array. You can create as many layers as you want, by passing a tuple with
the number of neurons in each layer -- just remember that the first number is
the input layer, and the last number is the output layer. The Peach module takes
care of the dimension coherence.

We indicate the ``Sigmoid`` activation function and the ``BackPropagation``
learning method. There are other :ref:`activation functions` and :ref:`learning
rules` available. One interesting thing is that these parameters are, actually,
classes that are internally instantiated to work with the neural network. But,
by instantiating them yourself, you can configure their behavior.

The complete list of parameters in the class instantiation is::

   FeedForward(layers, phi, lrule, bias)

In this, ``layers`` is a tuple of numbers that indicate how many neurons in each
layer, where the first is the number of neurons in the input layer, the last is
the number of neurons in the output layer and the others are the number of
neurons in the hidden layers. ``phi`` is the activation function, it defaults to
the ``Linear`` function, that is, the identity function. ``lrule`` is the
learning algorithm, it defaults to the standard ``BackPropagation``. ``bias`` is
a boolean value that indicates, if ``True``, that the neurons in the network are
biased. It defaults to ``False``, that is, non-biased neurons.

There are a number of properties that we can consult to inspect the neural
network. Some of these are given below:

  ``nlayers``
    The number of layers in the network.

  ``bias``
    A tuple containing the bias of each layer.

  ``y``
    This property is the activation of the network. It corresponds to an array
    with the outputs of every neuron in the last layer. This property can only
    be used, however, after the network is fed some input vector.

  ``phi``
    Activation functions for every layer in the network.

  ``[n]``
    The ``[]`` operator can be used to access a specific ``Layer`` of the
    network. There are some properties of the layers that can be very useful.

A ``Layer`` is an object that represents exactly that: a layer of neurons. It
has some properties that are very useful, some of them are listed below:

  ``size``
    The number of neurons on the layer.

  ``inputs``
    The number of inputs in each neuron.

  ``shape``
    A tuple in the form ``(size, inputs)`` with the two information above.

  ``weights``
    A ``numpy`` array containing the synaptic weights of the neurons on the
    layer. Each line of this array is the weight vector of the corresponding
    neuron.

  ``phi``
    The activation function appliedd to every neuron of the layer.

  ``v``
    A vector containing the activation potential of the neurons of the layer.
    This property is only available and can only be used after the layer was fed
    an input, and will give the activation potential to the last input.

  ``y``
    The output vector of the neurons of the layer. This property is only
    available and can only be used after the layer was fed an input, and
    will give the output to the last input.

When the neural network is created, a randomized array of synaptic weights is
created for every layer. We can inspect and modify those using the ``weights``
property of a ``Layer``. The synaptic weights are ``numpy`` arrays of floating
point numbers. Let's give our network an initial condition::

  >>> nn[0].weights = array([[  0.5,  0.5 ],
                             [ -0.5, -0.5 ]])

  >>> nn[1].weights = array([ 0.25, -0.25 ])

``nn[0]`` is the first layer. Notice that the input layer doesn't have synaptic
weights, so they are not considered here -- in other words, ``nn[0]`` is the
first hidden layer. ``nn[1]`` is the second layer which, in this case, is the
output layers. It could be accessed using ``nn[-1]``, because a ``FeedForward``
network is actually a list of ``Layers``.

We must feed the network to get some results. Actually, we will present an
example to the network and tell it to learn the example. We create the input
vector and the desired output by the following commands::

  >>> x = array([ 0.8, 0.2 ])
  >>> d = 0.9

Let's see what this neural network outputs with this input. We feed the neural
network and receive its output by calling it as a function::

  >>> nn(x)
  array([[ 0.51530264]])

We tell the network to learn the example by showing it to the ``feed()`` method,
as shown below::

  >>> nn.feed(x, 0.9)
  0.3846973641912852

This method outputs the error obtained with the example. Let's inspect the
synaptic weights now and see how they were modified:

  >>> nn[0].weights
  array([[ 0.5002258 ,  0.50005645],
         [-0.5002258 , -0.50005645]])
  >>> nn[1].weights
  array([[ 0.25299043, -0.24818621]])

We can see that the error, for this example, is now a little smaller::

  >>> 0.9 - nn(x)
  array([[ 0.38405579]])

Notice that the answer of the neural network is a column-vector of the outputs
of the neurons in the last layer. That is why the last command resulted in an
array.