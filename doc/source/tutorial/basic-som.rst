Basic Self-Organizing Maps
==========================

A self-organizing map is a neural network that can be used to solve a number of
problems including clustering and classification. The goal of this tutorial is
to show how to use Peach to deal with self-organizing maps. We start by creating
a small map with only five neurons, with two inputs each. To inspect the
behaviour of the map, we will set the weights to known values.

We start, as always, by importing ``numpy`` and ``peach``. We will assume that
we are working in the command line to check what we are doing::

  >>> from numpy import *
  >>> from peach import *

Since we are using the command line, there is no problem in using this, but
remember that, usually, it is recommended that a module is not imported in the
main namespace.

To create a self-organizing map, we instantiate the ``SOM`` class. Its
initializer receives as arguments the dimensions of the network and the learning
rule. The shape of the network is given as a tuple ``(M, N)``, meaning a map
with ``M`` neurons with ``N`` inputs each. We won't worry about the learning
rule at this moment::

  >>> nn = SOM((5, 2))

The complete list of parameters in the class instantiation is::

  SOM(shape, lrule)

Here, the ``shape`` are the dimensions as given above. ``lrule`` can be one of
``WinnerTakesAll``, ``Competitive`` or ``Cooperative``. Please, check the
learning rules documentation for more information. The default value for the
learning rule is the ``Competitive`` method.

There are a number of properties that we can consult to inspect the neural
network. Some of these are given below:

  ``y``
    This property is the activation of the network. It corresponds to an array
    with the outputs of every neuron in the map. This property can only be used,
    however, after the network is fed some input vector.

  ``[n]``
    The ``[]`` operator can be used to access a specific neuron of the map.

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

We want to see how the map behaves, so we feed the network with one vector.
We create it as the next step. Notice that it is a simple ``numpy`` array::

  >>> x = array([ 0.0, -0.5 ], dtype=float)

It is expected that the winning neuron is the one represented in the third line.
We will check to see if that's the case. To activate the network and see which
of the neurons is the winner, we use the ``__call__`` interface, with the input
vector as an argument. This is the main way to use the network::

  >>> y = nn(x)

Just feeding the network is not enough to make it learn. The learning process is
separated from the feeding to allow for stability, since we don't want to modify
the network once it is in production. The ``SOM`` retains information about the
winning neuron for the last input vector presented to the network, so all we
need to do is to pass the input vector to update it. Since the learning is
``Competitive``, only the winning neuron will be updated::

  >>> nn.learn(x)

We can check the results by inspecting the map itself::

  >>> nn.y
  2
  >>> nn.weights
  [[ 0.5    0.   ]
   [-0.5    0.   ]
   [ 0.    -0.025]
   [ 0.5    0.5  ]
   [-0.5    0.5  ]]