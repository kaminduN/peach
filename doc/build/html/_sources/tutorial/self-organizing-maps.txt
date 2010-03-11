Extended Example on Self-Organizing Maps
========================================

A self-organizing map is a neural network that can be used to solve a number of
problems including clustering and classification. The goal of this tutorial is
to show how the synaptic weights of a SOM converge to the mass center of a cloud
of points, thus allowing for clustering and classification of patterns.

We will create clouds of points around five centers by adding a random number
with gaussian distribution (zero average and small variance) to the vectors
representing the center. Then we will create a SOM that will converge five
neurons to the clouds. While this is a graphical tutorial, we won't show the
commands to create the plot.

We start, as always, by importing ``numpy`` and ``peach``. We will need the
``random`` module also to generate the training set::

  from numpy import *
  import random
  import peach as p

The most important part on this tutorial is the creation of the training set. We
will use five centers, positioned at the coordinates ``(1, 0)``, ``(1, 1)``,
``(0, 1)``, ``(-1, 1)``, ``(-1, 0)``. Our training set will have 300 points.
Each point of the training set is found by adding a gaussian generated number to
each coordinate of the corresponding center. The following sequence of commands
does the job::

  train_size = 300
  centers = [ array([ 1.0, 0.0 ], dtype=float),
              array([ 1.0, 1.0 ], dtype=float),
              array([ 0.0, 1.0 ], dtype=float),
              array([-1.0, 1.0 ], dtype=float),
              array([-1.0, 0.0 ], dtype=float) ]
  xs = [ ]
  for i in range(train_size):
      x1 = random.gauss(0.0, 0.1)
      x2 = random.gauss(0.0, 0.1)
      xs.append(centers[i%5] + array([ x1, x2 ], dtype=float))

Since we are working on the plane, each example and each neuron will have two
coordinates. We will use five neurons (since we have five centers). The
self-organizing map is created by the line below. Our goal is to show how the
weights converge to the mass center of the point clouds, so we initialize the
weights near the center, but in a way that we are sure that will converge to
the centers. It is not a realistic condition, but in the tutorial we want to
inspect the behaviour of the ``SOM``, so we will use it here::

  nn = p.SOM((5, 2))
  for i in range(5):
      nn.weights[i, 0] = 0.3 * cos(i*pi/4)
      nn.weights[i, 1] = 0.3 * sin(i*pi/4)

We will use lists to track the variation of each neuron::

  wlog = [ [ nn.weights[0] ],
           [ nn.weights[1] ],
           [ nn.weights[2] ],
           [ nn.weights[3] ],
           [ nn.weights[4] ] ]

Here we feed and update the network. We could use the ``train`` method to
present the whole set, but we want to track the weights. So, we feed the network
using the ``__call__()`` interface, using each example as the argument; and then
we instruct the network to learn that example. We track the values of the
winning neuron to show it later::
    
  for x in xs:
    y = nn(x)
    nn.learn(x)
    wlog[y].append(array(nn.weights[y]))

The figure shows the result. In black dots we see the training set, and in
crosses of different colors we see the synaptic weights approaching the center
of the clouds.

.. image:: figs/self-organizing-maps.png
   :align: center


