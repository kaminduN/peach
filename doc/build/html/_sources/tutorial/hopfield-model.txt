Hopfield Model
==============

A Hopfield network is a one-layer recurrent network, where the output of every
neuron is connected to the inputs of every neuron, except for itself. It was
discovered by John Hopfield in 1982, and was responsible for the ressurgence in
the interest in neural networks, as they weren't receiving much attention at the
time.

The Hopfield network functions as an autoassociative memory, that is, a memory
that is accessed by its contents. While it may seem strange to access a memory
by content (after all, if you have the content, you shouldn't need to access it
from anywhere), its power reside in the fact that a noisy version of the memory
can be used as a starting point, and, ideally, the non-noisy version is
recovered. This has many applications, for example, in pattern recognition.

In this tutorial, we will create a Hopfield network that will store patterns
representing the five vowels, and will use it to recover from a noisy version of
the letter "A". The letters are represented as matrices of pixels such as in the
picture below, but we change blank pixels for -1, and black pixels for 1:

.. image:: figs/hopfield-patterns.png
   :align: center

We start, as always, by importing ``numpy`` and ``peach``. We will need the
``shuffle`` function from the ``random`` module also to generate the noisy
pattern::

  from numpy import *
  import peach as p
  from random import shuffle

Then we must create the training set. A pattern is an array of any size and any
dimensions, but it will be internally converted to a vector column. Usually, it
consists of -1's and 1's to represent unset and set pixels, although there are
different ways to set the network. We will use the standard way, though. It
might be difficult to distinguish the patterns, but with a little effor they can
be seen (you can make a quick script to convert them to a form better for
visualization, if you like)::

  training_set = [
    array([ -1, -1,  1, -1, -1,     # A
            -1,  1, -1,  1, -1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1,  1,  1,  1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1 ]),
    array([  1,  1,  1,  1,  1,     # E
             1, -1, -1, -1, -1,
             1, -1, -1, -1, -1,
             1,  1,  1,  1, -1,
             1, -1, -1, -1, -1,
             1, -1, -1, -1, -1,
             1,  1,  1,  1,  1 ]),
    array([ -1,  1,  1,  1, -1,     # I
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1,  1,  1,  1, -1 ]),
    array([ -1, -1,  1, -1, -1,     # O
            -1,  1, -1,  1, -1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
            -1,  1, -1,  1, -1,
            -1, -1,  1, -1, -1 ]),
    array([  1, -1, -1, -1,  1,     # U
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
             1, -1, -1, -1,  1,
            -1,  1,  1,  1, -1 ])
  ]

Next, we will prepare a test pattern by adding some noise to one of the patterns
in the training set. We choose the letter ``A``, but you could use any other for
tests. In this piece of the script, we shuffle a ``range`` list and take the
first values. We use these as indices and invert the pixel given by them. This
will guarantee that no pixel is inverted twice::

  x = array(training_set[0])
  n = len(x)
  noise_position = range(n)
  shuffle(noise_position)
  for k in noise_position[:8]:   # We invert as much as 8 points in the pattern
      x[k] = -x[k]
  x = x.reshape((n, 1))

Finally, we need to create the network. To create a Hopfield network, just
instance the ``Hopfield`` class using as a parameter the number of components on
each pattern. You can also use a different activation function, if you need. A
standard call to the ``Hopfield`` class will be like::

  nn = p.Hopfield(size, phi=Signum)

Notice that the ``Signum`` is the default activation function. In this tutorial,
we create the Hopfield network by issuing::

  nn = Hopfield(n)

(Remember that ``n`` was set in the previous step: it is the size of the
patterns). Next, we train the network. To do that, we just present the training
set to it, using the ``train`` method::

  nn.train(training_set)

The network is here ready to converge a pattern. We will do that step-by-step,
but a single call to the network (the ``__call__`` interface) will converge it
automatically. Here, to do it step-by-step, we use the ``step`` method. It takes
as argument a state, computes the next state based on the standard Hopfield
algorithm, and returns the updated state. Here how we do that::

  i = 0
  while i < 100:
      x = nn.step(x)
      # show(x)
      i = i + 1

This loop doesn't do anything interesting, and you will notice that the stop
criterion is not very good (100 iterations). But it is enough to show how the
network performs. If you want to see the results of every step, just implement
the ``show`` function to print the pattern in a pleasant way. Notice also that
the ``__call__`` interface uses better stop criteria to guarantee the
convergence. The figure below shows the initial and final state of one execution
of the algorithm. If you don't get the same results, remember that the Hopfield
network is stochastic (*ie*, based on a random algorithm) and sometimes arrive
in different results.

.. image:: figs/hopfield-final-state.png
   :align: center

