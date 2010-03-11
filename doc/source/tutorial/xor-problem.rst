The XOR Problem
===============

The exclusive-or (XOR) is a well known problem with neural networks. In 1969,
Marvin Minsky published a book, named *Perceptrons*, in which he showed that a
single neuron cannot map the exclusive-or between two inputs. The pessimistic
nature of the books conclusion resulted in a set-back in the research of neural
networks that lasted until the backpropagation was created. Every book on neural
network has a description and the proof of the problem.

But we can solve this problem if we are allowing a network with more than one
neuron and more than one layer, and using an appropriate learning rule to make
the synaptic weights of the neurons converge to values that allow the XOR
operation to be mapped. We will do this in this tutorial.

It can be done with a two-layer biased feed forward network with two inputs, two
neurons in the hidden layer and one neuron in the output layer. The activation
function should be sigmoidal, and the learning rule the backpropagation
algorithm. As before, we will assume that the ``numpy`` and ``peach`` modules
were imported in the command line. We create the network::

   >>> nn = FeedForward((2, 2, 1), Sigmoid, BackPropagation(0.2), True)

Instead of presenting every single example, we will create a training set and
present the training set to the network. The training set is easy: it should be
that truth table of the exclusive-or operation::

   >>> train_set = [ ( array(( 0.0, 0.0)), 0.0 ),
                     ( array(( 0.0, 1.0)), 1.0 ),
                     ( array(( 1.0, 0.0)), 1.0 ),
                     ( array(( 1.0, 1.0)), 0.0 ) ]

A training set is a list of examples. Every example is a tuple with two
elements: the first one is the input vector, and the second is the desired
response. We present a complete training set to the network using the ``train``
method::

   >>> nn.train(train_set)

This will iterate over the training set. The complete signature of this method
is::

  train(train_set, imax=2000, emax=1e-05, randomize=False)

Here, ``train_set`` is the list of examples as described above; ``imax`` is the
maximum number of iterations over the training set; ``emax`` is the maximum
error allowed. The iteration over the training set will end when any of these
conditions are met. If the ``randomize`` is ``False``, the iteration over the
training set is sequential, else every example is randomly chosen.

In this tutorial, after presenting the network with the examples, we can inspect
the results with a simple loop::

   >>> for x, _ in train_set:
   ...   print x, ' => ', nn(x)
   ...
   [ 0.  0.]  =>  [[ 0.04868284]]
   [ 0.  1.]  =>  [[ 0.94078034]]
   [ 1.  0.]  =>  [[ 0.9422161]]
   [ 1.  1.]  =>  [[ 0.07817926]]

Notice that we don't get exactly 0 or 1 as the response of the network. This
happens because we are using a sigmoid as activation function, and it reaches
these values only when input is infinity. But notice that the output is very
near 0 when the answer should be 0, and very near 1 when the answer should be 1.
