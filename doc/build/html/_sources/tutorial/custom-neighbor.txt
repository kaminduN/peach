Using Custom Neighbor Functions
===============================

In this tutorial, we will show how to use custom neighbors in continuous
simulated annealing optimization algorithms. Peach is built in a way that is
very configurable. The aim of this is to allow you to change small parts of the
algorithms and test what does better for your problem. In this case, the
continuous simulated annealing algorithm is implemented, and you can change the
function that computes the neighbor of the estimate, without having to
reimplement the whole algorithm itself. Most of the algorithms in Peach are
implemented this way, so look in the documentation or other tutorials for more
information on what you need.

In the case of simulated annealing, what we need is a simple function that,
given an estimate, computes another estimate, close enough to the one given, and
return it to the object that implements the algorithm. We don't need to
implement nothing else -- the annealing is already there.

The default neighbor function in the standard simulated annealing is computed by
randomly choosing with a gaussian distribution around the present estimate. But
suppose we don't want a gaussian distribution, but a uniform one, distributed
from -2 to 2. Peach already implements a uniformly distributed neighbor (check
the ``UniformNeighbor`` class in the reference), but it is distributed from -1
to 1. So, let's see how it can be done.

There are three ways to do this. The simplest way is instantiating the
``UniformNeighbor`` passing the lower and upper limits in the instantiation. You
can pass this directly in the instantiation of the algorithm, but we will see
outside this scope. Of course, as always, we won't suppose you're working on the
command line, but these commands will work there too. The first thing we need to
do is import the ``numpy`` and ``peach`` modules, and also the ``uniform``
function from the ``numpy.random`` module::

    import numpy
    from numpy.random import uniform
    import peach

To create your custom function, just issue::

    CustomNeighbor1 = peach.UniformNeighbor(-2, 2)

and that does it. If what you need is already implemented, but you need
different parameters, this is the best way to deal with it.

If you need a different function, however, that behaves in other way, you should
define your own function. Just Define a function that receives a one-dimensional
array and returns another array with the same length, and you're done. If you
know exactly what are the dimensions of the array your objective function is
working with (you probably know that), you can define your function to work
exclusively with arrays of that size. But it is always a good thing to be able
to compute with any number of variables. To transform a simple function in a
neighbor function, instantiate ``ContinuousNeighbor``. Here is our definition::

    def uniform_neighbor(x, a=-1, b=1):
        return x + uniform(a, b, len(x))

    CustomNeighbor2 = peach.ContinuousNeighbor(uniform_neighbor)

The only parameter that the simulated annealing will be passing to your function
is the ``x`` array. The other parameters are there just to be clear -- after
all, the uniform distribution needs a lower and an upper limit, here represented
by the parameters ``a`` and ``b`` respectively. You could put these limits
directly in the ``uniform`` function call, but this way might be more readable.
Notice that the last step is not really necessary. If you just instantiate the
algorithm and pass ``uniform_neighbor`` as the ``neighbor`` parameter, the
conversion is done internally for you. Both lines below do an equivalent job::

    csa = peach.ContinuousSA(f, x0, neighbor=CustomNeighbor2)

or::

    csa = peach.ContinuousSA(f, x0, neighbor=uniform_neighbor)

Here ``f`` is the objective function, and ``x0`` is the first estimate. There
are other parameters available to the simulated annealing algorithm, but they
are not covered here.

The last, more flexible but a little more complicated way, is to create your own
class derived from ``ContinuousNeighbor``. You will have to implement the
``__init__`` and ``__call__`` methods. In the ``__init__`` method you pass any
configuration parameters that are needed in your function, and the ``__call__``
method is the function call -- it should receive a one-dimensional array of any
length and return another one-dimensional array with the same length containing
the coordinates of the neighbor. Here we do that with the same uniformly
distributed neighbor::

    class CustomNeighbor3(peach.ContinuousNeighbor):
        '''
        Don't forget to document your code!
        '''
        def __init__(self, a=-2, b=2):
            '''
            Always provide sensible defaults to your parameters. Some of the
            classes in Peach do not expect further parameters, so if you try to
            instantiate classes without them, an exception will be raised. Here,
            the parameters are the limits of the distribution that we want.
            '''
            peach.ContinuousNeighbor.__init__(self)
            self.a = a
            self.b = b

        def __call__(self, x):
            '''
            The __call__ interface should receive an array of scalars and return
            a vector of the same length. If needed, using the ``numpy``
            functions will certainly help you deal with arrays of any length.
            '''
            return x + uniform(a, b, len(x))

The class here created can be used at any place where a ``ContinuousNeighbor``
is needed, such as in the creation of the algorithm. We could use as any one of
the lines below::

    csa = peach.ContinuousSA(f, x0, neighbor=CustomNeighbor3)

or::

    csa = peach.ContinuousSA(f, x0, neighbor=CustomNeighbor3())
