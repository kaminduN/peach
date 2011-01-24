Dealing with Binary Objective Functions
=======================================

There are a number of algorithms implemented in Peach that works with binary
optimization, such as the binary simulated annealing and genetic algorithms.
These algorithms, instead of dealing with arrays of floating point variables,
deal with arrays of bits, that are converted to values of whatever type.

Usually, Peach deals with them in a very graceful way -- by using the ``struct``
module, that is standard in any Python distribution, Peach converts arrays of
bits before passing them to your objective function. Thus, you program any
function in the way you expect: by operating over numbers. Binary algorithms
always accept a ``format`` parameter to make the conversion. Please, consult the
``struct`` and the respective algorithms documentation for more information.

Alas, not all of our problems in life will be so well behaved that standard
functions are enough. Sometimes, you need to deal with a different pattern of
values, and the ``struct`` module won't help you. For situations like this,
Peach can also pass to the objective function an array of bits, without any
conversion. In this case, your function is responsible for decoding the
information in the bitstream and compute a real number that is the value of the
objective function.

In this tutorial, we show how to deal with that. Besides using the ``numpy`` and
``peach`` modules, we will need the ``bitarray`` module. Unfortunatelly, it is
not part of the standard Python distribution, so you will have to install it
separatelly. Also, consult the documentation on the module. Anyway, it is very
easy to work with ``bitarray``'s. They work exactly as an array, except for the
fact that each element is a bit, instead of an integer or a float.

Here, we will implement the simplified Rosenbrock function that will work in
pretty much the same way as before, except that here it will receive two
fields of 12 bits, each representing an integer from 0 to 4096. We divide these
integers by 2048 to represent numbers from 0. to 2. This number representation
scheme is called *fixed-point* -- there are lots of ways to represent numbers,
and notice that ours does not takes signs into account! Also, notice that,
unfortunatelly, there is no way to extract these numbers using ``struct``, so we
have to invent our own way.

We start by importing the modules. To define the function, only the ``numpy``
and ``bitarray`` modules are needed, so we will omit the other imports::

    import numpy
    import bitarray

Now, we must define the function. The algorithm -- whatever it is -- will pass
a bitarray as a single object to your function. Treat this as an array. To
extract the first 12 bits, just issue ``x[:12]``, and to extract the last 12
bits, use ``x[12:]``. Use the result as you like. It can be useful to create a
function just to do the conversion::

    def convert(x):
        return numpy.sum(2**numpy.arange(11, -1, -1) * x) / 2048.

What this function does is to create an array with the values of powers of 2,
multiply it by the bits, sum it all and divide by 2048. Now, we define our
objective function. It will receive the bitarray, separate the two numbers,
convert them and apply to the Rosenbrock function::

    def f(b):
        x = convert(b[:12])
        y = convert(b[12:])
        return (1.-x)**2. + (y-x*x)**2.

And that's it. You can pass it to any algorithm that does binary optimization,
such as the ``BinarySA`` or ``GeneticAlgorithm``. Please, check the
documentation on these classes for more information. Below is a table with some
values of the conversion:

===========================  ======  ======  =======
            bits                  x       y  f(x, y)
===========================  ======  ======  =======
000000000000,  000000000000  0.0000  0.0000   1.0000
000000000001,  000000000001  0.0005  0.0005   0.9990
000001000000,  000001000000  0.0312  0.0312   0.9394
000001000000,  000000000001  0.0312  0.0005   0.9385
000000000001,  000001000000  0.0005  0.0312   1.0000
001010110010,  100101110101  0.3369  1.1821   1.5816
010101101010,  110101010101  0.6768  1.6665   1.5650 
101010101010,  101010101010  1.3330  1.3330   0.3079
111111111111,  111111111111  1.9995  1.9995   4.9932
010000000000,  010000000000  0.5000  0.5000   0.3125
100000000000,  100000000000  1.0000  1.0000   0.0000
===========================  ======  ======  =======
