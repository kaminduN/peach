Norms and Conorms
=================

Fuzzy logic can be seen as an extension of classical logic. Classical sets can
be seen as fuzzy sets with membership values 0 or 1. The same holds true for
standard set operations. But things can be extended in infinite ways, so there
are infinite ways to define fuzzy set intersection, union and complement.

Intersections are defined as a function *t*\ (\ *x*, *y*) taking two arguments and
returning one result, and satisfying the properties:

    - It is comutative, that is: *t*\ (\ *x*, *y*) = *t*\ (\ *y*, *x*)
    - Its neutral element is 1, that is *t*\ (\ *x*, 1) = *x*
    - It has the property *t*\ (\ *x*, 0) = 0
    - It is associative, that is *t*\ (\ *x*, *t*\ (\ *y*, *z*)\ ) =
      *t*\ (\ *t*\ (\ *x*, *y*), *z*)
    - It is monotonic, that is, if *w* < *x* and *z* < *y*, then
      *t*\ (\ *w*, *z*) < *t*\ (\ *x*, *y*)

Functions with these properties are called norms, or *t*-norms. The traditional
function used as intersection operator over fuzzy sets is the minimum function,
that takes two values and returns the smallest of them. This is called the
*Zadeh intersection*, or *Zadeh and*.

Unions are defined as a function *s*\ (\ *x*, *y*) taking two arguments and
returning one result, and satisfying the properties:

    - It is comutative, that is: *s*\ (\ *x*, *y*) = *s*\ (\ *y*, *x*)
    - Its neutral element is 0, that is *s*\ (\ *x*, 0) = *x*
    - It has the property *s*\ (\ *x*, 1) = 1
    - It is associative, that is *s*\ (\ *x*, *s*\ (\ *y*, *z*)\ ) =
      *s*\ (\ *s*\ (\ *x*, *y*), *z*)
    - It is monotonic, that is, if *w* < *x* and *z* < *y*, then
      *s*\ (\ *w*, *z*) < *s*\ (\ *x*, *y*)

Functions with these properties are called conorms, or *s*-norms. The
traditional function used as union operator over fuzzy sets is the maximum
function, that takes two values and returns the largest of them. This is called
the *Zadeh union*, or *Zadeh or*.

Complements are implemented by negations. Negations are functions of one
variable with the following properties:

    - *n*\ (0) = 1
    - *n*\ (1) = 0
    - *n*\ (\ *n*\ (x)) = x
    - if *y* < *x*, then *n*\ (\ *y*) > *n*\ (\ *x*)

The standard negation used in fuzzy logic is the 1-complement, that is 1 - *x*.

In Peach, to set norms, conorms and negations, we use, respectively, the methods
``set_norm``, ``set_conorm`` and ``set_negation``. Notice that those are class
methods, so if you change the norm for one instance of a set, you change for
them all! So, it is better to use the class name to select the methods.

We import ``numpy`` for arrays and ``peach`` for the library::

    import numpy
    from peach import *

First, remember that we must create the sets. A FuzzySet instance is returned
when you apply a membership function over a domain. It is, in fact, a standard
array, but making it a new class allow us to redefine operations. Here we create
the sets::

    x = numpy.linspace(-5.0, 5.0, 500)
    a = Triangle(-3.0, -1.0, 1.0)(x)
    b = Triangle(-1.0, 1.0, 3.0)(x)

Zadeh norms (``min``, ``max`` and 1-complement) are the default methods for
intersection, union and complement, respectively. Notice that we use the
standard operators for and, or and not operations (that is, ``&``, ``|`` e
``~``)::

    aandb_zadeh = a & b             # A and B
    aorb_zadeh = a | b              # A or B

There are a number of norms implemented in Peach. Probabilistic norms are based
on the corresponding operations in probability. We will configure the
``FuzzySet`` class to use them::

    FuzzySet.set_norm(ProbabilisticAnd)
    FuzzySet.set_conorm(ProbabilisticOr)
    aandb_prob = a & b
    aorb_prob = a | b

There are other norms that we could use. Please, check the documentation for a
complete list. Here are some of them:

    Norms:
       ``ZadehAnd``, ``ProbabilisticAnd``, ``DrasticProduct``,
       ``EinsteinProduct``

    Conorms:
       ``ZadehOr``, ``ProbabilisticOr``, ``DrasticSum``, ``EinsteinSum``

    Negations:
       ``ZadehNot``, ``ProbabilisticNot``

You can easily create your own norms, too. Norms and negations are simple
functions, with no need to instantiation of classes. To create a norm or a
conorm, implement it as a function that takes two arguments and return the
result of the operation. To create a negation, implement it as a function that
takes one argument and return the result of the operation. These functions can
be used anywhere a norm, a conorm or a negation are expected.

The figure below shows the results of the using the norms.

.. image:: figs/norms-conorms.png
   :align: center
