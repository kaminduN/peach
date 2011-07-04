Using Membership Functions
==========================

The aim of this tutorial is to show how to create and use membership functions
with the fuzzy logic package of Peach. Most of the more used membership
functions are already implemented, but there are ways to use different functions
if it is needed.

Membership functions in Peach are implemented as classes. You can instance a
class, passing some parameters to it, to create a function that can be used with
scalar numbers or arrays in general. To create a function, just instance it,
passing whatever parameters are needed. To use it, just call the function you
just created. Suppose we have already imported Peach in the command line. Then,
you can create (say) an increasing ramp starting in ``x = -1`` and ending in
``x = 1``, and use it, just by issuing the commands::

    >>> ramp = peach.IncreasingRamp(-1., 1.)
    >>> ramp(-1)
    FuzzySet(0.0)
    >>> ramp(1)
    FuzzySet(1.0)
    >>> ramp(0.)
    FuzzySet(0.5)

Notice that the membership function returns a ``FuzzySet`` object. A
``FuzzySet`` object is a scalar or array with the logic operations defined to
work as in fuzzy logic.

Below we have some of the implemented functions with the corresponding
parameters. In these, consider ``x`` as the parameter passed when calling the
function:

    ``IncreasingRamp(x0, x1)``
      An increasing ramp, returning 0 if ``x`` is less than ``x0``, 1 if ``x``
      is greater than ``x1``, and a straight line linking these points if ``x``
      is inbetween. Notice that ``x0`` must be lower than ``x1``.

    ``DecreasingRamp(x0, x1)``
      A increasing ramp, returning 1 if ``x`` is less than ``x0``, 0 if ``x``
      is greater than ``x1``, and a straight line linking these points if ``x``
      is inbetween. Notice that ``x0`` must be lower than ``x1``.

    ``Triangle(x0, x1, x2)``
      A triangle function, returning 0 if ``x`` is less than ``x0`` of greater
      than ``x2``, a maximum value of 1 if ``x`` is equal to ``x1`` and straight
      lines connecting these points. Notice that ``x0`` must be lower than
      ``x1``, and that both must be lower than ``x2``.

    ``Trapezoid(x0, x1, x2, x3)``
      A trapezoid function, returning 0 if ``x`` is less than ``x0`` of greater
      than ``x3``, a value of 1 if ``x`` is between ``x1`` and ``x2`` and
      straight lines connecting these points. Notice that we must assure that
      ``x0`` < ``x1`` < ``x2`` < ``x3``.

    ``Gaussian(x0, a)``
      A gaussian function centered at ``x0`` and width ``a``. Notice that ``a``
      is not the variance of the gaussian, but behaves in the same way. That
      means that the bigger the value of ``a``, the more open will be the
      function. The default value of ``x0`` is 0, and of ``a`` is 1.

    ``IncreasingSigmoid(x0, a)``
      An increasing sigmoid with middle point at ``x0`` and inclination ``a``.
      The bigger the value of ``a``, the steepest will be the sigmoid. The
      default value of ``x0`` is 0, and of ``a`` is 1.

    ``DecreasingSigmoid(x0, a)``
      A decreasing sigmoid with middle point at ``x0`` and inclination ``a``.
      The bigger the value of ``a``, the steepest will be the sigmoid. The
      default value of ``x0`` is 0, and of ``a`` is 1.

    ``Bell(x0, a, b)``
      A generalized bell centered at ``x0``, width ``a`` and exponent ``2*b``.
      The default value of ``x0`` is 0, of ``a`` and ``b`` is 1.

There are other predefined membership functions. Please, consult the reference
for more information. The figure below shows the aspect of the functions, in
the following order: in the first plot: a decreasing ramp, a triangle, a
trapezoid and an increasing ramp; in the second plot, a decreasing sigmoid, a
gaussian, a generalized bell and an increasing sigmoid.

.. image:: figs/membership-functions.png
   :align: center
