Linear Optimization of a Single Variable with Derivative Methods
================================================================

The optimization problem for a function of a single variable can be put as the
following: finding the :math:`x` in a function :math:`f(x)` that wields the
function's smallest value. Mathematically, it is usually put as

.. math::

  x^*  = \min f(x)

One property of the function respects to the value of its derivatives in the
minimum -- at that point the derivative of the function is zero, and its second
derivative is positive, that is:

.. math::

  f^\prime(x^*) = 0

.. math::

  f^{\prime\prime}(x^*) > 0

Two methods based on the derivatives are used to find the location of the
minimum. Those are briefly described below, but further information can be
found in any book about the subject:

  Gradient
    The gradient search uses the information of the derivative to direct the
    search. Basically, the update step has the opposite signal of the derivative
    and its magnitude is proportional to that of the derivative.

  Newton
    The Newton method is based on the expansion of the original function in
    Taylor series, and uses both the derivative and the second derivative of the
    function to compute the update step.

We will use, in this tutorial, both methods to find the minimum of the
Rosenbrock function. For that, we will need its derivatives, which are given by:

.. math::

  f^\prime(x) = -2(1-x) - 4x(1-x^2)

.. math::

  f^{\prime\prime}(x) = 2 - 4(1 - 3x^2)

As always, we first import ``numpy`` for arrays and ``peach`` for the library.
Actually, ``peach`` also the ``numpy`` module, but we want it in a separate
namespace::

    from numpy import *
    import peach as p

Let's define the function to be optimized. By using Peach, you can pass to an
optimizer any function that receives a floating point number and returns a
number. The function will be called by the optimizer, so you must ensure that it
can deal with the parameter and return an appropriate result. The derivative and
the second derivative, if used, must follow the same requirements::

    def f(x):
        return (1 - x)**2 + (1 - x*x)**2

    def df(x):
        return -2.*(1.-x) - 4.*(1.-x*x)*x

    def ddf(x):
        return 2. - 4.*(1. - 3.*x*x)

Now we will create the optimizer. We create these optimizers in the same way we
created other optimizers: by instantiating the corresponding class, passing the
function and the first estimate. These optimizers, however, accept as parameters
the derivatives. To create the optimizers, we issue::

    grad = p.Gradient(f, 0.84, (0., 2.), df)
    newton = p.Newton(f, 0.84, (0., 2.), df, ddf)

This will create, respectivelly, a gradient optimizer in the variable ``grad``
and a Newton optimizer in the variable ``newton``. The third parameter is the
interval in which the search will be done. This parameter is not needed -- if
given, however, it will instruct the algorithm to never let the estimates fall
outside the given interval. The range of values is given as a duple representing
the lower and upper limit of the search interval.

Notice that the fourth parameter in the creation of the gradient search is the
function that computes the derivative of the objective function for a given
value, and the fourth and fifth parameters in the creation of the Newton search
are the derivative and the second derivative.

Those parameters, however, can be omitted -- if omitted, an estimate based on
difference equations will be used instead. However, notice that, in general, the
estimates will be less accurate, and the computation can be a little slower. To
create the optimizers by using estimates, we issue::

    grad = p.Gradient(f, 0.84)
    newton = p.Newton(f, 0.84)

These optimizers will also search without restriction in the set of accepted
values for the variable.

Every optimizer has two additional parameters that can be specified at
instantiation time. The ``emax`` parameter estipulates what will be the maximum
error allowed. The ``imax`` parameter estipulates the maximum number of
iterations the algorithm will perform. Their default values are, respectivelly,
:math:`10^{-8}` and 1000. However, if we wanted different values, say, an error
of 0.001 or 500 iterations, we could create the optimizer by issuing the
following command::

    grad = p.Gradient(f, 0.84, (0., 2.), df, emax=0.001, imax=500)

Also, every optimizer has three interfaces. The ``step()`` method is called
without any parameters and computes the next estimate. It returns a tuple
``(x, e)``, where ``x`` is the new estimate and ``e`` is an estimate of the
error. The ``restart()`` method can be used to reset the algorithm, allowing to
change the estimate or another parameters. The ``__call__()`` method is also
called without any parameters, and computes the best estimate until a given
precision or a maximum number of iterations of the algorithm are achieved. The
``x`` property holds the last estimate of the minimum, and can be read or
written at any time. However, we suggest that, if you need to write to the
estimate, that the optimizer is reset using the ``restart()`` method.

We will execute the algorithm step by step. We can do this to keep track of the
estimates to plot a graphic. We do this using the commands::

    xg = [ ]
    xn = [ ]
    i = 0
    while i < 100:
        x, e = grad.step()
        xg.append(x)
        x, e = newton.step()
        xn.append(x)
        i = i + 1

The ``xg`` and  ``xn`` variables will hold, in sequence, the estimates. We can
plot them to see the convergence trace. The figure below is a representation of
the execution of these methods, with given and estimated derivatives, in the
optimization of the same function. It is easy to see that they converge fastly.
In fact, the Newton optimizer could converge even faster than the gradient, by
adjusting the convergence step (please, consult the documentation on the
methods, as this is one of the parameters of the instantiation of the
algorithms).

.. image:: figs/derivative-optimization.png
   :align: center
