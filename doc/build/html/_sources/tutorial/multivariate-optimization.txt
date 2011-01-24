Optimization of a Multivariate Function
=======================================

The gradient and Newton search can be used without many changes to optimize
functions of multiple variables. In fact, there is no change at all in the way
that the optimizers are created -- but some care must be taken in the way that
the objective function is created, and how the estimates must be taken care of.

The objective function to be minimized must be programmed to receive a
one-dimensional array, and should return one single scalar value. The ``shape``
property of the array that the function will receive is ``(n, )``, meaning that
it is an array with only one dimension, with ``n`` components, each component
being one of the variables to be optimized.

Notice that this will affect the manner that derivatives are informed. The first
derivative will be, in fact, the gradient vector of the function, by applying to
the function partial derivatives in respect to each variable. The second
derivative will be the hessian matrix. We strongly suggest consulting any good
reference on the subject. Any way, if it is too difficult to create the
functions that will create these objects, you can always omit them and let the
algorithm estimate them.

The estimates can be passed to the algorithm as any iterable, with any
dimensions. Internally, however, it will be converted to a one-dimensional array
with the characteristics described above. The conversion is done by means of the
``ravel()`` method of the arrays. When finished, the value returned by the
algorithm will also be a one-dimension array.

Other than that, the algorithms behave in the same way, with no distinction on
how the methods are used. We will use them to minimize the two-dimensional
Rosenbrock function, given by:

.. math::

  f(x, y) = (1 - x)^2 + (y - x^2)^2

Its gradient vector can be computed by the expression:

.. math::

  \nabla f(x, y) = \left[ \begin{array}{c}
    -2 (1 - x) - 4x (y - x^2) \\
    2 (y - x^2)
  \end{array} \right]

And its hessian matrix by:

.. math::

  H(x, y) = \left[ \begin{array}{cc}
    2 - 4(y - 3x^2) & -4x \\
                -4x &   2
  \end{array} \right]

So, knowing these facts, we can program the optimization. We start by importing
``numpy`` for arrays and ``peach`` for the library::

    from numpy import *
    import peach as p

Let's define the functions that will help in the convergence. Notice that the
functions receive only one argument, which will be a one-dimensional array with
two components, and split them to get the value of each variable. Also, notice
that the gradient of the objective function returns a one-dimensional array with
two components, corresponding to the partial derivatives to :math:`x` and
:math:`y`, and the hessian function returns a two-dimensional array, with the
corresponding derivatives::

    def f(xy):
        x, y = xy
        return (1.-x)**2. + (y-x*x)**2.

    def df(xy):
        x, y = xy
        return array( [ -2.*(1.-x) - 4.*x*(y - x*x), 2.*(y - x*x) ])

    def hf(xy):
        x, y = xy
        return array([ [ 2. - 4.*(y - 3.*x*x), -4.*x ],
                       [ -4.*x, 2. ] ])

Now we will create the optimizers. We create these optimizers in the same way we
created other optimizers: by instantiating the corresponding class, passing the
function and the first estimate. Notice that the first estimates are given in
the form of a tuple, with the first estimate of :math:`x` in the first place,
and the first estimate of :math:`y` in the second place. There is no need to use
tuples: lists or arrays will do. To create the optimizers, we issue::

    grad = p.Gradient(f, (0.1, 0.2), [ (0., 2.), (0., 2.) ], df)
    newton = p.Newton(f, (0.1, 0.2), [ (0., 2.), (0., 2.) ], df, ddf)

Notice that we can specify the range allowed for each variable, as a list of
ranges. Each range is given as before: a duple with the lower and upper limit.
Alternatively, this can be a list with only one duple -- if given this way, the
same range will be applied to every variable. This parameter can be omitted --
if not given, the values of the variables are not restricted in any way.

Also, the derivatives parameters can be omitted -- if this is done, the
algorithm will estimate them. We will execute the algorithm step by step. We can
do this to keep track of the estimates to plot a graphic. We do this using the
commands::

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
optimization of the same function. The function itself is represented as contour
curves in the plane, and the estimate tracks over them. It is difficult to see
how fast they converged with this representation -- nonetheless, we can see that
the results were those desired.

.. image:: figs/multivariate-optimization.png
   :align: center
