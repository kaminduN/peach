Optimization of a Multivariate Function by Quasi-Newton Methods
===============================================================

Theoretically, the Newton search is the best known method to minimize a
function. It has its drawbacks, however. The main problem with the method is
that it must compute the inverse of the hessian matrix at every iteration of the
algorithm. This is not a problem if the problem to be solved has low
dimensionality (that is, few variables to optimize). Unfortunatelly, this is not
always the case, so alternative methods had to be developed.

The *quasi-Newton methods* are a class of methods based on the Newton search
where, instead of computing the hessian of the function, and then inverting the
resulting matrix, an estimate of the inverse hessian is kept and updated by the
algorithm. There are some different methods that implement this idea. Peach has
implemented in it three of them: DFP, BFGS and SR1.

There is absolutelly no difference in how a quasi-Newton optimizer is created
and used. We will be minimizing the Rosenbrock function, as always. We start by
importing ``peach`` and ``numpy`` in different namespaces::

    from numpy import *
    import peach as p

We must also create the objective function and the gradient function, as the
algorithms use them. Notice however that, as before, we can omit the gradient
function and let Peach estimate them for us if needed::

    def f(xy):
        x, y = xy
        return (1.-x)**2. + (y-x*x)**2.

    def df(xy):
        x, y = xy
        return array( [ -2.*(1.-x) - 4.*x*(y - x*x), 2.*(y - x*x) ])


Now we will create the optimizers. We create these optimizers in the same way we
created other optimizers: by instantiating the corresponding class, passing the
function and the first estimate. Notice that the first estimates are given in
the form of a tuple, with the first estimate of :math:`x` in the first place,
and the first estimate of :math:`y` in the second place. There is no need to use
tuples: lists or arrays will do. To create the optimizers, we issue::

    dfp = p.DFP(f, (0.1, 0.2), [ (0., 2.), (0., 2.) ], df)
    bfgs = p.BFGS(f, (0.1, 0.2), [ (0., 2.), (0., 2.) ], df)
    sr1 = p.SR1(f, (0.1, 0.2), [ (0., 2.), (0., 2.) ], df)

Notice that we specified ranges and derivative of the objective function. As
before, these are not needed, and if not given, the same results apply: no
restrictions on the values of the variables, and derivatives are estimated. As
we done in the other optimization tutorials, we will execute the algorithm step
by step. We can do this to keep track of the estimates to plot a graphic. We do
this using the commands::

    xd = [ ]
    xb = [ ]
    xs = [ ]
    i = 0
    while i < 200:
        x, e = dfp.step()
        xd.append(x)
        x, e = bfgs.step()
        xb.append(x)
        x, e = sr1.step()
        xs.append(x)
        i = i + 1

Notice that we used 200 iterations here. In general, although each iteration is
executed faster than in standard Newton search, we will need a little more
iterations to achieve the same results. But the real culprit here is the SR1
method: although its derivation is simple (consult a good reference for a
demonstration), it is a very inefficient method if compared with the other two.
The ``xd``, ``xb`` and  ``xs`` variables will hold, in sequence, the estimates.
We can plot them to see the convergence trace. The figure below is a
representation of the execution of these methods. The function itself is
represented as contour curves in the plane, and the estimate tracks over them.
It is difficult to see how fast they converged with this representation --
nonetheless, we can see that the results were those desired.

.. image:: figs/quasi-newton-optimization.png
   :align: center
