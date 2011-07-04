Generating a Control Surface
============================

In this tutorial we will show how to use Peach to implement a two-variable
Mamdani controller and generate its control surface. It is very easy to deal
with controllers in Peach, as it implements the whole logic and a lot of support
methods and functions. We won't go into details of how a controller should work
-- please, consult the literature on the subject, as it is very rich and
explains the topic a lot better than we could do here.

We will build a controller that could control an inverted pendulum. The
controller described here works fine in simulations, but was never tested on a
physical implementation. Nonetheless, it is a nice example of how to use a
controller in Peach. We won't however, simulate it with a model of an inverted
pendulum -- if you want to see such simulation at work, please give a look in
the Inverted Pendulum demo.

We will control the angular position and the angular velocity of the pendulum.
To do that, we need to create the membership functions for each controlled
variable. We will use five membership functions for the angular position
(in general represented by the greek letter :math:`\theta`): big negative
(``tbn``), small negative (``tsn``), near zero (``tz``), small positive
(``tsp``) and big positive (``tbp``). Also, we will use five membership
functions for the angular velocity (in general represented by the greek letter
:math:`\omega`): big negative (``wbn``), small negative (``wsn``), near zero
(``wz``), small positive (``wsp``) and big positive (``wbp``).

Remember that, in the context of a controller, you should supply functions, not
fuzzy sets! We define these functions below::

    Points = 50

    theta = numpy.linspace(-pi, pi, Points)
    tbn = DecreasingRamp(-pi/2.0, -pi/4.0)
    tsn = Triangle(-pi/2.0, -pi/4.0, 0.0)
    tz = Triangle(-pi/4.0, 0.0, pi/4.0)
    tsp = Triangle(0.0, pi/4.0, pi/2.0)
    tbp = IncreasingRamp(pi/4.0, pi/2.0)

    omega = numpy.linspace(-pi/2.0, pi/2.0, Points)
    wbn = DecreasingRamp(-pi/4.0, -pi/8.0)
    wsn = Triangle(-pi/4.0, -pi/8.0, 0.0)
    wz = Triangle(-pi/8.0, 0.0, pi/8.0)
    wsp = Triangle(0.0, pi/8.0, pi/4.0)
    wbp = IncreasingRamp(pi/8.0, pi/4.0)

Notice that we explicitly created each of the membership functions, but we could
use auxiliary functions to do that. Since it is very common, in fuzzy
controllers, to use a sequence of triangle functions to represent the membership
functions of the variables, Peach supplies two functions to deal with that,
``Saw``, which generates a sequence of triangles equally spaced in a given
interval, and ``FlatSaw``, which generates also a sequence of triangles, but
ramps in the extremes. The creation of the membership functions could be done
like this::

    tbn, tsn, tz, tsp, tbp = FlatSaw((-pi, pi), 5)
    wbn, wsn, wz, wsp, wbp = FlatSaw((-pi/2, pi/2), 5)

We also need to create membership functions to the output variable. In the case
of the control of an inverted pendulum, this is the force applied to the chart.
We will use, also, five membership functions, with naming similar to the ones
above. The force F will range from -30 to 30 newtons. In the case of this
example, this range is very arbitrary, it should be adjusted for more specific
cases. The information about the output variable can be supplied as membership
functions but, since these will be used to defuzzify the control, we can get an
answer a little bit faster if we supply fuzzy sets::

    f = numpy.linspace(-30.0, 30.0, 500)
    fbn = Triangle(-30.0, -20.0, -10.0)(f)
    fsn = Triangle(-20.0, -10.0, 0.0)(f)
    fz = Triangle(-10.0, 0.0, 10.0)(f)
    fsp = Triangle(0.0, 10.0, 20.0)(f)
    fbp = Triangle(10.0, 20.0, 30.0)(f)

Now we create the controller and input the decision rules. Rules are tipically
given in the form of a table, if there are two variables being controlled. A
controller in Peach has a method, add_table, that allows to give all the
decision rules in that form. Notice, however, that single variable controllers
should use a different method to input the rules (see the previous tutorial for
more information on that).

In the case of add_table, there are three parameters: the first one is a list of
membership functions for the first input variable and represent the rows of the
table; the second is a list of membership functions for the second variable and
represents the columns of the table; the last parameter is a list of list that
makes the table itself -- its elements are the membership function corresponding
to the consequent of the crossing of the row and the column.

In this example, we will use the following table:

================================ ======= ======= ======= ======= =======
:math:`\theta` \\ :math:`\omega` ``wbn`` ``wsn``  ``wz`` ``wsp`` ``wbp``
================================ ======= ======= ======= ======= =======
                         ``tbn`` ``fbn`` ``fbn`` ``fbn`` ``fsn``  ``fz``
                         ``tsn`` ``fbn`` ``fbn`` ``fsn``  ``fz`` ``fsp``
                          ``tz`` ``fbn`` ``fsn``  ``fz`` ``fsp`` ``fbp``
                         ``tsp`` ``fsn``  ``fz`` ``fsp`` ``fbp`` ``fbp``
                         ``tbp``  ``fz`` ``fsp`` ``fbp`` ``fbp`` ``fbp``
================================ ======= ======= ======= ======= =======

Here is what these rules mean:

IF :math:`\theta` is ``tbn`` AND :math:`\omega` is ``wbn`` THEN F is ``fbn``

IF :math:`\theta` is ``tbn`` AND :math:`\omega` is ``wsn`` THEN F is ``fbn``

IF :math:`\theta` is ``tbn`` AND :math:`\omega` is ``wz`` THEN F is ``fbn``

IF :math:`\theta` is ``tbn`` AND :math:`\omega` is ``wsp`` THEN F is ``fsn``

IF :math:`\theta` is ``tbn`` AND :math:`\omega` is ``wbp`` THEN F is ``fz``

and so on.

With the commands below we create the controller. We won't be adding directly
any rules, and we will use centroid as defuzzification method::

    c = Controller(f, [], Centroid)
    c.add_table([ tbn, tsn, tz, tsp, tbp ], [ wbn, wsn, wz, wsp, wbp ],
        [ [ fbn, fbn, fbn, fsn, fz  ],
          [ fbn, fbn, fsn, fz,  fsp ],
          [ fbn, fsn, fz,  fsp, fbp ],
          [ fsn, fz,  fsp, fbp, fbp ],
          [ fz,  fsp, fbp, fbp, fbp ] ] )

Notice how the decision table was directly converted in a list of lists. The
format of the table in the ``add_table`` method is exactly the same as the table
in the definition of the controller. The following code generates the surface.
This iterates over every point in the :math:`\theta` and :math:`\omega`
intervals and calls the controller to receive the value of the output variable.
That will be ``Points**2`` samples, so it might take a while to compute::

    fh = numpy.zeros((Points, Points))
    for i in range(0, Points):
        for j in range(0, Points):
            t = (i - Points/2.0) / (Points / 2.0) * pi
            w = (j - Points/2.0) / Points * pi
            fh[i, j] = c(t, w)

The ``matplotlib`` module has some capabilities to plot 3D graphics. We use them
to obtain the following figure:

.. image:: figs/control-surface.png
   :align: center