################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/control-surface.py
# Generating the control surface for a two-variable fuzzy controller
################################################################################


# We import numpy for arrays and peach for the library. Actually, peach also
# imports the numpy module, but we want numpy in a separate namespace:
import numpy
from peach.fuzzy import *
from math import pi
import pylab as p
import mpl_toolkits.mplot3d as p3


# This tutorial shows how to work with a fuzzy-based controller. It is really
# easy to build a standard controller using Peach. We won't go into details of
# how a controller should work -- please, consult the literature on the subject,
# as it is very rich and explains the topic a lot better than we could do here.
#
# We will build a controller that could control an inverted pendulum. This
# controller works fine with simulations, but was never tested on a physical
# implementation. Nonetheless, it is a nice example of how to use a controller
# in Peach. We won't however, simulate it with a model of an inverted pendulum
# -- if you want to see such simulation at work, please give a look in the
# Inverted Pendulum demo.
#
# We will control the angular position and the angular velocity of the pendulum.
# To do that, we need to create the membership functions for each controlled
# variable. We will use five membership functions for the angular position
# (in general represented by theta): big negative (tbn), small negative (tsn),
# near zero (tz), small positive (tsp) and big positive (tbp). Also, we will use
# five membership functions for the angular velocity (in general represented by
# the greek letter omega): big negative (wbn), small negative (wsn), near zero
# (wz), small positive (wsp) and big positive (wbp). We define these functions
# below:

Points = 50

# Theta ranges from -pi to pi, angles given in radians.
theta = numpy.linspace(-pi, pi, Points)
tbn = DecreasingRamp(-pi/2.0, -pi/4.0)
tsn = Triangle(-pi/2.0, -pi/4.0, 0.0)
tz = Triangle(-pi/4.0, 0.0, pi/4.0)
tsp = Triangle(0.0, pi/4.0, pi/2.0)
tbp = IncreasingRamp(pi/4.0, pi/2.0)

# Omega ranges from -pi/2 to pi/2, given in radians per second.
omega = numpy.linspace(-pi/2.0, pi/2.0, Points)
wbn = DecreasingRamp(-pi/4.0, -pi/8.0)
wsn = Triangle(-pi/4.0, -pi/8.0, 0.0)
wz = Triangle(-pi/8.0, 0.0, pi/8.0)
wsp = Triangle(0.0, pi/8.0, pi/4.0)
wbp = IncreasingRamp(pi/8.0, pi/4.0)


# We also need to create membership functions to the output variable. In the
# case of the control of an inverted pendulum, this is the force applied to the
# chart. We will use, also, five membership functions, with naming similar to
# the ones above. F will range from -30 to 30 Newtons. In the case of this
# example, this range is very arbitrary, it should be adjusted for more specific
# cases.
f = numpy.linspace(-30.0, 30.0, 500)
fbn = Triangle(-30.0, -20.0, -10.0)(f)
fsn = Triangle(-20.0, -10.0, 0.0)(f)
fz = Triangle(-10.0, 0.0, 10.0)(f)
fsp = Triangle(0.0, 10.0, 20.0)(f)
fbp = Triangle(10.0, 20.0, 30.0)(f)


# Now we create the controller and input the decision rules. Rules are tipically
# given in the form of a table, if there are two variables being controlled.
# A controller in Peach has a method, add_table, that allows to give all the
# decision rules in that form. Notice, however, that single variable controllers
# should use a different method to input the rules.
#
# In the case of add_table, there are three parameters: the first one is a list
# of membership functions for the first input variable and represent the rows
# of the table; the second is a list of membership functions for the second
# variable and represents the columns of the table; the last parameter is a list
# of list that makes the table itself -- its elements are the membership
# function corresponding to the consequent of the crossing of the row and the
# column.
#
# In this example, we will use the following table:
#
#          | wbn | wsn |  wz | wsp | wbp
#          +-----+-----+-----+-----+-----
#     tbn  | fbn | fbn | fbn | fsn |  fz
#     tsn  | fbn | fbn | fsn |  fz | fsp
#      tz  | fbn | fsn |  fz | fsp | fbp
#     tsp  | fsn |  fz | fsp | fbp | fbp
#     tbp  |  fz | fsp | fbp | fbp | fbp
#
# Here is what these rules mean:
#
# IF Theta is tbn AND Omega is wbn THEN f is fbn
# IF Theta is tbn AND Omega is wsn THEN f is fbn
# IF Theta is tbn AND Omega is  wz THEN f is fbn
# IF Theta is tbn AND Omega is wsp THEN f is fsn
# IF Theta is tbn AND Omega is wbp THEN f is  fz
#
# and so on.
c = Controller(f, [], Centroid)
c.add_table([ tbn, tsn, tz, tsp, tbp ], [ wbn, wsn, wz, wsp, wbp ],
    [ [ fbn, fbn, fbn, fsn, fz  ],
      [ fbn, fbn, fsn, fz,  fsp ],
      [ fbn, fsn, fz,  fsp, fbp ],
      [ fsn, fz,  fsp, fbp, fbp ],
      [ fz,  fsp, fbp, fbp, fbp ] ] )


# This section of code generates the surface. This iterates over every point
# in the Theta and Omega intervals and calls the controller to receive the value
# of the output variable. That will be Points**2 samples, so it might take a
# while to compute.
fh = numpy.zeros((Points, Points))
for i in range(0, Points):
    for j in range(0, Points):
        t = (i - Points/2.0) / (Points / 2.0) * pi
        w = (j - Points/2.0) / Points * pi
        fh[i, j] = c(t, w)


# We will use the matplotlib module to plot these functions. We save the plot in
# a figure called 'control-surface.png'.
fig = p.figure()
a1 = p3.Axes3D(fig)

theta, omega = numpy.meshgrid(theta, omega)
a1.plot_surface(theta, omega, fh)
p.savefig("control-surface.png")
