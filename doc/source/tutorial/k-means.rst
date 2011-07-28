K-Means Clustering
==================

K-Means is a very well known clustering method that finds use in many areas,
from computer vision to signal processing. Its principle is very simple: given
a set of examples, and a set of clusters represented by its centers, first it
classifies the points in the training set according to the clusters. This makes
the position of the centers to change, so they are recomputed. This, on the
other hand, changes the clustering for some of the examples, and so on. This
procedure is repeated as long as changes in the clustering happen. This tutorial
shows how to work with K-Means using Peach.

We will create clouds of points around five centers by adding a random number
with gaussian distribution (zero average and small variance) to the vectors
representing the center. Then we will create a K-Mean algorithm will converge
five clusters to the clouds. We show the final result in a plot.

We start, as always, by importing ``numpy`` and ``peach``. We will need the
``random`` module also to generate the training set::

  from numpy import *
  import random
  import peach as p

The most important part on this tutorial is the creation of the training set. We
will use five centers, positioned at the coordinates ``(1, 0)``, ``(1, 1)``,
``(0, 1)``, ``(-1, 1)``, ``(-1, 0)``. Our training set will have 300 points.
Each point of the training set is found by adding a gaussian generated number to
each coordinate of the corresponding center. The following sequence of commands
does the job::

  train_size = 300
  centers = [ array([ 1.0, 0.0 ], dtype=float),
              array([ 1.0, 1.0 ], dtype=float),
              array([ 0.0, 1.0 ], dtype=float),
              array([-1.0, 1.0 ], dtype=float),
              array([-1.0, 0.0 ], dtype=float) ]
  xs = [ ]
  for i in range(train_size):
      x1 = random.gauss(0.0, 0.1)
      x2 = random.gauss(0.0, 0.1)
      xs.append(centers[i%5] + array([ x1, x2 ], dtype=float))

Since we are working on the plane, each example and each cluster will have two
components. We will have five clusters, since we have five centers. The
K-Means instance is created below::

  km = p.KMeans(xs, 5)
  for i in range(5):
      km.c[i, 0] = 0.3 * cos(i*pi/4)
      km.c[i, 1] = 0.3 * sin(i*pi/4)

The ``__call__`` interface runs the algorithm till completion. It returns the
centers of the classification. We might pass the parameter ``imax`` to the
algorithm. This is the maximum number of passes. In general, K-Means will
converge very fastly and with little error. The default value for this parameter
is 20. Notice, however, that the algorithm automatically stops if there are no
more changes in the clusters::

  c = km()

We get the following results for the centers (with four significant digits):
    
=======  ===================
cluster  center
=======  ===================
 0       [  0.9978 -0.0069 ]
 1       [  0.9945  1.0046 ]
 2       [  0.0018  1.0086 ]
 3       [ -0.9857  1.0060 ]
 4       [ -1.0202 -0.0035 ]
=======  ===================

The figure shows the result. In black crosses we see the training set, and in
red points we see the centers of the clusters.

.. image:: figs/k-means.png
   :align: center


