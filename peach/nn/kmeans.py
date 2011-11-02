################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: nn/kmeans.py
# Clustering for use in radial basis functions
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
K-Means clustering algorithm

This sub-package implements the K-Means clustering algorithm. This algorithm,
given a set of points, finds a set of vectors that best represents a partition
for these points. These vectors represent the center of a cloud of points that
are nearest to them.

This algorithm is one that can be used with radial basis function (RBF) networks
to find the centers of the RBFs. Usually, training a RBFN in two passes -- first
positioning them, and then computing their variance.
"""

################################################################################
from numpy import sum, argmin, array, mean, reshape
from numpy.random import standard_normal


################################################################################
# Functions
################################################################################

################################################################################
# Classifiers
# These functions classify a set of points associating them to centers according
# to a given metric. To create a classifier, the first parameter must be the set
# of points, and the second parameter must be the list of centers. No other
# parameters are needed.
def ClassByDistance(xs, c):
    '''
    Given a set of points and a list of centers, classify the points according
    to their euclidian distance to the centers.

    :Parameters:
      xs
        Set of points to be classified. They must be given as a list or array of
        one-dimensional vectors, one per line.
      c
        Set of centers. Must also be given as a lista or array of
        one-dimensional vectors, one per line.

    :Returns:
      A list of index of the classification. The indices are the position of the
      cluster in the given parameters ``c``.
    '''
    res = [ ]
    for x in xs:
        dists = sum((x - c)**2, axis=1)
        res.append(argmin(dists))
    return res


################################################################################
# Clusterers
# These functions compute, from a set of points, a single vector that represents
# the cluster. To create a clusterer, the function needs only one parameter, the
# set of points to be clustered. This is given in form of a list. The function
# must return a single vector representing the cluster.
def ClusterByMean(x):
    '''
    This function computes the center of a cluster by averaging the vectors in
    the input set by simply averaging each component.

    :Parameters:
      x
        Set of points to be clustered. They must be given in the form of a list
        or array of one-dimensional points.

    :Returns:
      A one-dimensional array representing the center of the cluster.
    '''
    return mean(x, axis=0)

################################################################################
# Classes
################################################################################
class KMeans(object):
    '''
    K-Means clustering algorithm

    This class implements the known and very used K-Means clustering algorithm.
    In this algorithm, the centers of the clusters are selected randomly. The
    points on the training set are classified in accord to their closeness to
    the cluster centers. This changes the positions of the centers, which
    changes the classification of the points. This iteration is repeated until
    no changes occur.

    Traditional K-Means implementations classify the points in the training set
    according to the euclidian distance to the centers, and centers are computed
    as the average of the points associated to it. This is the default behaviour
    of this implementation, but it is configurable. Please, read below for more
    detail.
    '''
    def __init__(self, training_set, nclusters, classifier=ClassByDistance,
                 clusterer=ClusterByMean):
        '''
        Initializes the algorithm.

        :Parameters:
          training_set
            A list or array of vectors containing the data to be classified.
            Each of the vectors in this list *must* have the same dimension, or
            the algorithm won't behave correctly. Notice that each vector can be
            given as a tuple -- internally, everything is converted to arrays.
          nclusters
            The number of clusters to be found. This must be, of course, bigger
            than 1. These represent the number of centers found once the
            algorithm terminates.
          classifier
            A function that classifies each of the points in the training set.
            This function receives the training set and a list of centers, and
            classify each of the points according to the given metric. Please,
            look at the documentation on these functions for more information.
            Its default value is ``ClassByDistance` , which uses euclidian
            distance as metric.
          clusterer
            A function that computes the center of the cluster, given a set of
            points. This function receives a list of points and returns the
            vector representing the cluster. For more information, look at the
            documentation for these functions. Its default value is
            ``ClusterByMean``, in which the cluster is represented by the mean
            value of the vectors.
        '''
        self.__nclusters = nclusters
        self.__x = array(training_set)
        self.__c = standard_normal((nclusters, len(self.__x[0])))
        self.classify = classifier
        self.cluster = clusterer
        self.__xc = self.classify(self.__x, self.__c)

    def __getc(self):
        return self.__c
    def __setc(self, c):
        self.__c = array(reshape(c, self.__c.shape))
    c = property(__getc, __setc)
    '''A ``numpy`` array containing the centers of the classes in the algorithm.
    Each line represents a center, and the number of lines is the number of
    classes. This property is read and write, but care must be taken when
    setting new centers: if the dimensions are not exactly the same as given in
    the instantiation of the class (*ie*, *C* centers of dimension *N*, an
    exception will be raised.'''
        
    def step(self):
        '''
        This method runs one step of the algorithm. It might be useful to track
        the changes in the parameters.

        :Returns:
          The computed centers for this iteration.
        '''
        x = self.__x
        c = self.__c
        xc = self.classify(x, c)
        self.__xc = xc
        cnew = [ ]
        for i in range(self.__nclusters):
            xi = [ xij for xij, clij in zip(x, xc) if clij == i ]
            if xi:
                cnew.append(self.cluster(array(xi)))
            else:
                cnew.append(standard_normal(c[i,:].shape))
        return array(cnew)

    def __call__(self, imax=20):
        '''
        The ``__call__`` interface is used to run the algorithm until
        convergence is found.

        :Parameters:
          imax
            Specifies the maximum number of iterations admitted in the execution
            of the algorithm. It defaults to 20.

        :Returns:
          An array containing, at each line, the vectors representing the
          centers of the clustered regions.
        '''
        i = 0
        xc = [ ]
        while i < imax and xc != self.__xc:
            xc = self.__xc
            self.__c = self.step()
            i = i + 1
        return self.__c


if __name__ == "__main__":

    from random import shuffle
    from basic import *
    
    xs = [ ]
    for i in range(7):
        xs.append(array([ -1., -1. ] + 0.1*standard_normal((2,))))
    for i in range(7):
        xs.append(array([ 1., -1. ] + 0.1*standard_normal((2,))))
    for i in range(7):
        xs.append(array([ 0., 1. ] + 0.1*standard_normal((2,))))
    #shuffle(xs)
    
    k = KMeans(xs, 3)
    c = k()

    print c
    xc = k.classify(xs, c)
    for xx, xxc in zip(xs, xc):
        print xx, xxc, c[xxc,:]

    xs = array(xs)
    a1 = start_square()
    a1.hold(True)
    a1.grid(True)
    for xx in xs:
        a1.scatter(xx[0], xx[1], c='black', marker='x')
    a1.scatter(c[:,0], c[:,1], c='red', marker='o')
    savefig('kmeans.png')