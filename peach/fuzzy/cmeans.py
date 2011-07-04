################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/cmeans.py
# Fuzzy C-Means algorithm
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Fuzzy C-Means

Fuzzy C-Means is a clustering algorithm based on fuzzy logic.

This package implements the fuzzy c-means algorithm for clustering and
classification. This algorithm is very simple, yet very efficient. From a
training set and an initial condition which gives the membership values of each
example in the training set to the clusters, it converges very fastly to crisper
sets.

The initial conditions, ie, the starting membership, must follow some rules.
Please, refer to any bibliography about the subject to see why. Those rules are:
no example might have membership 1 in every class, and the sum of the membership
of every component must be equal to 1. This means that the initial condition is
a fuzzy partition of the universe.
"""


################################################################################
import numpy
from numpy import dot, array, sum, zeros, outer


################################################################################
# Fuzzy C-Means class
################################################################################
class FuzzyCMeans(object):
    '''
    Fuzzy C-Means convergence.

    Use this class to instantiate a fuzzy c-means object. The object must be
    given a training set and initial conditions. The training set is a list or
    an array of N-dimensional vectors; the initial conditions are a list of the
    initial membership values for every vector in the training set -- thus, the
    length of both lists must be the same. The number of columns in the initial
    conditions must be the same number of classes. That is, if you are, for
    example, classifying in ``C`` classes, then the initial conditions must have
    ``C`` columns.

    There are restrictions in the initial conditions: first, no column can be
    all zeros or all ones -- if that happened, then the class described by this
    column is unnecessary; second, the sum of the memberships of every example
    must be one -- that is, the sum of the membership in every column in each
    line must be one. This means that the initial condition is a perfect
    partition of ``C`` subsets.

    Notice, however, that *no checking* is done. If your algorithm seems to be
    behaving strangely, try to check these conditions.
    '''
    def __init__(self, training_set, initial_conditions, m=2.):
        '''
        Initializes the algorithm.

        :Parameters:
          training_set
            A list or array of vectors containing the data to be classified.
            Each of the vectors in this list *must* have the same dimension, or
            the algorithm won't behave correctly. Notice that each vector can be
            given as a tuple -- internally, everything is converted to arrays.
          initial_conditions
            A list or array of vectors containing the initial membership values
            associated to each example in the training set. Each column of this
            array contains the membership assigned to the corresponding class
            for that vector. Notice that each vector can be given as a tuple --
            internally, everything is converted to arrays.
          m
            This is the aggregation value. The bigger it is, the smoother will
            be the classification. Please, consult the bibliography about the
            subject. ``m`` must be bigger than 1. Its default value is 2
        '''
        self.__x = array(training_set)
        self.__mu = array(initial_conditions)
        self.m = m
        '''The fuzzyness coefficient. Must be bigger than 1, the closest it is
        to 1, the smoother the membership curves will be.'''
        self.__c = self.centers()

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

    def __getmu(self):
        return self.__mu
    mu = property(__getmu, None)
    '''The membership values for every vector in the training set. This property
    is modified at each step of the execution of the algorithm. This property is
    not writable.'''

    def __getx(self):
        return self.__x
    x = property(__getx, None)
    '''The vectors in which the algorithm bases its convergence. This property
    is not writable.'''

    def centers(self):
        '''
        Given the present state of the algorithm, recalculates the centers, that
        is, the position of the vectors representing each of the classes. Notice
        that this method modifies the state of the algorithm if any change was
        made to any parameter. This method receives no arguments and will seldom
        be used externally. It can be useful if you want to step over the
        algorithm. *This method has a colateral effect!* If you use it, the
        ``c`` property (see above) will be modified.

        :Returns:
          A vector containing, in each line, the position of the centers of the
          algorithm.
        '''
        mm = self.__mu ** self.m
        c = dot(self.__x.transpose(), mm) / sum(mm, axis=0)
        self.__c = c.transpose()
        return self.__c

    def membership(self):
        '''
        Given the present state of the algorithm, recalculates the membership of
        each example on each class. That is, it modifies the initial conditions
        to represent an evolved state of the algorithm. Notice that this method
        modifies the state of the algorithm if any change was made to any
        parameter.

        :Returns:
          A vector containing, in each line, the membership of the corresponding
          example in each class.
        '''
        x = self.__x
        c = self.__c
        M, _ = x.shape
        C, _ = c.shape
        r = zeros((M, C))
        m1 = 1./(self.m-1.)
        for k in range(M):
            den = sum((x[k] - c)**2., axis=1)
            frac = outer(den, 1./den)**m1
            r[k, :] = 1. / sum(frac, axis=1)
        self.__mu = r
        return self.__mu

    def step(self):
        '''
        This method runs one step of the algorithm. It might be useful to track
        the changes in the parameters.

        :Returns:
          The norm of the change in the membership values of the examples. It
          can be used to track convergence and as an estimate of the error.
        '''
        old = self.__mu
        self.membership()
        self.centers()
        return sum(self.__mu - old)**2.

    def __call__(self, emax=1.e-10, imax=20):
        '''
        The ``__call__`` interface is used to run the algorithm until
        convergence is found.

        :Parameters:
          emax
            Specifies the maximum error admitted in the execution of the
            algorithm. It defaults to 1.e-10. The error is tracked according to
            the norm returned by the ``step()`` method.
          imax
            Specifies the maximum number of iterations admitted in the execution
            of the algorithm. It defaults to 20.

        :Returns:
          An array containing, at each line, the vectors representing the
          centers of the clustered regions.
        '''
        error = 1.
        i = 0
        while error > emax and i < imax:
            error = self.step()
            i = i + 1
        return self.c


################################################################################
# Test.
if __name__ == "__main__":
    pass