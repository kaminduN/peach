################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/multivar.py
# Gradient and multivariable search methods
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements basic multivariable optimizers, including gradient and
Newton searches.
"""

################################################################################
from numpy import array, dot, abs, sum, roll, ones, eye, isscalar, zeros
from numpy.linalg import inv
from base import Optimizer, gradient, hessian


################################################################################
# Classes
################################################################################
class Direct(Optimizer):
    '''
    Multidimensional direct search

    This optimization method is a generalization of the 1D method, using
    variable swap as search direction. This results in a very simplistic and
    inefficient method that should be used only when any other method fails.
    '''
    def __init__(self, f, x0, ranges=None, h=0.5, emax=1e-8, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.

          x0
            First estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a list of ranges
            for each variable of the objective function. It is specified as a
            list of tuples of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. It can also be given as a list with a simple
            tuple in the same format. In that case, the same range will be
            applied for every variable in the optimization.

          h
            The initial step of the search. Defaults to 0.5

          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.

          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        Optimizer.__init__(self)
        self.__f = f
        self.__x = array(x0).ravel()
        n = self.__x.size
        self.__h = ones((n, ))
        self.__h[0] = -0.5
        self.__dx = h * eye(n, 1).reshape(self.__x.shape)

        # Determine ranges of the variables
        if ranges is not None:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0, h=0.5):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.
          h
            The initial step of the search. Defaults to 0.5
        '''
        self.__x = array(x0).ravel()
        n = self.__x.size
        self.__h = ones((n, ))
        self.__h[0] = -0.5
        self.__dx = h * eye(n, 1).reshape(self.__x.shape)


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is highly dependent of the steps
        executed before, as the search step is updated at each call to this
        method.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        f = self.__f
        x = self.__x
        dx = self.__dx
        fo = f(x)

        # Next estimate
        x = x + dx

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            x = where(x < r0, r0, x)
            x = where(x > r1, r1, x)

        # Update state
        fn = f(x)
        if fn > fo:
            self.__dx = self.__h * roll(dx, 1)
        self.__x = x
        return x, sum(abs(self.__dx))


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        i = 0
        e = sum(abs(self.__dx))
        while e > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
class Gradient(Optimizer):
    '''
    Gradient search

    This method uses the fact that the gradient of a function points to the
    direction of largest increase in the function (in general called *uphill*
    direction). So, the contrary direction (*downhill*) is used as search
    direction.
    '''
    def __init__(self, f, x0, ranges=None, df=None, h=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.

          x0
            First estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a list of ranges
            for each variable of the objective function. It is specified as a
            list of tuples of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. It can also be given as a list with a simple
            tuple in the same format. In that case, the same range will be
            applied for every variable in the optimization.

          df
            A function to calculate the gradient vector of the cost function
            ``f``. Defaults to ``None``, if no gradient is supplied, then it is
            estimated from the cost function using Euler equations.

          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.

          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.

          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        Optimizer.__init__(self)
        self.__f = f
        self.__x = array(x0).ravel()
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__h = h

        # Determine ranges of the variables
        if ranges is not None:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0, h=None):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
        '''
        self.__x = array(x0).ravel()
        if h is not None:
            self.__h = h


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent only of the given
        estimated, so it can be used for different kind of investigations on the
        same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        x = self.__x
        xold = x

        # New estimate
        x = x - self.__h * self.__df(x)

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            x = where(x < r0, r0, x)
            x = where(x > r1, r1, x)

        # Update state
        self.__x = x
        return x, sum(abs(x - xold))


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
class MomentumGradient(Optimizer):
    '''
    Gradient search with momentum

    This method uses the fact that the gradient of a function points to the
    direction of largest increase in the function (in general called *uphill*
    direction). So, the contrary direction (*downhill*) is used as search
    direction. A momentum term is added to avoid local minima.
    '''
    def __init__(self, f, x0, ranges=None, df=None, h=0.1, a=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.

          x0
            First estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a list of ranges
            for each variable of the objective function. It is specified as a
            list of tuples of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. It can also be given as a list with a simple
            tuple in the same format. In that case, the same range will be
            applied for every variable in the optimization.

          df
            A function to calculate the gradient vector of the cost function
            ``f``. Defaults to ``None``, if no gradient is supplied, then it is
            estimated from the cost function using Euler equations.

          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
            Defaults to 0.1.

          a
            Momentum term. This term is a measure of the memory of the optmizer.
            The bigger it is, the more the past values influence in the outcome
            of the optimization. Defaults to 0.1

          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.

          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        Optimizer.__init__(self)
        self.__f = f
        self.__x = array(x0).ravel()
        self.__dx = zeros(self.__x.shape)
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        self.__h = h
        self.__a = a

        # Determine ranges of the variables
        if ranges is not None:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0, h=None, a=None):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
            If not given in this method, the old value is used.
          a
            Momentum term. This term is a measure of the memory of the optmizer.
            The bigger it is, the more the past values influence in the outcome
            of the optimization. If not given in this method, the old value is
            used.
        '''
        self.__x = array(x0).ravel()
        if h is not None:
            self.__h = h
        if a is not None:
            self.__a = a


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent only of the given
        estimated, so it can be used for different kind of investigations on the
        same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        x = self.__x
        xold = x

        # New estimate
        dx = - self.__h * self.__df(x) + self.__a * self.__dx
        x = x + dx
        self.__dx = dx

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            x = where(x < r0, r0, x)
            x = where(x > r1, r1, x)

        # Update state
        self.__x = x
        return x, sum(abs(x - xold))


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
class Newton(Optimizer):
    '''
    Newton search

    This is a very effective method to find minimum points in functions. In a
    very basic fashion, this method corresponds to using Newton root finding
    method on f'(x). Converges *very* fast if the cost function is quadratic
    of similar to it.
    '''
    def __init__(self, f, x0, ranges=None, df=None, hf=None, h=0.1, emax=1e-5, imax=1000):
        '''
        Initializes the optimizer.

        To create an optimizer of this type, instantiate the class with the
        parameters given below:

        :Parameters:
          f
            A multivariable function to be optimized. The function should have
            only one parameter, a multidimensional line-vector, and return the
            function value, a scalar.

          x0
            First estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.

          ranges
            A range of values might be passed to the algorithm, but it is not
            necessary. If supplied, this parameter should be a list of ranges
            for each variable of the objective function. It is specified as a
            list of tuples of two values, ``(x0, x1)``, where ``x0`` is the
            start of the interval, and ``x1`` its end. Obviously, ``x0`` should
            be smaller than ``x1``. It can also be given as a list with a simple
            tuple in the same format. In that case, the same range will be
            applied for every variable in the optimization.

          df
            A function to calculate the gradient vector of the cost function
            ``f``. Defaults to ``None``, if no gradient is supplied, then it is
            estimated from the cost function using Euler equations.

          hf
            A function to calculate the hessian matrix of the cost function
            ``f``. Defaults to ``None``, if no hessian is supplied, then it is
            estimated from the cost function using Euler equations.

          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.

          emax
            Maximum allowed error. The algorithm stops as soon as the error is
            below this level. The error is absolute.

          imax
            Maximum number of iterations, the algorithm stops as soon this
            number of iterations are executed, no matter what the error is at
            the moment.
        '''
        Optimizer.__init__(self)
        self.__f = f
        self.__x = array(x0).ravel()
        if df is None:
            self.__df = gradient(f)
        else:
            self.__df = df
        if hf is None:
            self.__hf = hessian(f)
        else:
            self.__hf = hf
        self.__h = h

        # Determine ranges of the variables
        if ranges is not None:
            ranges = list(ranges)
            if len(ranges) == 1:
                ranges = array(ranges * len(x0[0]))
            else:
                ranges = array(ranges)
        self.ranges = ranges
        '''Holds the ranges for every variable. Although it is a writable
        property, care should be taken in changing parameters before ending the
        convergence.'''

        self.__emax = float(emax)
        self.__imax = int(imax)


    def __get_x(self):
        return self.__x

    def __set_x(self, x0):
        self.restart(x0)

    x = property(__get_x, __set_x)
    '''The estimate of the position of the minimum.'''


    def restart(self, x0, h=None):
        '''
        Resets the optimizer, returning to its original state, and allowing to
        use a new first estimate.

        :Parameters:
          x0
            New estimate of the minimum. Estimates can be given in any format,
            but internally they are converted to a one-dimension vector, where
            each component corresponds to the estimate of that particular
            variable. The vector is computed by flattening the array.
          h
            Convergence step. This method does not takes into consideration the
            possibility of varying the convergence step, to avoid Stiefel cages.
        '''
        self.__x = array(x0).ravel()
        if h is not None:
            self.__h = h


    def step(self):
        '''
        One step of the search.

        In this method, the result of the step is dependent only of the given
        estimated, so it can be used for different kind of investigations on the
        same cost function.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the updated
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        x = self.__x
        xold = x
        df = self.__df(x)
        hf = self.__hf(x)

        # New estimate
        try:
            x = x - self.__h * dot(inv(hf), df)
        except:
            x = x - self.__h * df / hf

        # Sanity check
        if self.ranges is not None:
            r0 = self.ranges[:, 0]
            r1 = self.ranges[:, 1]
            x = where(x < r0, r0, x)
            x = where(x > r1, r1, x)

        # Update state
        self.__x = x
        return x, sum(abs(x - xold))


    def __call__(self):
        '''
        Transparently executes the search until the minimum is found. The stop
        criteria are the maximum error or the maximum number of iterations,
        whichever is reached first. Note that this is a ``__call__`` method, so
        the object is called as a function. This method returns a tuple
        ``(x, e)``, with the best estimate of the minimum and the error.

        :Returns:
          This method returns a tuple ``(x, e)``, where ``x`` is the best
          estimate of the minimum, and ``e`` is the estimated error.
        '''
        emax = self.__emax
        imax = self.__imax
        e = emax
        i = 0
        while e > emax/2. and i < imax:
            _, e = self.step()
            i = i + 1
        return self.__x, e


################################################################################
# Test
if __name__ == "__main__":

    # Rosenbrock function
    def f(xy):
        x, y = xy
        return (1.-x)**2. + (y-x*x)**2.

    # Gradient of Rosenbrock function
    def df(xy):
        x, y = xy
        return array( [ -2.*(1.-x) - 4.*x*(y - x*x), 2.*(y - x*x) ])

    # Hessian of Rosenbrock function
    def hf(xy):
        x, y = xy
        return array([ [ 2. - 4.*(y - 3.*x*x), -4.*x ],
                       [ -4.*x, 2. ] ])

    linear = Direct(f, (0., 0.), emax=1e-12)
    print linear()
    grad = Gradient(f, (0., 0.), df=df, emax=1e-12)
    print grad()
    grad2 = Gradient(f, (0., 0.), emax=1e-12)
    print grad2()
    mgrad = MomentumGradient(f, (0., 0.), df=df, emax=1e-12)
    print mgrad()
    mgrad2 = MomentumGradient(f, (0., 0.), emax=1e-12)
    print mgrad2()
    newton = Newton(f, (0., 0.), df=df, hf=hf, emax=1e-12)
    print newton()
    newton2 = Newton(f, (0., 0.), emax=1e-12)
    print newton2()