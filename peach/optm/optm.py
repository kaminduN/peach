################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: optm/optm.py
# Basic definitions and base class
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic definitons and base class for optimizers

This sub-package exports some auxiliary functions to work with cost functions,
namely, a function to calculate gradient vectors and hessian matrices, which are
extremely important in optimization.

Also, a base class, ``Optimizer``, for all optimizers. Sub-class this class if
you want to create your own optmizer, and follow the interface. This will allow
easy configuration of your own scripts and comparison between methods.
"""


################################################################################
from numpy import array, zeros


################################################################################
# Auxiliary functions
################################################################################
def gradient(f, dx=1e-5):
    '''
    Creates a function that calculates the gradient vector of a scalar field.

    This function takes as a parameter a scalar function and creates a new
    function that is able to calculate the derivative (in case of single
    variable functions) or the gradient vector (in case of multivariable
    functions. Please, note that this function takes as a parameter a
    *function*, and returns as a result *another function*. Calling the returned
    function on a point will give the gradient vector of the original function
    at that point::

      >>> def f(x):
              return x^2

      >>> df = gradient(f)
      >>> df(1)
      2

    In the above example, ``df`` is a generated function which will return the
    result of the expression ``2*x``, the derivative of the original function.
    In the case ``f`` is a multivariable function, it is assumed that its
    argument is a line vector.

    :Parameters:
      f
        Any function, one- or multivariable. The function must be an scalar
        function, though there is no checking at the moment the function is
        created. If ``f`` is not an scalar function, an exception will be
        raised at the moment the returned function is used.
      dx
        Optional argument that gives the precision of the calculation. It is
        recommended that ``dx = sqrt(D)``, where ``D`` is the machine precision.
        It defaults to ``1e-5``, which usually gives a good estimate.

    :Returns:
      A new function which, upon calling, gives the derivative or gradient
      vector of the original function on the analised point. The parameter of
      the returned function is a real number or a line vector where the gradient
      should be calculated.
    '''
    def _df(x):
        try:
            x = float(x)
            return (f(x+dx) - f(x-dx)) / (2.*dx)
        except TypeError:
            n = x.size
            df = zeros((n, ))
            for i in xrange(n):
                xl = array(x)
                xl[i] = xl[i] - dx
                xr = array(x)
                xr[i] = xr[i] + dx
                df[i] = (f(xr) - f(xl)) / (2.*dx)
            return df
    return _df


def hessian(f, dx=1e-5):
    '''
    Creates a function that calculates the hessian matrix of a scalar field.

    This function takes as a parameter a scalar function and creates a new
    function that is able to calculate the second derivative (in case of single
    variable functions) or the hessian matrix (in case of multivariable
    functions. Please, note that this function takes as a parameter a
    *function*, and returns as a result *another function*. Calling the returned
    function on a point will give the hessian matrix of the original function
    at that point::

      >>> def f(x):
              return x^4

      >>> ddf = hessian(f)
      >>> ddf(1)
      12

    In the above example, ``ddf`` is a generated function which will return the
    result of the expression ``12*x**2``, the second derivative of the original
    function. In the case ``f`` is a multivariable function, it is assumed that
    its argument is a line vector.

    :Parameters:
      f
        Any function, one- or multivariable. The function must be an scalar
        function, though there is no checking at the moment the function is
        created. If ``f`` is not an scalar function, an exception will be
        raised at the moment the returned function is used.
      dx
        Optional argument that gives the precision of the calculation. It is
        recommended that ``dx = sqrt(D)``, where ``D`` is the machine precision.
        It defaults to ``1e-5``, which usually gives a good estimate.

    :Returns:
      A new function which, upon calling, gives the second derivative or hessian
      matrix of the original function on the analised point. The parameter of
      the returned function is a real number or a line vector where the hessian
      should be calculated.
    '''
    def _hf(x):
        try:
            x = float(x)
            return (f(x+dx) - 2*f(x) + f(x-dx)) / (4.*dx*dx)
        except TypeError:
            n = x.size
            hf = zeros((n, n))
            for i in range(n):
                for j in range(n):
                    xll = array(x)
                    xll[i] = xll[i] - dx
                    xll[j] = xll[j] - dx
                    xul = array(x)
                    xul[i] = xul[i] - dx
                    xul[j] = xul[j] + dx
                    xlr = array(x)
                    xlr[i] = xlr[i] + dx
                    xlr[j] = xlr[j] - dx
                    xur = array(x)
                    xur[i] = xur[i] + dx
                    xur[j] = xur[j] + dx
                    hf[i, j] = (f(xur) - f(xlr) - f(xul) + f(xll)) / (4.*dx*dx)
            return hf
    return _hf


################################################################################
# Base classes
################################################################################
class Optimizer(object):
    '''
    Base class for all optimizers.

    This class does nothing, and shouldn't be instantiated. Its only purpose is
    to serve as a template (or interface) to implemented optimizers. To create
    your own optimizer, subclass this.

    This class defines 3 methods that should be present in any subclass. They
    are defined here:

      __init__
        Initializes the optimizer. There are three usual parameters in this
        method, which signature should be::

        __init__(self, f, ..., emax=1e-8, imax=1000)

        where:
          - ``f`` is the cost function to be minimized;
          - ``...`` represent additional configuration of the optimizer, and it
            is dependent of the technique implemented;
          - ``emax`` is the maximum allowed error. The default value above is
            only a suggestion;
          - ``imax`` is the maximum number of iterations of the method. The
            default value above is only a suggestions.

      step
        This method should take an estimate and calculate the next, possibly
        better, estimate. Notice that the next estimate is strongly dependent of
        the method, the optimizer state and configuration, and two calls to this
        method with the same estimate might not give the same results. The
        method signature is::

          step(self, x)

        and the implementation should keep track of all the needed parameters.
        The method should return a tuple ``(x, e)`` with the new estimate of the
        solution and the estimate of the error.

      __call__
        This method should take an estimate and iterate the optimizer until one
        of the stop criteria is met: either less than the maximum error or more
        than the maximum number of iterations. Error is usually calculated as an
        estimate using the previous estimate, but any technique might be used.
        Use a counter to keep track of the number of iterations. The method
        signature is::

          __call__(self, x)

        and the implementation should keep track of all the needed parameters.
        The method should return a tuple ``(x, e)`` with the final estimate of
        the solution and the estimate of the error.
    '''
    def __init__(self, f=None, emax=1e-8, imax=1000):
        pass

    def step(self, x):
        pass

    def __call__(self, x):
        pass


################################################################################
# Test
if __name__ == "__main__":
    pass