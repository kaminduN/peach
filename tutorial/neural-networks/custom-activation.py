################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/custom-activation.py
# Using custom activation functions
################################################################################

# Please, for more information on this demo, see the tutorial documentation.


# First, we import the needed modules
from numpy import *
import peach as p


# Peach can work with custom activation functions, if you need to use them.
# There are a number of ways of doing that. Please, use this file as a template
# to create your own.

# An existing activation function can be customized during its instantiation.
# For example, if you want to use a diferent ramp, starting in (-1, -1) and
# ending in (1, 1), you can use the simple command:
CustomActivationFunction1 = p.Ramp((-1., -1.), (1., 1.))

# You can also create your activation function as a simple function, and turn it
# into an activation function. Let's use the ramp example as above. You can
# create a simple activation function like this:
def custom_ramp(x):
    if x < -1. : return -1.
    elif x > 1.: return 1.
    else: return x

CustomActivationFunction2 = p.Activation(custom_ramp)

# But, please, notice that the derivative for a function create as above will
# be estimated. While it is not a problem for a ramp function, it might be a
# problem with diferent functions, and it can be less efficient too.

# The last way to create an activation function is by subclassing Activation.
# To do that, you will have to implement the __init__, __call__ and derivative
# methods. Use the code below (where we implement, again, a ramp) as a template:
class CustomActivationFunction3(p.Activation):
    '''
    Don't forget to document your code!
    '''
    def __init__(self):
        '''
        We won't pass any parameter to the initializer of the class, since we
        don't want further customization.
        '''
        p.Activation.__init__(self)

    def __call__(self, x):
        '''
        The __call__ interface should receive a (vector of) scalar and return a
        scalar. Remember that activation functions should be able to deal with
        vectors, if needed, so using the ``numpy`` functions will really help!
        Please consult the numpy documentation to understand what ``select``
        does.
        '''
        return select([ x < -1., x < 1. ], [ -1., x ], 1.)

    def derivative(self, x, dx=1.e-5):
        '''
        The derivative of your function must be implemented in this method,
        because a lot of the convergence methods use it. The method should
        receive a (vector of) scalar and return a scalar. The second parameter
        will be the precision of the derivative, and it is seldom used. It is
        a good measure to put it as a named parameter, just to make it sure.
        '''
        return select([ x < -1., x < 1. ], [ 0., 1. ], 0.)


# The functions thus generated can be used in any place where an activation
# function or an activation class would be used.