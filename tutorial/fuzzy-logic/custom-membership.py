################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: tutorial/custom-membership.py
# Using custom membership functions
################################################################################

# Please, for more information on this demo, see the tutorial documentation.


# First, we import the needed modules
from numpy import *
import peach as p


# Peach can work with custom membership functions, if you need to use them.
# There are a number of ways of doing that. Please, use this file as a template
# to create your own.

# An existing membership function can be customized during its instantiation.
# For example, you can set parameters of an increasing ramp, starting in x = -1,
# and ending in x = 1, you can use the simple command:
CustomMembershipFunction1 = p.IncreasingRamp(-1., 1.)

# You can also create your membership function as a simple function, and turn it
# into a membership function. Let's use the ramp example as above. You can
# create a simple membership function like this:
def custom_ramp(x):
    if x < -1. : return 0.
    elif x > 1.: return 1.
    else: return (x+1.)/2.

CustomMembershipFunction2 = p.Membership(custom_ramp)

# The last way to create a membership function is by subclassing Membership.
# To do that, you will have to implement the __init__ and __call__ methods. Use
# the code below (where we implement, again, a ramp) as a template:
class CustomMembershipFunction3(p.Membership):
    '''
    Don't forget to document your code!
    '''
    def __init__(self):
        '''
        We won't pass any parameter to the initializer of the class, since we
        don't want further customization.
        '''
        p.Membership.__init__(self)

    def __call__(self, x):
        '''
        The __call__ interface should receive a (vector of) scalar and return a
        scalar. Remember that activation functions should be able to deal with
        vectors, if needed, so using the ``numpy`` functions will really help!
        Please consult the numpy documentation to understand what ``select``
        does.
        '''
        s = select([ x < -1., x < 1. ], [ 0., (x+1.)/2. ], 1.)
        return FuzzySet(s)

# Notice that the __call__ interface should return a FuzzySet object!
# The functions thus generated can be used in any place where a membership
# function or a membership class would be used, such as in a controller.