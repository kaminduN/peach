Using Custom Membership Functions
=================================

The aim of this tutorial is to show how to create your own membership functions
to use with the fuzzy logic package. Almost all of the most used membership
functions are already programmed within Peach, so you will seldom need this use,
but in case you want to try something different to see what happens, take a look
in this tutorial.

This tutorial won't take into account that you are using the command line, but
what is said here will work there, of course. The first thing to do is to be
sure that the modules are imported::

    import numpy
    import peach

Peach can work with custom membership functions, if you need to use them. There
are a number of ways of doing that. Please, use this file as a template to
create your own.

The first thing you should try is to use an existing membership function, since
those can be customized during its instantiation. For example, there is a
``IncreasingRamp`` class that implements an increasing ramp function. If you
want to use this kind of ramp, just configure the correct parameters. Suppose
the ramp you need starts in -1 and ends in 1. Then you can use the simple
command::

    CustomMembershipFunction1 = peach.IncreasingRamp(-1., 1.)

If you want something completelly different from what is implemented, you can
also create your membership function as a simple function, and turn it into a
membership function. Let's use the ramp example as above. You can create a
simple membership function like this::

    def custom_ramp(x):
        if x < -1. : return 0.
        elif x > 1.: return 1.
        else: return (x+1.)/2.

    CustomMembershipFunction2 = peach.Membership(custom_ramp)

The last way to create an activation function is by subclassing ``Membership``.
To do that, you will have to implement the ``__init__`` and ``__call__``
methods. Use the code below (where we implement, again, a ramp) as a
template::

    class CustomMembershipFunction3(peach.Membership):
        '''
        Don't forget to document your code!
        '''
        def __init__(self):
            '''
            We won't pass any parameter to the initializer of the class, since we
            don't want further customization.
            '''
            peach.Membership.__init__(self)

        def __call__(self, x):
            '''
            The __call__ interface should receive a (vector of) scalar and return a
            scalar. Remember that activation functions should be able to deal with
            vectors, if needed, so using the ``numpy`` functions will really help!
            Please consult the numpy documentation to understand what ``select``
            does.
            '''
            s = peach.select([ x < -1., x < 1. ], [ 0., (x+1.)/2. ], 1.)
            return peach.FuzzySet(s)

The functions thus generated can be used in any place where a membership
function or a membership class would be used, for example, in a fuzzy logic
based controller.