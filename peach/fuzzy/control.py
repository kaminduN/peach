################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/control.py
# Fuzzy based controllers, or fuzzy inference systems
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This package implements fuzzy controllers, of fuzzy inference systems.

There are two types of controllers implemented in this package. The Mamdani
controller is the traditional approach, where input (or controlled) variables
are fuzzified, a set of decision rules determine the outcome in a fuzzified way,
and a defuzzification method is applied to obtain the numerical result.

The Sugeno controller operates in a similar way, but there is no defuzzification
step. Instead, the value of the output (or manipulated) variable is determined
by parametric models, and the final result is determined by a weighted average
based on the decision rules. This type of controller is also known as parametric
controller.
"""


################################################################################
import numpy
from numpy import zeros, array, dot
import types

from base import *
from mf import *
from norms import *
from defuzzy import *


################################################################################
# Basic Mamdani controller
################################################################################
class Controller(object):
    '''
    Basic Mamdani controller

    This class implements a standard Mamdani controller. A controller based on
    fuzzy logic has a somewhat complex behaviour, so it is not explained here.
    There are numerous references that can be consulted.

    It is essential to understand the format that decision rules must follow to
    obtain correct behaviour of the controller. A rule is a tuple given by::

        ((mx0, mx1, ..., mxn), my)

    where ``mx0`` is a membership function of the first input variable, ``mx1``
    is a membership function of the second input variable and so on; and ``my``
    is a membership function or a fuzzy set of the output variable.

    Notice that ``mx``'s are *functions* not fuzzy sets! They will be applied to
    the values of the input variables given in the function call, so, if they
    are anything different from a membership function, an exception will be
    raised. Please, consult the examples to see how they must be used.
    '''
    def __init__(self, yrange, rules=[], defuzzy=Centroid,
                 norm=ZadehAnd, conorm=ZadehOr, negation=ZadehNot,
                 imply=MamdaniImplication, aglutinate=MamdaniAglutination):
        '''
        Creates and initialize the controller.

        :Parameters:
          yrange
            The range of the output variable. This must be given as a set of
            points belonging to the interval where the output variable is
            defined, not only the start and end points. It is strongly suggested
            that the interval is divided in some (eg.: 100) points equally
            spaced;
          rules
            The set of decision rules, as defined above. If none is given, an
            empty set of rules is assumed;
          defuzzy
            The defuzzification method to be used. If none is given, the
            Centroid method is used;
          norm
            The norm (``and`` operation) to be used. Defaults to Zadeh and.
          conorm
            The conorm (``or`` operation) to be used. Defaults to Zadeh or.
          negation
            The negation (``not`` operation) to be used. Defaults to Zadeh not.
          imply
            The implication method to be used. Defaults to Mamdani implication.\
          aglutinate
            The aglutination method to be used. Defaults to Mamdani
            aglutination.
        '''
        self.__y = yrange
        self.__rules = [ ]
        if isinstance(rules, list):
            for r in rules:
                self.add_rule(r)
        self.defuzzify = defuzzy
        self.__AND__ = norm
        self.__OR__ = conorm
        self.__NOT__ = negation
        self.__IMP__ = imply
        self.__AGL__ = aglutinate


    def __gety(self):
        return self.__y
    y = property(__gety, None)
    '''Property that returns the output variable interval. Not writable'''

    def __getrules(self):
        return self.__rules[:]
    rules = property(__getrules, None)
    '''Property that returns the list of decision rules. Not writable'''

    def set_norm(self, f):
        '''
        Sets the norm (``and``) to be used.

        This method must be used to change the behavior of the ``and`` operation
        of the controller.

        :Parameters:
          f
            The function can be any function that takes two numerical values and
            return one numerical value, that corresponds to the ``and`` result.
        '''
        if isinstance(f, numpy.vectorize):
            self.__AND__ = f
        elif isinstance(f, types.FunctionType):
            self.__AND__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'


    def set_conorm(self, f):
        '''
        Sets the conorm (``or``) to be used.

        This method must be used to change the behavior of the ``or`` operation
        of the controller.

        :Parameters:
          f
            The function can be any function that takes two numerical values and
            return one numerical value, that corresponds to the ``or`` result.
        '''
        if isinstance(f, numpy.vectorize):
            self.__OR__ = f
        elif isinstance(f, types.FunctionType):
            self.__OR__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'


    def set_negation(self, f):
        '''
        Sets the negation (``not``) to be used.

        This method must be used to change the behavior of the ``not`` operation
        of the controller.

        :Parameters:
          f
            The function can be any function that takes one numerical value and
            return one numerical value, that corresponds to the ``not`` result.
        '''
        if isinstance(f, numpy.vectorize):
            self.__NOT__ = f
        elif isinstance(f, types.FunctionType):
            self.__NOT__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'


    def set_implication(self, f):
        '''
        Sets the implication to be used.

        This method must be used to change the behavior of the implication
        operation of the controller.

        :Parameters:
          f
            The function can be any function that takes two numerical values and
            return one numerical value, that corresponds to the implication
            result.
        '''
        if isinstance(f, numpy.vectorize):
            self.__IMP__ = f
        elif isinstance(f, types.FunctionType):
            self.__IMP__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'


    def set_aglutination(self, f):
        '''
        Sets the aglutination to be used.

        This method must be used to change the behavior of the aglutination
        operation of the controller.

        :Parameters:
          f
            The function can be any function that takes two numerical values and
            return one numerical value, that corresponds to the aglutination
            result.
        '''
        if isinstance(f, numpy.vectorize):
            self.__AGL__ = f
        elif isinstance(f, types.FunctionType):
            self.__AGL__ = numpy.vectorize(f)
        else:
            raise ValueError, 'invalid function'


    def add_rule(self, rule):
        '''
        Adds a decision rule to the knowledge base.

        It is essential to understand the format that decision rules must follow
        to obtain correct behaviour of the controller. A rule is a tuple must
        have the following format::

        ((mx0, mx1, ..., mxn), my)

        where ``mx0`` is a membership function of the first input variable,
        ``mx1`` is a membership function of the second input variable and so on;
        and ``my`` is a membership function or a fuzzy set of the output
        variable.

        Notice that ``mx``'s are *functions* not fuzzy sets! They will be
        applied to the values of the input variables given in the function call,
        so, if they are anything different from a membership function, an
        exception will be raised when the controller is used. Please, consult
        the examples to see how they must be used.
        '''
        mx, my = rule
        for m in mx:
            if not (isinstance(m, Membership) or m is None):
                raise ValueError, 'condition not a membership function'
        if isinstance(my, Membership):
            rule = (mx, my(self.__y))
        elif not isinstance(my, FuzzySet):
            raise ValueError, 'consequent not a fuzzy set or membership function'
        self.__rules.append(rule)


    def add_table(self, lx1, lx2, table):
        '''
        Adds a table of decision rules in a two variable controller.

        Typically, fuzzy controllers are used to control two variables. In that
        case, the set of decision rules are given in the form of a table, since
        that is a more compact format and very easy to visualize. This is a
        convenience function that allows to add decision rules in the form of a
        table. Notice that the resulting knowledge base will be the same if this
        function is used or the ``add_rule`` method is used with every single
        rule. The second method is in general easier to read in a script, so
        consider well.

        :Parameters:
          lx1
            The set of membership functions to the variable ``x1``, or the
            lines of the table
          lx2
            The set of membership functions to the variable ``x2``, or the
            columns of the table
          table
            The consequent of the rule where the condition is the line ``and``
            the column. These can be the membership functions or fuzzy sets.
        '''
        for i in range(len(lx1)):
            for j in range(len(lx2)):
                my = table[i][j]
                if my is not None:
                    self.add_rule(((lx1[i], lx2[j]), my))


    def eval(self, r, xs):
        '''
        Evaluates one decision rule in this controller

        Takes a rule from the controller and evaluates it given the values of
        the input variables.

        :Parameters:
          r
            The rule in the standard format, or an integer number. If ``r`` is
            an integer, then the ``r`` th rule in the knowledge base will be
            evaluated.
          xs
            A tuple, a list or an array containing the values of the input
            variables. The dimension must be coherent with the given rule.

        :Returns:
          This method evaluates each membership function in the rule for each
          given value, and ``and`` 's the results to obtain the condition. If
          the condition is zero, a tuple ``(0.0, None) is returned. Otherwise,
          the condition is ``imply`` ed in the membership function of the output
          variable. A tuple containing ``(condition, imply)`` (the membership
          value associated to the condition and the result of the implication)
          is returned.
        '''
        if type(r) is types.IntType:
            r = self.__rules[r]
        mx, my = r
        # Finds the membership value for each xn
        cl = [ m(x) for m, x in zip(mx, xs) if m is not None ]
        # Apply the ``and`` operation
        mr = reduce(lambda x0, x1: self.__AND__(x0, x1), cl)
        # Implication, unnecessary if mr == 0
        if mr == 0.0:
            return (0.0, None)
        else:
            return (mr, self.__IMP__(mr, my))


    def eval_all(self, *xs):
        '''
        Evaluates all the rules and aglutinates the results.

        Given the values of the input variables, evaluate and apply every rule
        in the knowledge base (with the ``eval`` method) and aglutinates the
        results.

        :Parameters:
          xs
            A tuple, a list or an array with the values of the input variables.

        :Returns:
          A fuzzy set containing the result of the evaluation of every rule in
          the knowledge base, with the results aglutinated.
        '''
        ry = FuzzySet(zeros(self.__y.shape))
        for r in self.__rules:
            mr, iy = self.eval(r, xs)
            if mr != 0.0:
                ry = self.__AGL__(ry, iy)
        return ry


    def __call__(self, *xs):
        '''
        Apply the controller to the set of input variables

        Given the values of the input variables, evaluates every decision rule,
        aglutinates the results and defuzzify it. Returns the response of the
        controller.

        :Parameters:
          xs
            A tuple, a list or an array with the values of the input variables.

        :Returns:
          The response of the controller.
        '''
        ry = self.eval_all(*xs)
        return self.defuzzify(ry, self.__y)


class Mamdani(Controller):
    '''``Mandani`` is an alias to ``Controller``'''
    pass


################################################################################
# Basic Takagi-Sugeno controller
################################################################################
class Parametric(object):
    '''
    Basic Parametric controller

    This class implements a standard parametric (or Takagi-Sugeno) controller. A
    controller based on fuzzy logic has a somewhat complex behaviour, so it is
    not explained here. There are numerous references that can be consulted.

    It is essential to understand the format that decision rules must follow to
    obtain correct behaviour of the controller. A rule is a tuple given by::

        ((mx0, mx1, ..., mxn), (a0, a1, ..., an))

    where ``mx0`` is a membership function of the first input variable, ``mx1``
    is a membership function of the second input variable and so on; and ``a0``
    is the linear parameter, ``a1`` is the parameter associated with the first
    input variable, ``a2`` is the parameter associated with the second input
    variable and so on. The response to the rule is calculated by::

        y = a0 + a1*x1 + a2*x2 + ... + an*xn

    Notice that ``mx``'s are *functions* not fuzzy sets! They will be applied to
    the values of the input variables given in the function call, so, if they
    are anything different from a membership function, an exception will be
    raised. Please, consult the examples to see how they must be used.
    '''
    def __init__(self, rules = [], norm=ProbabilisticAnd,
                 conorm=ProbabilisticOr, negation=ProbabilisticNot):
        '''
        Creates and initializes the controller.

        :Parameters:
          rules
            List containing the decision rules for the controller. If not given,
            an empty set of decision rules is used.
          norm
            The norm (``and`` operation) to be used. Defaults to Probabilistic
            and.
          conorm
            The conorm (``or`` operation) to be used. Defaults to Probabilistic
            or.
          negation
            The negation (``not`` operation) to be used. Defaults to
            Probabilistic not.
        '''
        self.__rules = [ ]
        if isinstance(rules, list):
            for r in rules:
                self.add_rules(r)
        self.__AND__ = norm
        self.__OR__ = conorm
        self.__NOT__ = negation


    def __getrules(self):
        return self.__rules[:]
    rules = property(__getrules, None)
    '''Property that returns the list of decision rules. Not writable'''


    def add_rule(self, rule):
        '''
        Adds a decision rule to the knowledge base.

        It is essential to understand the format that decision rules must follow
        to obtain correct behaviour of the controller. A rule is a tuple given
        by::

            ((mx0, mx1, ..., mxn), (a0, a1, ..., an))

        where ``mx0`` is a membership function of the first input variable,
        ``mx1`` is a membership function of the second input variable and so on;
        and ``a0`` is the linear parameter, ``a1`` is the parameter associated
        with the first input variable, ``a2`` is the parameter associated with
        the second input variable and so on.

        Notice that ``mx``'s are *functions* not fuzzy sets! They will be
        applied to the values of the input variables given in the function call,
        so, if they are anything different from a membership function, an
        exception will be raised. Please, consult the examples to see how they
        must be used.
        '''
        mx, a = rule
        for m in mx:
            if not (isinstance(m, Membership) or m is None):
                raise ValueError, 'condition not a membership function'
        a = array(a, dtype=float)
        rule = (mx, a)
        self.__rules.append(rule)


    def eval(self, r, xs):
        '''
        Evaluates one decision rule in this controller

        Takes a rule from the controller and evaluates it given the values of
        the input variables. The format of the rule is as given, and the
        response to the rule is calculated by::

            y = a0 + a1*x1 + a2*x2 + ... + an*xn

        :Parameters:
          r
            The rule in the standard format, or an integer number. If ``r`` is
            an integer, then the ``r`` th rule in the knowledge base will be
            evaluated.
          xs
            A tuple, a list or an array containing the values of the input
            variables. The dimension must be coherent with the given rule.

        :Returns:
          This method evaluates each membership function in the rule for each
          given value, and ``and`` 's the results to obtain the condition. If
          the condition is zero, a tuple ``(0.0, 0.0) is returned. Otherwise,
          the result as given above is calculate, and a tuple containing
          ``(condition, result)`` (the membership value associated to the
          condition and the result of the calculation) is returned.
        '''
        if type(r) is types.IntType:
            r = self.__rules[r]
        mx, a = r
        # Finds the membership value for each xn
        cl = [ m(x) for m, x in zip(mx, xs) if m is not None ]
        # Apply ``and`` operation
        mr = reduce(lambda x0, x1: self.__AND__(x0, x1), cl)
        # Implication, returns 0.0 if mr == 0
        if mr > 0.0:
            return (mr, dot(a, xs))
        else:
            return (0.0, 0.0)


    def __call__(self, *xs):
        '''
        Apply the controller to the set of input variables

        Given the values of the input variables, evaluates every decision rule,
        and calculates the weighted average of the results. Returns the response
        of the controller.

        :Parameters:
          xs
            A tuple, a list or an array with the values of the input variables.

        :Returns:
          The response of the controller.
        '''
        ys = array([ self.eval(r, xs) for r in self.__rules ])
        m = ys[:, 0]
        y = ys[:, 1]
        return sum(m*y) / sum(m)


class Sugeno(Parametric):
    '''``Sugeno`` is an alias to ``Parametric``'''
    pass


################################################################################
# Test
if __name__ == "__main__":
    pass
