#! /usr/bin/python
#-*- coding:utf-8 -*-

import unittest

class Test_Activation(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import Activation
        return Activation

    def test_customFunction_activation(self):
        customFunction = self._getTargetClass()(lambda x:'activation')
        assert customFunction(None) == 'activation'

    def test_customFunction_derivative(self):
        customFunction = self._getTargetClass()(df=lambda:'derivative')
        assert customFunction.derivative() == 'derivative'


class Test_Threshold(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import Threshold
        return Threshold

    def test_alias(self):
        from peach.nn.af import Step
        assert Step == self._getTargetClass()

    def test_init(self):
        function = self._getTargetClass()()
        assert function(-1) == 0
        assert function(0) == 1

    def test_activation(self):
        function = self._getTargetClass()(1, 5)
        assert function(0) == 0
        assert function(1) == 5

    def test_activationVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function(array([-1, 0, 1])) == array([0, 1, 1])
        assert result.all()

    def test_derivative(self):
        function = self._getTargetClass()()
        assert function.derivative(2) == 1

    def test_derivativeVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function.derivative(array([1, 2, 3, 4])) == array([1, 1, 1, 1])
        assert result.all()


class Test_Linear(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import Linear
        return Linear

    def test_alias(self):
        from peach.nn.af import Identity
        assert Identity == self._getTargetClass()

    def test_activation(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function(3) == array([3.])
        assert result.all()

    def test_activationVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function(array([-1, 0, 1])) == array([-1, 0, 1])
        assert result.all()

    def test_derivative(self):
        function = self._getTargetClass()()
        assert function.derivative(5) == 1

    def test_derivativeVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function.derivative(array([1, 2, 3, 4])) == array([1, 1, 1, 1])
        assert result.all()


class Test_Ramp(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import Ramp
        return Ramp

    def test_activation(self):
        function = self._getTargetClass()((-1, -1), (1, 1))
        assert function(-2) == -1
        assert function(0.5) == 0.5
        assert function(2) == 1

    def test_activationVector(self):
        from numpy import array
        function = self._getTargetClass()((-1, -1), (1, 1))
        result = function(array([-2, 0, 2])) == array([-1, 0, 1])
        assert result.all()

    def test_derivative(self):
        function = self._getTargetClass()()
        assert function.derivative(-1) == 0
        assert function.derivative(0.1) == 1
        assert function.derivative(1) == 0

    def test_derivativeVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function.derivative(array([-1, 0.1, 1])) == array([0, 1, 0])
        assert result.all()


class Test_Sigmoid(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import Sigmoid
        return Sigmoid

    def test_alias(self):
        from peach.nn.af import Logistic
        assert Logistic == self._getTargetClass()

    def test_activation(self):
        function = self._getTargetClass()()
        assert function(0) == 0.5
        self.assertAlmostEquals(function(-1), 0.268941421)
        self.assertAlmostEquals(function(1), 0.731058578)

    def test_activationVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function(array([0, 0])) == array([0.5, 0.5])
        assert result.all()

    def test_derivative(self):
        function = self._getTargetClass()()
        assert function.derivative(0) == 0.25
        self.assertAlmostEquals(function.derivative(-1), 0.196611933)
        self.assertAlmostEquals(function.derivative(1), 0.196611933)

    def test_derivativeVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function.derivative(array([0, 0])) == array([0.25, 0.25])
        assert result.all()


class Test_Signum(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import Signum
        return Signum

    def test_activation(self):
        function = self._getTargetClass()()
        assert function(0) == 0
        assert function(-2) == -1
        assert function(2) == 1

    def test_activationVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function(array([-2, -1, 0, 1, 2])) == array([-1, -1, 0, 1, 1])
        assert result.all()

    def test_derivative(self):
        function = self._getTargetClass()()
        assert function.derivative(2) == 1

    def test_derivativeVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function.derivative(array([1, 2, 3, 4])) == array([1, 1, 1, 1])
        assert result.all()


class Test_ArcTan(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import ArcTan
        return ArcTan

    def test_activation(self):
        function = self._getTargetClass()()
        assert function(-1) == -0.25
        assert function(0) == 0
        assert function(1) == 0.25

    def test_activationVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function(array([-1, 0, 1])) == array([-0.25, 0, 0.25])
        assert result.all()

    def test_derivative(self):
        function = self._getTargetClass()()
        self.assertAlmostEquals(function.derivative(-1), 0.15915494309189)
        self.assertAlmostEquals(function.derivative(0), 0.31830988618379)
        self.assertAlmostEquals(function.derivative(1), 0.15915494309189)

    def test_derivativeVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function.derivative(array([-1, 0, 1]))
        self.assertAlmostEquals(result[0], 0.15915494309189)
        self.assertAlmostEquals(result[1], 0.31830988618379)
        self.assertAlmostEquals(result[2], 0.15915494309189)


class Test_TanH(unittest.TestCase):
    def _getTargetClass(self):
        from peach.nn.af import TanH
        return TanH

    def test_activation(self):
        function = self._getTargetClass()()
        assert function(0) == 0
        self.assertAlmostEquals(function(1), 0.7615941559)
        self.assertAlmostEquals(function(-1), -0.7615941559)

    def test_activationVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function(array([0, 0])) == array([0, 0])
        assert result.all()

    def test_derivative(self):
        function = self._getTargetClass()()
        assert function.derivative(0) == 1
        self.assertAlmostEquals(function.derivative(1), 0.41997434161)
        self.assertAlmostEquals(function.derivative(-1), 0.41997434161)

    def test_derivativeVector(self):
        from numpy import array
        function = self._getTargetClass()()
        result = function.derivative(array([0, 0])) == array([1, 1])
        assert result.all()


if __name__ == '__main__':
    unittest.main()
