#! /usr/bin/python
#-*- coding:utf-8 -*-

import unittest


class Test_Layer(unittest.TestCase):
    def _getLayer(self, *args, **kwargs):
        from peach.nn.base import Layer
        return Layer(*args, **kwargs)

    def test_init(self):
        layer = self._getLayer((5, 3), bias=True)
        assert layer.size == 5
        assert layer.inputs == 3
        assert layer.shape == (5, 3)
        assert layer.bias == True
        assert layer.weights.shape == (5, 4)

    def test_phi(self):
        from peach.nn.af import Linear
        phi = Linear()

        layer = self._getLayer((2, 1), phi=phi)
        assert layer.phi == phi

    def test_getitem(self):
        layer = self._getLayer((3, 2))
        result = layer[0] == layer.weights[0]
        assert result.all()

    def test_setitem(self):
        from numpy import array
        layer = self._getLayer((3, 2))
        layer[1] = array([1, 2])
        result = layer[1] == array([1, 2])
        assert result.all()

    def test_call(self):
        from numpy import array
        layer = self._getLayer((1, 2))
        layer[0] = 0.5, 0.5
        y = layer(array([2, 4])) # 3

        result = y == array([3])
        assert result.all()

        result = layer.y == array([3])
        assert result.all()

        result = layer.v == array([3])
        assert result.all()

    def test_callWithBias(self):
        from numpy import array
        layer = self._getLayer((1, 2), bias=True)
        layer[0] = 100, 0.5, 0.5
        y = layer(array([4, 6])) # 105

        result = y == array([105])
        assert result.all()

        result = layer.y == array([105])
        assert result.all()

        result = layer.v == array([105])
        assert result.all()


if __name__ == '__main__':
    unittest.main()

