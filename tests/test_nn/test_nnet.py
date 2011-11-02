#! /usr/bin/python
#-*- coding:utf-8 -*-

import unittest
from numpy import array

class Test_GRNN(unittest.TestCase):
    samples = array([0.000000, 0.111111, 0.222222, 0.333333, 0.444444, 
                    0.555556, 0.666667, 0.777778, 0.888889, 1.000000])
    
    targets = array([0.349486, 0.830839, 1.007332, 0.971507, 0.133066, 
                     0.166823, -0.848307, -0.445686, -0.563567, 0.261502])

    samples2d = array([[0.03554131, 0.03554131], [0.03554131, 0.07108261],
                      [0.03554131, 0.10662392], [0.07108261, 0.03554131],
                      [0.07108261, 0.07108261], [0.07108261, 0.10662392],
                      [0.10662392, 0.03554131], [0.10662392, 0.07108261],
                      [0.10662392, 0.10662392]])
    
    targets2d = array([0.34119655, 0.30210111, 0.29143872, 0.35541307, 
                       0.32698002, 0.28433045, 0.34119655, 0.3127635, 
                       0.27011393])

    def _getTargetClass(self, *args, **kwargs):
        from peach.nn.nnet import GRNN
        return GRNN(*args, **kwargs)

    def test_init(self):
        grnn = self._getTargetClass()
        assert grnn.sigma == 0.1

    def test_kernel(self):
        grnn = self._getTargetClass(sigma=0.1)
        self.assertAlmostEqual(grnn._kernel(0.5, 0.2), 0.01110899)

    def test_kernelVectorInput(self):
        grnn = self._getTargetClass(sigma=0.1)
        x1 = array([0.3, 0.3])
        x2 = array([0.5, 0.7])
        self.assertAlmostEqual(grnn._kernel(x1, x2), 0.0000453999, places=10)

    def test_train(self):
        grnn = self._getTargetClass()
        grnn.train(self.samples, self.targets)

        assert grnn._samples[9] == 1
        self.assertAlmostEqual(grnn._targets[2], 1.007332, places=6)

    def test_call(self):
        grnn = self._getTargetClass()
        grnn._samples = self.samples.copy()
        grnn._targets = self.targets.copy()
        self.assertAlmostEqual(grnn(0.25), 0.891242, places=6)
        self.assertAlmostEqual(grnn(0.63), -0.3545186, places=7)

    def test_callVectorInput(self):
        grnn =self._getTargetClass()
        grnn._samples = self.samples2d.copy()
        grnn._targets = self.targets2d.copy()
        self.assertAlmostEqual(grnn([0.05, 0.02]), 0.3179468)


class Test_PNN(unittest.TestCase):
    trainSet = [
        [array([0, 0]), 0],
        [array([0, 1]), 1],
        [array([1, 0]), 1],
        [array([1, 1]), 0]
    ]

    def _getTargetClass(self, *args, **kwargs):
        from peach.nn.nnet import PNN
        return PNN(*args, **kwargs)
    
    def test_init(self):
        pnn = self._getTargetClass()
        assert pnn.sigma == 0.1

    def test_kernel(self):
        pnn = self._getTargetClass(sigma=0.1)
        self.assertAlmostEqual(pnn._kernel(0.5, 0.2), 0.01110899)

    def test_kernelVectorInput(self):
        pnn = self._getTargetClass(sigma=0.1)
        x1 = array([0.3, 0.3])
        x2 = array([0.5, 0.7])
        self.assertAlmostEqual(pnn._kernel(x1, x2), 0.0000453999, places=10)

    def test_train(self):
        pnn = self._getTargetClass()
        pnn.train(self.trainSet)

        assert len(pnn._categorys[0]) == 2
        assert len(pnn._categorys[1]) == 2
    
    def test_call(self):
        pnn = self._getTargetClass()
        pnn._categorys = {0: [array([0, 0]), array([1, 1])], 
                          1: [array([0, 1]), array([1, 0])]}
        
        assert pnn([0, 0]) == 0
        assert pnn([0, 1]) == 1
        assert pnn([1, 0]) == 1
        assert pnn([1, 1]) == 0

        assert pnn([0.2, 0.1]) == 0
        assert pnn([0, 0.6]) == 1


if __name__ == '__main__':
    unittest.main()
