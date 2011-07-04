#-*- coding:utf-8 -*-

# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest


class  Test_LMS(unittest.TestCase):
    def _getRule(self):
        from peach.nn.lrules import LMS
        return LMS

    def test_init(self):
        rule = self._getRule()()
        assert rule.lrate == 0.05

    def test_alias(self):
        from peach.nn.lrules import WidrowHoff
        from peach.nn.lrules import DeltaRule
        rule = self._getRule()
        assert rule == WidrowHoff
        assert rule == DeltaRule

class  Test_Backpropagation(unittest.TestCase):
    def _getRule(self):
        from peach.nn.lrules import BackPropagation
        return BackPropagation

    def test_init(self):
        rule = self._getRule()()
        assert rule.lrate == 0.05


if __name__ == '__main__':
    unittest.main()

