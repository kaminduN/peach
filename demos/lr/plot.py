# -*- coding: utf-8 -*-
################################################################################
# General graph plotting widget for the PyQt4 toolkit
# Some of this was taken from a page that I didn't register the link to.
# probably the Qwt page itself.
# Jose Alexandre Nalon
#
# Date: 28-01-2008
# Graphic plotting
################################################################################


################################################################################
# Used modules
################################################################################
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.Qt import QEvent
import PyQt4.Qwt5 as Qwt

# NumPy is essential to plotting.
from numpy import *


################################################################################
# Classes
################################################################################
class LRPlotWindow(Qwt.QwtPlot):

    def __init__(self, xlim, ylim, *args):
        '''
        Initializes the graph plotting. The usual parameters are available.

        :Parameters:
          nplots
            Number of plots in the same window.
        '''
        Qwt.QwtPlot.__init__(self, *args)
        self.set_scale(xlim, ylim)

        self.setCanvasBackground(Qt.white)
        grid = Qwt.QwtPlotGrid()
        grid.attach(self)
        grid.setMajPen(QPen(Qt.black, 0, Qt.DotLine))

        self.lr = Qwt.QwtPlotCurve('')
        self.lr.attach(self)
        self.lr.setPen(QPen(Qt.darkYellow))
        self.lr.setRenderHint(Qwt.QwtPlotItem.RenderAntialiased)
        
        scatter_symbol = Qwt.QwtSymbol(Qwt.QwtSymbol.Ellipse,
            QBrush(Qt.white), QPen(Qt.darkCyan), QSize(9, 9))
        self.scatter = Qwt.QwtPlotCurve('')
        self.scatter.attach(self)
        self.scatter.setPen(QPen(Qt.NoPen))
        self.scatter.setSymbol(scatter_symbol)
        self.scatter.setRenderHint(Qwt.QwtPlotItem.RenderAntialiased)

        example_symbol = Qwt.QwtSymbol(Qwt.QwtSymbol.Ellipse,
            QBrush(Qt.red), QPen(Qt.red), QSize(9, 9))
        self.example = Qwt.QwtPlotCurve('')
        self.example.attach(self)
        self.example.setPen(QPen(Qt.NoPen))
        self.example.setSymbol(example_symbol)
        self.example.setRenderHint(Qwt.QwtPlotItem.RenderAntialiased)

    def set_scale(self, xlim, ylim):
        '''
        Set boundaries of the plots
        '''
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.setAxisScale(2, self.xmin, self.xmax)
        self.setAxisScale(0, self.ymin, self.ymax)

    def setLRData(self, x, y):
        '''
        Sets data on the line plot
        
        :Parameters:
          x
            horizontal data
          y
            vertical data
        '''
        x = array(x)
        y = array(y)
        self.lr.setData(x, y)
        #self.replot()

    def setScatterData(self, x, y):
        '''
        Sets data on the scatter plot
        
        :Parameters:
          x
            horizontal data
          y
            vertical data
        '''
        x = array(x)
        y = array(y)
        self.scatter.setData(x, y)
        #self.replot()

    def setExampleData(self, x, y):
        '''
        Sets data on the example plot
        
        :Parameters:
          x
            horizontal data
          y
            vertical data
        '''
        x = array([ x, x ])
        y = array([ y, y ])
        self.example.setData(x, y)
        #self.replot()

        
################################################################################
class PlotWindow(Qwt.QwtPlot):

    def __init__(self, *args):
        '''
        Initializes the graph plotting. The usual parameters are available.

        :Parameters:
          nplots
            Number of plots in the same window.
        '''
        Qwt.QwtPlot.__init__(self, *args)

        self.setCanvasBackground(Qt.white)
        grid = Qwt.QwtPlotGrid()
        grid.attach(self)
        grid.setMajPen(QPen(Qt.black, 0, Qt.DotLine))

        self.curve = Qwt.QwtPlotCurve('')
        self.curve.attach(self)
        self.curve.setPen(QPen(Qt.darkYellow))
        self.curve.setRenderHint(Qwt.QwtPlotItem.RenderAntialiased)
        
    def setData(self, y):
        '''
        Sets data on the plots
        
        :Parameters:
          y
            Vertical data
        '''
        x = arange(0, len(y))
        y = array(y)
        self.curve.setData(x, y)
        self.replot()
        

################################################################################
