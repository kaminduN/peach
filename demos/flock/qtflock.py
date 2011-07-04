# -*- coding: utf-8 -*-
################################################################################
# Widget to draw the flock.
# Jose Alexandre Nalon
#
# Date: 22-10-2007
################################################################################


################################################################################
# Used modules
################################################################################
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from numpy import *


################################################################################
# Classes
################################################################################
class FlockView(QGraphicsView):
    """
    Visualization of flock.
    """
    def __init__(self, spot, flock, *cnf):
        QGraphicsView.__init__(self, *cnf)
        self.__xsize = self.width() - 150
        self.__ysize = self.height() - 150
        self.__set_scale()
        self.__create_gs()
        self.__create_objects(len(flock))
        self.__spot = spot
        self.__flock = flock
        self.set_state(spot, flock)
        self.show()


    def __set_scale(self):
        xmin = -0.1
        ymin = -0.1
        xmax = 1.1
        ymax = 1.1
        self.__ay = - float(self.__ysize) / (ymax - ymin)
        self.__by = - self.__ay * ymax
        self.__ax = - self.__ay        # Same scale for x-axis
        self.__bx = - self.__ax * xmin


    def __create_gs(self):
        self.gs = QGraphicsScene(0, 0, self.__xsize, self.__ysize)
        self.gs.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.setScene(self.gs)


    def __create_objects(self, n_points):
        self.__dots = [ ]
        # Draws the spot to where the birds must converge
        s_pen = QPen(QColor(192, 0, 0), 2)
        s_brush = QBrush(QColor(192, 0, 0))
        s = self.gs.addEllipse(QRectF(0, 0, 1, 1), s_pen, s_brush)
        s.setZValue(300)
        s.show()
        self.__dots.append(s)
        # Draws one small circle for the flock
        b_pen = QPen(QColor(255, 255, 255), 2)
        b_brush = QBrush(QColor(255, 255, 255))
        for i in range(n_points):
            b = self.gs.addEllipse(QRectF(0, 0, 1, 1), b_pen, b_brush)
            b.setZValue(301+i)
            b.show()
            self.__dots.append(b)


    def set_state(self, spot, flock):
        self.__spot = spot
        self.__flock = flock
        radius = 5
        # Updates the spot
        sx, sy = self.__transform(spot[0], spot[1])
        self.__dots[0].setRect(sx, sy, radius, radius)
        # Updates birds in the flock
        radius = 5
        for p, b in zip(flock, self.__dots[1:]):
            fx, fy = self.__transform(p[0], p[1])
            b.setRect(fx, fy, radius, radius)


    def __transform(self, x, y):
        '''
        Transforms a pair of real world coordinates to screen coordinates.
        '''
        xr = int(self.__ax * x + self.__bx)
        yr = int(self.__ay * y + self.__by)
        return (xr, yr)


    def resizeEvent(self, event):
        self.__xsize = event.size().width()
        self.__ysize = event.size().height()
        self.__set_scale()
        self.set_state(self.__spot, self.__flock)


################################################################################
