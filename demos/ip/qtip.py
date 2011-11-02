# -*- coding: utf-8 -*-
################################################################################
# Widget to draw the inverted pendulum.
# Jose Alexandre Nalon
#
# Date: 06-12-2007
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
class ArrowItem(QGraphicsItem):
    def __init__(self, gs, pen, brush, *cnf):
        QGraphicsItem.__init__(self, *cnf)
        self.body = gs.addLine(QLineF(), pen)
        self.a = gs.addPolygon(QPolygonF(), pen, brush)

    def setZValue(self, zvalue):
        self.body.setZValue(zvalue)
        self.a.setZValue(zvalue)

    def show(self):
        self.body.show()
        self.a.show()

    def hide(self):
        self.body.hide()
        self.a.hide()

    def set_coordinates(self, v_xo, v_yo, v_xi, v_yi):
        ca = v_xi - v_xo
        co = v_yi - v_yo
        t = arctan2(co, ca)
        t1 = t + pi/8.
        t2 = t - pi/8.
        P1 = QPointF(v_xi, v_yi)
        P2 = QPointF(v_xi-6*cos(t1), v_yi-6*sin(t1))
        P3 = QPointF(v_xi-6*cos(t2), v_yi-6*sin(t2))
        self.body.setLine(v_xo, v_yo, v_xi, v_yi)
        self.a.setPolygon(QPolygonF([ P1, P2, P3 ]))
        self.body.show()
        self.a.show()


################################################################################
class PendulumView(QGraphicsView):
    """
    Visualization of the pendulum.
    """
    def __init__(self, l = 0.5, m = 0.1, *cnf):
        QGraphicsView.__init__(self, *cnf)
        self.__xsize = self.width()
        self.__ysize = self.height()
        self.pend_radius = 0.5 * m
        self.pole_length = l
        self.__set_scale()
        self.__create_gs()
        self.__create_objects()
        self.set_state(0., 0., 0., 0., 0.)
        self.show()


    def __set_scale(self):
        xmin = -0.1
        ymin = -1.1
        xmax = 1.1
        ymax = 1.1
        self.__ay = - float(self.__ysize) / (ymax - ymin)
        self.__by = - self.__ay * ymax
        self.__ax = - self.__ay        # Same scale for x-axis
        self.__bx = self.__xsize / 2.


    def __create_gs(self):
        self.gs = QGraphicsScene(0, 0, self.__xsize, self.__ysize)
        self.gs.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setScene(self.gs)


    def __create_objects(self):
        # Draws the floor
        ip_pen = QPen(QColor(0, 0, 0), 2)
        ip_brush = QBrush(QColor(255, 255, 255))
        _, fy = self.__transform(0., -0.04)
        fw = 0.025 * self.__ay
        floor = self.gs.addRect(QRectF(5, fy, self.__xsize-10, fw), ip_pen, ip_brush)
        floor.setZValue(200)
        floor.show()

        # Dimensions of the pole in meters and radians
        self.pole = self.gs.addLine(QLineF(), ip_pen)
        self.pole.setZValue(300)
        self.pole.show()

        # Dimensions of the weight in meters and kilograms
        self.pend = self.gs.addEllipse(QRectF(), ip_pen, ip_brush)
        self.pend.setZValue(500)
        self.pend.show()

        # Feedback of the angle
        ref_pen = QPen(QColor(0, 128, 0))
        ref_pen.setStyle(Qt.DashLine)
        self.reference = self.gs.addLine(QLineF(), ref_pen)
        self.reference.setZValue(100)
        self.reference.show()
        self.angle_text = self.gs.addText('')
        self.angle_text.setDefaultTextColor(QColor(0, 128, 0))
        self.angle_text.setZValue(105)
        self.angle_text.show()

        # Feedback of angular velocity
        av_pen = QPen(QColor(0, 0, 128))
        av_pen.setWidth(2)
        av_brush = QBrush(QColor(0, 0, 128))
        self.angle_velocity = ArrowItem(self.gs, av_pen, av_brush)
        self.angle_velocity.setZValue(400)
        self.angle_velocity.show()
        self.av_text = self.gs.addText('')
        self.av_text.setDefaultTextColor(QColor(0, 0, 128))
        self.av_text.setZValue(106)
        self.av_text.show()

        # Dimensions of the cart in meters
        self.cart_width = 0.3
        self.cart_height = 0.1
        self.cart = self.gs.addRect(QRectF(), ip_pen, ip_brush)
        self.cart.setZValue(502)
        self.cart.show()

        # Feedback of force vector (in newtons)
        vector_pen = QPen(QColor(192, 0, 0))
        vector_pen.setWidth(2)
        vector_brush = QBrush(QColor(192, 0, 0))
        self.vector_length = 0.1
        self.force = ArrowItem(self.gs, vector_pen, vector_brush)
        self.force.setZValue(100)
        self.force.show()

        # Force vector text (in newtons)
        self.force_text = self.gs.addText('')
        self.force_text.setDefaultTextColor(QColor(192, 0, 0))
        self.force_text.setZValue(104)
        self.force_text.show()


    def set_state(self, O, w, x, v, F):
        self.__state = (O, w, x, v, F)

        # Updates the pole
        pole_xo = x
        pole_yo = self.cart_height
        pole_xi = pole_xo + self.pole_length*sin(O)
        pole_yi = pole_yo + self.pole_length*cos(O)
        pole_xo, pole_yo = self.__transform(pole_xo, pole_yo)
        pole_xi, pole_yi = self.__transform(pole_xi, pole_yi)
        self.pole.setLine(pole_xo, pole_yo, pole_xi, pole_yi)

        # Updates the weight
        pend_radius = self.pend_radius * self.__ax
        self.pend.setRect(pole_xi-pend_radius, pole_yi-pend_radius, pend_radius*2, pend_radius*2)

        # Updates the angle reference
        _, ref_yi = self.__transform(0, self.pole_length)
        self.reference.setLine(pole_xo, pole_yo, pole_xo, pole_yo-ref_yi)
        self.angle_text.setHtml('O = %7.2f'%(O*180./pi))
        self.angle_text.setPos(pole_xo-37, pole_yo-ref_yi-20)
        self.angle_text.show()

        # Updates the cart
        cart_x = x - self.cart_width/2.
        cart_y = 0.
        cart_x, cart_y = self.__transform(cart_x, cart_y)
        cart_width = self.cart_width * self.__ax
        cart_height = self.cart_height * self.__ay
        self.cart.setRect(cart_x, cart_y, cart_width, cart_height)

        # Updates the angular velocity
        if -0.1 < w < 0.1:
            av_l = sign(w)*0.01
        else:
            av_l = 0.1*w
        if w > 0.01:
            av_xi = pole_xi + pend_radius*cos(O)
            av_yi = pole_yi + pend_radius*sin(O)
            av_xo = av_xi + self.__ax * av_l * cos(O)
            av_yo = av_yi - self.__ay * av_l * sin(O)
            self.angle_velocity.set_coordinates(av_xi, av_yi, av_xo, av_yo)
            self.av_text.setHtml('w = %7.4f' % w)
            self.av_text.setPos(av_xo, av_yo-8)
            self.av_text.show()
        elif w < -0.01:
            av_xi = pole_xi - pend_radius*cos(O)
            av_yi = pole_yi - pend_radius*sin(O)
            av_xo = av_xi + self.__ax * av_l * cos(O)
            av_yo = av_yi - self.__ay * av_l * sin(O)
            self.angle_velocity.set_coordinates(av_xi, av_yi, av_xo, av_yo)
            self.av_text.setHtml('w = %7.4f' % w)
            self.av_text.setPos(av_xo-70, av_yo-8)
            self.av_text.show()
        else:
            self.angle_velocity.hide()
            self.av_text.hide()

        # Updates the force vector
        if -0.2 < F < 0.2:
            vector_l = sign(F)*0.02
        else:
            vector_l = 0.1*F
        if vector_l > 0.1:
            vector_x = x - self.cart_width/2. - vector_l
            v_x, v_y = self.__transform(vector_x, self.cart_height/2.)
            vector_l = vector_l * self.__ax
            self.force.set_coordinates(v_x, v_y, v_x + vector_l-2, v_y)
            self.force_text.setHtml('F = %7.4f' % F)
            self.force_text.setPos(v_x+vector_l-80, v_y - 20)
            self.force_text.show()
        elif vector_l < -0.1:
            vector_x = x + self.cart_width/2. - vector_l
            v_x, v_y = self.__transform(vector_x, self.cart_height/2.)
            vector_l = vector_l * self.__ax
            self.force.set_coordinates(v_x, v_y, v_x + vector_l+2, v_y)
            self.force_text.setHtml('F=%7.4f' % F)
            self.force_text.setPos(v_x+vector_l+5, v_y - 20)
            self.force_text.show()
        else:
            self.force.hide()
            self.force_text.hide()

        # Updates the scene
        self.gs.update()


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
        self.__create_gs()
        self.__create_objects()
        self.set_state(*self.__state)


################################################################################
