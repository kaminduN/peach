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
    def __init__(self, gs, pen, *cnf):
        QGraphicsItem.__init__(self, *cnf)
        self.body = gs.addLine(QLineF(0, 0, 1, 1), pen)
        self.a1 = gs.addLine(QLineF(0, 0, 1, 1), pen)
        self.a2 = gs.addLine(QLineF(0, 0, 1, 1), pen)

    def setZValue(self, zvalue):
        self.body.setZValue(zvalue)
        self.a1.setZValue(zvalue+1)
        self.a2.setZValue(zvalue+2)

    def show(self):
        self.body.show()
        self.a1.show()
        self.a2.show()

    def hide(self):
        self.body.hide()
        self.a1.hide()
        self.a2.hide()

    def setCoordinates(self, vXo, vYo, vXi, vYi):
        ca = vXi - vXo
        co = vYi - vYo 
        t = arctan2(co, ca)
        t1 = t + pi/4.
        t2 = t - pi/4.
        self.body.setLine(vXo, vYo, vXi, vYi)
        self.a1.setLine(vXi, vYi, vXi-6*cos(t1), vYi-6*sin(t1))
        self.a2.setLine(vXi, vYi, vXi-6*cos(t2), vYi-6*sin(t2))
        self.body.show()
        self.a1.show()
        self.a2.show()


################################################################################
class PendulumView(QGraphicsView):
    """
    Visualization of the pendulum.
    """
    def __init__(self, l = 0.5, m = 0.1, *cnf):
        QGraphicsView.__init__(self, *cnf)
        self.__xsize = self.width()
        self.__ysize = self.height()
        self.pendRadius = 0.5 * m
        self.poleLength = l
        self.__setScale()
        self.__createGS()
        self.__createObjects()
        self.setState(0., 0., 0., 0., 0.)
        self.show()


    def __setScale(self):
        self.__ay = - float(self.__ysize) / 1.2
        self.__by = - 1.1 * self.__ay
        self.__ax = - self.__ay
        self.__bx = float(self.__xsize) / 2.
        self.__xmin = - self.__bx/ self.__ax
        self.__xmax = (self.__xsize - self.__bx) / self.__ax


    def __createGS(self):
        self.gs = QGraphicsScene(0, 0, self.__xsize, self.__ysize)
        self.gs.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setScene(self.gs)


    def __createObjects(self):
        # Draws the floor
        ipPen = QPen(QColor(0, 0, 0), 2)
        ipBrush = QBrush(QColor(255, 255, 255))
        _, fY = self.__transform(0., -0.04)
        fW = 0.025 * self.__ay
        floor = self.gs.addRect(QRectF(5, fY, self.__xsize-10, fW), ipPen, ipBrush)
        floor.setZValue(200)
        floor.show()

        # Dimensions of the pole in meters and radians
        self.pole = self.gs.addLine(QLineF(0, 0, 1, 1), ipPen)
        self.pole.setZValue(300)
        self.pole.show()

        # Dimensions of the weight in meters and kilograms
        self.pend = self.gs.addEllipse(QRectF(0, 0, 1, 1), ipPen, ipBrush)
        self.pend.setZValue(301)
        self.pend.show()

        # Feedback of the angle
        refPen = QPen(QColor(0, 128, 0))
        refPen.setStyle(Qt.DashLine)
        self.reference = self.gs.addLine(QLineF(0, 0, 1, 1), refPen)
        self.reference.setZValue(100)
        self.reference.show()
        self.angleText = self.gs.addText('')
        self.angleText.setDefaultTextColor(QColor(0, 128, 0))
        self.angleText.setZValue(105)
        self.angleText.show()

        # Feedback of angular velocity
        avPen = QPen(QColor(0, 0, 128))
        avPen.setWidth(2)
        self.angleVelocity = ArrowItem(self.gs, avPen)
        self.angleVelocity.setZValue(400)
        self.angleVelocity.show()
        self.avText = self.gs.addText('')
        self.avText.setDefaultTextColor(QColor(0, 0, 128))
        self.avText.setZValue(106)
        self.avText.show()

        # Dimensions of the cart in meters
        self.cartWidth = 0.3
        self.cartHeight = 0.1
        self.cart = self.gs.addRect(QRectF(0, 0, 1, 1), ipPen, ipBrush)
        self.cart.setZValue(302)
        self.cart.show()

        # Feedback of force vector (in newtons)
        vectorPen = QPen(QColor(192, 0, 0))
        vectorPen.setWidth(2)
        self.vectorLength = 0.1
        self.force = ArrowItem(self.gs, vectorPen)
        self.force.setZValue(100)
        self.force.show()

        # Force vector text (in newtons)
        self.forceText = self.gs.addText('')
        self.forceText.setDefaultTextColor(QColor(192, 0, 0))
        self.forceText.setZValue(104)
        self.forceText.show()


    def setState(self, O, w, x, v, F):
        self.__state = (O, w, x, v, F)

        # Updates the pole
        poleXo = x
        poleYo = self.cartHeight
        poleXi = poleXo + self.poleLength*sin(O)
        poleYi = poleYo + self.poleLength*cos(O)
        poleXo, poleYo = self.__transform(poleXo, poleYo)
        poleXi, poleYi = self.__transform(poleXi, poleYi)
        self.pole.setLine(poleXo, poleYo, poleXi, poleYi)

        # Updates the weight
        pendRadius = self.pendRadius * self.__ax
        self.pend.setRect(poleXi-pendRadius, poleYi-pendRadius, pendRadius*2, pendRadius*2)

        # Updates the angle reference
        _, refYi = self.__transform(0, self.poleLength)
        self.reference.setLine(poleXo, poleYo, poleXo, poleYo-refYi)
        self.angleText.setHtml('O = %7.2f'%(O*180./pi))
        self.angleText.setPos(poleXo-37, poleYo-refYi-16)

        # Updates the cart
        cartX = x - self.cartWidth/2.
        cartY = 0.
        cartX, cartY = self.__transform(cartX, cartY)
        cartWidth = self.cartWidth * self.__ax
        cartHeight = self.cartHeight * self.__ay
        self.cart.setRect(cartX, cartY, cartWidth, cartHeight)

        # Updates the angular velocity
        if -0.1 < w < 0.1:
            avL = sign(w)*0.01
        else:
            avL = 0.1*w
        if w > 0.:
            avXi = poleXi + pendRadius*cos(O)
            avYi = poleYi + pendRadius*sin(O)
            avXo = avXi + self.__ax * avL * cos(O)
            avYo = avYi - self.__ay * avL * sin(O)
            self.angleVelocity.setCoordinates(avXi, avYi, avXo, avYo)
            self.avText.setHtml('w = %7.4f' % w)
            self.avText.setPos(avXo + 5, avYo)
        elif w < 0.:
            avXi = poleXi - pendRadius*cos(O)
            avYi = poleYi - pendRadius*sin(O)
            avXo = avXi + self.__ax * avL * cos(O)
            avYo = avYi - self.__ay * avL * sin(O)
            self.angleVelocity.setCoordinates(avXi, avYi, avXo, avYo)
            self.avText.setHtml('w = %7.4f' % w)
            self.avText.setPos(avXo-70, avYo-8)
        else:
            self.angleVelocity.hide()
            #self.avText.hide()

        # Updates the force vector
        if -0.2 < F < 0.2:
            vectorL = sign(F)*0.02
        else:
            vectorL = 0.1*F
        if vectorL > 0.:
            vectorX = x - self.cartWidth/2. - vectorL
            vX, vY = self.__transform(vectorX, self.cartHeight/2.)
            vectorL = vectorL * self.__ax
            self.force.setCoordinates(vX, vY, vX + vectorL, vY)
            self.forceText.setHtml('F = %7.4f' % F)
            self.forceText.setPos(vX+vectorL-70, vY - 16)
        elif vectorL < 0.:
            vectorX = x + self.cartWidth/2. - vectorL
            vX, vY = self.__transform(vectorX, self.cartHeight/2.)
            vectorL = vectorL * self.__ax
            self.force.setCoordinates(vX, vY, vX + vectorL, vY)
            self.forceText.setHtml('F=%7.4f' % F)
            self.forceText.setPos(vX+vectorL+5, vY - 16)
        else:
            self.force.hide()
            #self.forceText.hide()

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
        self.__setScale()
        self.__createGS()
        self.__createObjects()
        self.setState(*self.__state)


################################################################################
