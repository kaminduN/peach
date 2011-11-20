# -*- coding: utf-8 -*-
################################################################################
# A Demonstration of Swarm in Action
# Jose Alexandre Nalon
#
# Date: 22-10-2007
# This is the main program
################################################################################

# Obs.: Since this is a standalone program, we do not document using the
#       epydoc API. So, no documentation for classes, etc. This program is
#       documented, however, using a Sphinx interface. Please, consult the
#       standard module documentation.


################################################################################
# Used modules
################################################################################
# We use the PyQt4 toolkit
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from numpy import *
from time import sleep

from qtflock import *
from peach import *

from numpy.random import random


################################################################################
# Classes
################################################################################
class ControlFrame(QGroupBox):
    '''
    This frame shows the application control buttons.
    '''
    def __init__(self, *cnf):
        QGroupBox.__init__(self, *cnf)

        self.setTitle("Control:")
        self.go_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)
        self.step_button = QPushButton("Step", self)
        self.change_button = QPushButton("Change Spot", self)
        self.reset_button = QPushButton("Reset", self)
        self.delay_label = QLabel("Delay (ms):", self)
        self.delay_edit = QLineEdit(self)

        self.delay_edit.setText('100')

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.go_button, Qt.AlignLeft)
        layout.addWidget(self.stop_button, Qt.AlignLeft)
        layout.addWidget(self.step_button, Qt.AlignLeft)
        layout.addWidget(self.change_button, Qt.AlignLeft)
        layout.addWidget(self.reset_button, Qt.AlignLeft)
        layout.addWidget(self.delay_label, Qt.AlignLeft)
        layout.addWidget(self.delay_edit, Qt.AlignLeft)

        self.enable()
        self.show()


    def enable(self):
        self.go_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.step_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.delay_edit.setEnabled(True)


    def disable(self):
        self.go_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.step_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.delay_edit.setEnabled(False)


################################################################################
# Function to represent the searched spot. Implemented in the form of a class
# to allow dynamic change of parameters.
class Function(object):
    def __init__(self):
        self.center = array([ [ 0.5 ], [ 0.5 ] ])
        self.offset = 0.
    def __call__(self, x):
        x = x - self.center
        v = sum(x*x) - self.offset
        return v


################################################################################
class FlockFrame(QFrame):
    '''
    Shows every control and process events.
    '''
    def __init__(self, app, *cnf):

        # Optimizer initialization
        self.__fn = Function()
        self.__spot = random((2, ))
        self.__fn.center = self.__spot
        flock = random((10, 2))     # First estimates
        ranges = [ (0., 1.), (0., 1.) ]
        self.pso = ParticleSwarmOptimizer(self.__fn, flock, ranges)
        self.running = False
        self.count = 0

        # Frame Inicialization
        QFrame.__init__(self, *cnf)
        self.app = app
        self.setWindowTitle("Flock of Birds")

        # Graphic Elements
        self.flock_view = FlockView(self.__spot, flock)
        self.ctrl_frame = ControlFrame(self)

        layout = QGridLayout(self)
        layout.addWidget(self.flock_view, 0, 0, 2, 1)
        layout.addWidget(self.ctrl_frame, 0, 1)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)

        # Connects the events
        self.connect(self.ctrl_frame.go_button, SIGNAL("clicked()"), self.on_go_button)
        self.connect(self.ctrl_frame.stop_button, SIGNAL("clicked()"), self.on_stop_button)
        self.connect(self.ctrl_frame.step_button, SIGNAL("clicked()"), self.on_step_button)
        self.connect(self.ctrl_frame.change_button, SIGNAL("clicked()"), self.on_change_button)
        self.connect(self.ctrl_frame.reset_button, SIGNAL("clicked()"), self.on_reset_button)

        self.show()

    def enable(self):
        self.ctrl_frame.enable()

    def disable(self):
        self.ctrl_frame.disable()

    def set_state(self, spot, flock):
        self.flock_view.set_state(spot, flock)

    def reset(self):
        self.__spot = random((2, ))
        self.__fn.center = self.__spot
        self.__fn.offset = self.__fn.offset + 0.25
        flock = random((10, 2))
        #self.pso.reset(flock)
        self.pso[:] = flock[:]
        self.set_state(self.__spot, flock)

    def step(self):
        self.pso.step()
        self.count = self.count + 1
        if self.count % 73 == 0:
            self.reset()
        else:
            self.set_state(self.__spot, self.pso[:])

    def on_go_button(self):
        self.disable()
        self.running = True
        try:
            delay = int(self.ctrl_frame.delay_edit.text()) / 1000.
        except ValueError:
            delay = 0
        while self.running:
            self.step()
            self.app.processEvents()
            sleep(delay)
        self.enable()

    def on_stop_button(self):
        self.running = False

    def on_step_button(self):
        if self.running:
            return
        self.step()

    def on_change_button(self):
        self.__spot = random((2, ))
        self.__fn.center = self.__spot
        self.set_state(self.__spot, self.pso[:])

    def on_reset_button(self):
        if self.running:
            return
        self.reset()

    def closeEvent(self, event):
        self.on_stop_button()
        self.app.exit(0)


################################################################################
# Main Program
################################################################################
if __name__ == "__main__":
    q = QApplication([])
    f = FlockFrame(q, None)
    q.exec_()
