# -*- coding: utf-8 -*-
################################################################################
# Simulation of linear regression using a single linear neuron
# Jose Alexandre Nalon
#
# Date: 14-11-2011
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

from numpy.random import uniform, standard_normal
from plot import *
from peach import *
import time


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
        self.delay_label = QLabel(u"Delay: ", self)
        self.delay_mag = QSpinBox(self)
        self.delay_mag.setMinimum(0)
        self.delay_mag.setMaximum(10000)
        self.delay_mag.setValue(1000)
        self.delay_mag.setSuffix(" ms")

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.go_button, Qt.AlignLeft)
        layout.addWidget(self.stop_button, Qt.AlignLeft)
        layout.addWidget(self.step_button, Qt.AlignLeft)
        layout.addWidget(self.delay_label, Qt.AlignLeft)
        layout.addWidget(self.delay_mag, Qt.AlignLeft)

        self.enable()
        self.show()


    def enable(self):
        self.go_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.step_button.setEnabled(True)
        self.delay_mag.setEnabled(True)


    def disable(self):
        self.go_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.step_button.setEnabled(False)
        self.delay_mag.setEnabled(False)

        
    def get_delay(self):
        return self.delay_mag.value()


################################################################################
class RedefFrame(QGroupBox):
    '''
    This frame shows controls to reset and redefine the variables on the
    control.
    '''
    def __init__(self, *cnf):
        QGroupBox.__init__(self, *cnf)

        self.setTitle("Reset:")
        self.a0_label = QLabel("a0: ")
        self.a1_label = QLabel("a1: ")
        self.lrate_label = QLabel("Learning Rate: ")
        self.a0_edit = QLineEdit(self)
        self.a1_edit = QLineEdit(self)
        self.lrate_edit = QLineEdit(self)
        self.redef_button = QPushButton("Reset", self)

        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.a0_label, 0, 0, 1, 1)
        layout.addWidget(self.a1_label, 1, 0, 1, 1)
        layout.addWidget(self.lrate_label, 2, 0, 1, 1)
        layout.addWidget(self.a0_edit, 0, 1, 1, 1)
        layout.addWidget(self.a1_edit, 1, 1, 1, 1)
        layout.addWidget(self.lrate_edit, 2, 1, 1, 1)
        layout.addWidget(self.redef_button, 3, 0, 1, 2)

        self.enable()
        self.show()


    def enable(self):
        self.a0_edit.setEnabled(True)
        self.a1_edit.setEnabled(True)
        self.lrate_edit.setEnabled(True)
        self.redef_button.setEnabled(True)


    def disable(self):
        self.a0_edit.setEnabled(False)
        self.a1_edit.setEnabled(False)
        self.lrate_edit.setEnabled(False)
        self.redef_button.setEnabled(False)


    def feedback(self, a0, a1, lrate):
        self.a0_edit.setText("%7.4f" % a0)
        self.a1_edit.setText("%7.4f" % a1)
        self.lrate_edit.setText("%7.4f" % lrate)


    def get_values(self):
        a0 = float(self.a0_edit.text())
        a1 = float(self.a1_edit.text())
        lrate = float(self.lrate_edit.text())
        return (a0, a1, lrate)


################################################################################
class ExampleFrame(QGroupBox):
    '''
    This frame shows the example being presented to the neuron at each step.
    '''
    def __init__(self, *cnf):
        QGroupBox.__init__(self, *cnf)

        self.setTitle("Example:")
        self.x_label = QLabel("x: ")
        self.y_label = QLabel("y: ")
        self.x_edit = QLineEdit(self)
        self.y_edit = QLineEdit(self)

        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.x_label, 0, 0, 1, 1)
        layout.addWidget(self.y_label, 1, 0, 1, 1)
        layout.addWidget(self.x_edit, 0, 1, 1, 1)
        layout.addWidget(self.y_edit, 1, 1, 1, 1)

        self.x_edit.setEnabled(False)
        self.y_edit.setEnabled(False)
        self.show()


    def feedback(self, x, y):
        self.x_edit.setText("%7.4f" % x)
        self.y_edit.setText("%7.4f" % y)


################################################################################
class ModelFrame(QGroupBox):
    '''
    This frame shows the parameters obtained by presenting examples from the
    training set.
    '''
    def __init__(self, *cnf):
        QGroupBox.__init__(self, *cnf)

        self.setTitle("Model:")
        self.w0_label = QLabel("w0: ")
        self.w1_label = QLabel("w1: ")
        self.y_label = QLabel("Estimate: ")
        self.e_label = QLabel("Error: ")
        self.w0_edit = QLineEdit(self)
        self.w1_edit = QLineEdit(self)
        self.y_edit = QLineEdit(self)
        self.e_edit = QLineEdit(self)

        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.w0_label, 0, 0, 1, 1)
        layout.addWidget(self.w1_label, 1, 0, 1, 1)
        layout.addWidget(self.y_label, 2, 0, 1, 1)
        layout.addWidget(self.e_label, 3, 0, 1, 1)
        layout.addWidget(self.w0_edit, 0, 1, 1, 1)
        layout.addWidget(self.w1_edit, 1, 1, 1, 1)
        layout.addWidget(self.y_edit, 2, 1, 1, 1)
        layout.addWidget(self.e_edit, 3, 1, 1, 1)

        self.w0_edit.setEnabled(False)
        self.w1_edit.setEnabled(False)
        self.y_edit.setEnabled(False)
        self.e_edit.setEnabled(False)
        self.show()


    def feedback(self, w0, w1, y, e):
        self.w0_edit.setText("%7.4f" % w0)
        self.w1_edit.setText("%7.4f" % w1)
        self.y_edit.setText("%7.4f" % y)
        self.e_edit.setText("%7.4f" % e)


################################################################################
class LRFrame(QFrame):
    '''
    Shows every control and process events.
    '''
    def __init__(self, app, *cnf):

        # Coefficients of the model
        self.a0 = -0.5
        self.a1 = 0.75
        self.lrate = 0.5
        
        # Neuron. The learning parameter is made huge to make more obvious the
        # updating of the neuron.
        self.nn = FeedForward((1, 1), lrule=LMS(self.lrate), bias=True)
        self.error_log = [ ]
        self.x_log = [ ]
        self.y_log = [ ]
        
        # Control
        self.running = False

        # Frame Inicialization
        QFrame.__init__(self, *cnf)
        self.app = app
        self.setWindowTitle("Linear Regression")

        # Graphic Elements
        self.graph = LRPlotWindow((-1., 1.), (self.a0-self.a1, self.a0+self.a1))
        self.error = PlotWindow()
        self.ctrl_frame = ControlFrame(self)
        self.redef_frame = RedefFrame(self)
        self.example_frame = ExampleFrame(self)
        self.model_frame = ModelFrame(self)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.graph, 'Model')
        self.tabs.addTab(self.error, 'Error')

        layout = QGridLayout(self)
        layout.addWidget(self.tabs, 0, 0, 5, 1)
        layout.addWidget(self.ctrl_frame, 0, 1)
        layout.addWidget(self.redef_frame, 1, 1)
        layout.addWidget(self.example_frame, 2, 1)
        layout.addWidget(self.model_frame, 3, 1)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 0)
        layout.setRowStretch(3, 0)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)

        # Connects the events
        self.connect(self.ctrl_frame.go_button, SIGNAL("clicked()"), self.on_go_button)
        self.connect(self.ctrl_frame.stop_button, SIGNAL("clicked()"), self.on_stop_button)
        self.connect(self.ctrl_frame.step_button, SIGNAL("clicked()"), self.on_step_button)
        self.connect(self.redef_frame.redef_button, SIGNAL("clicked()"), self.on_redef_button)

        # Shows the frame
        w0, w1 = self.nn[0].weights[0]
        self.redef_frame.feedback(self.a0, self.a1, self.lrate)
        self.example_frame.feedback(0, 0)
        self.model_frame.feedback(w0, w1, 0, 0)
        self.show()


    def enable(self):
        self.ctrl_frame.enable()
        self.redef_frame.enable()

    def disable(self):
        self.ctrl_frame.disable()
        self.redef_frame.disable()

        
    def feedback(self):
        w0, w1 = self.nn[0].weights[0]
        x = self.x_log[-1]
        y = self.y_log[-1]
        ye = self.nn(array([ x ]))
        self.example_frame.feedback(x, y)
        self.model_frame.feedback(w0, w1, ye, y-ye)
        self.graph.setLRData([ -1., 1. ], [ w0-w1, w0+w1 ])
        if len(self.x_log) > 0:
            self.graph.setExampleData(self.x_log[-1], self.y_log[-1])
            self.graph.replot()
        if len(self.x_log) > 1:
            self.graph.setScatterData(self.x_log, self.y_log)
            self.graph.replot()
        self.error.setData(array(self.error_log))
        self.error.replot()


    def step(self):
        x = uniform(-1., 1.)               # Generates an example
        y = self.a0 + self.a1*x            # Line equation
        y = y + 0.05*standard_normal()     # Adds noise
        self.x_log.append(x)
        self.y_log.append(y)
        self.feedback()                    # Feedback made before updating the neuron
        self.error_log.append(self.nn.feed(array([ x ]), y))


    def on_go_button(self):
        self.disable()
        self.running = True
        delay = self.ctrl_frame.get_delay() / 1000.0
        while self.running:
            self.step()
            self.app.processEvents()
            time.sleep(delay)
        self.enable()


    def on_stop_button(self):
        self.running = False


    def on_step_button(self):
        if self.running:
            return
        self.step()


    def on_redef_button(self):
        if self.running:
            return
        self.a0, self.a1, self.lrate = self.redef_frame.get_values()
        if self.a1 > 0:
            self.graph.set_scale((-1., 1.), (self.a0-self.a1, self.a0+self.a1))
        elif self.a1 < 0:
            self.graph.set_scale((-1., 1.), (self.a0+self.a1, self.a0-self.a1))
        self.graph.replot()
        # Not possible to change the learning rate right now. Going to TODO.


    def closeEvent(self, event):
        self.on_stop_button()
        self.app.exit(0)


################################################################################
# Main Program
################################################################################
if __name__ == "__main__":
    q = QApplication([])
    f = LRFrame(q, None)
    q.exec_()
