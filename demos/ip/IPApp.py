# -*- coding: utf-8 -*-
################################################################################
# A Complete Simulation of a Inverted Pendulum, controlled with a
# Fuzzy Logic Controller
# Jose Alexandre Nalon
#
# Date: 06-12-2007
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

import ip
from qtip import *
from plot import *
from peach import *


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

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.go_button, Qt.AlignLeft)
        layout.addWidget(self.stop_button, Qt.AlignLeft)
        layout.addWidget(self.step_button, Qt.AlignLeft)

        self.enable()
        self.show()


    def enable(self):
        self.go_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.step_button.setEnabled(True)


    def disable(self):
        self.go_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.step_button.setEnabled(False)


################################################################################
class RedefFrame(QGroupBox):
    '''
    This frame shows controls to reset and redefine the variables on the
    control.
    '''
    def __init__(self, *cnf):
        QGroupBox.__init__(self, *cnf)

        self.setTitle("Reset:")
        self.theta_label = QLabel("Theta: ")
        self.omega_label = QLabel("Omega: ")
        self.x_label = QLabel("Position: ")
        self.v_label = QLabel("Speed: ")
        self.theta_edit = QLineEdit(self)
        self.omega_edit = QLineEdit(self)
        self.x_edit = QLineEdit(self)
        self.v_edit = QLineEdit(self)
        self.redef_button = QPushButton("Reset", self)

        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.theta_label, 0, 0, 1, 1)
        layout.addWidget(self.omega_label, 1, 0, 1, 1)
        layout.addWidget(self.x_label, 2, 0, 1, 1)
        layout.addWidget(self.v_label, 3, 0, 1, 1)
        layout.addWidget(self.theta_edit, 0, 1, 1, 1)
        layout.addWidget(self.omega_edit, 1, 1, 1, 1)
        layout.addWidget(self.x_edit, 2, 1, 1, 1)
        layout.addWidget(self.v_edit, 3, 1, 1, 1)
        layout.addWidget(self.redef_button, 4, 0, 1, 2)

        self.enable()
        self.show()


    def enable(self):
        self.theta_edit.setEnabled(True)
        self.omega_edit.setEnabled(True)
        self.x_edit.setEnabled(True)
        self.v_edit.setEnabled(True)
        self.redef_button.setEnabled(True)


    def disable(self):
        self.theta_edit.setEnabled(False)
        self.omega_edit.setEnabled(False)
        self.x_edit.setEnabled(False)
        self.v_edit.setEnabled(False)
        self.redef_button.setEnabled(False)


    def feedback(self, O, w, x, v, F):
        self.theta_edit.setText("%5.2f" % (O*180./pi))
        self.omega_edit.setText("%7.4f" % w)
        self.x_edit.setText("%7.4f" % x)
        self.v_edit.setText("%7.4f" % v)


    def get_values(self):
        O = float(self.theta_edit.text()) * pi/180.
        w = float(self.omega_edit.text())
        x = float(self.x_edit.text())
        v = float(self.v_edit.text())
        return (O, w, x, v)


################################################################################
class ConfigFrame(QGroupBox):
    '''
    This frame shows the redefinitions allowed for the controller. You can
    select different defuzzification, logic or inference operations.
    '''
    def __init__(self, *cnf):
        QGroupBox.__init__(self, *cnf)
        self.setTitle("Configuration")

        self.logic_label = QLabel("Fuzzy Logic:")
        self.logic_combo = QComboBox(self)
        self.logic_combo.addItems([ "Zadeh", "Probabilistic", "Einstein",
            "Drastic" ])
        self.infer_label = QLabel("Inference:")
        self.infer_combo = QComboBox(self)
        self.infer_combo.addItems([ "Mamdani", "Probabilistic", "Zadeh/Mamdani",
            "Dienes-Rescher/Mamdani", "Lukasiewicz/Mamdani", "Godel/Mamdani" ])
        self.defuzzy_label = QLabel("Defuzzification:")
        self.defuzzy_combo = QComboBox(self)
        self.defuzzy_combo.addItems([ "Centroid", "Bisector", "SOM", "LOM", "MOM" ])

        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.logic_label, 0, 0)
        layout.addWidget(self.logic_combo, 0, 1)
        layout.addWidget(self.infer_label)
        layout.addWidget(self.infer_combo)
        layout.addWidget(self.defuzzy_label, 2, 0)
        layout.addWidget(self.defuzzy_combo, 2, 1)

        self.enable()
        self.show()


    def enable(self):
        self.logic_combo.setEnabled(True)
        self.infer_combo.setEnabled(True)
        self.defuzzy_combo.setEnabled(True)


    def disable(self):
        self.logic_combo.setEnabled(False)
        self.infer_combo.setEnabled(False)
        self.defuzzy_combo.setEnabled(False)


################################################################################
class IPFrame(QFrame):
    '''
    Shows every control and process events.
    '''
    def __init__(self, app, *cnf):

        # Pendulum data (MKS units)
        l = 0.5
        m = 0.1
        mc = 0.5
        dt = 0.01
        self.ip = ip.InvertedPendulum(l, m, mc, dt)
        self.pc = ip.PendulumController
        self.running = False
        self.Orange = linspace(-3.*pi/8., 3.*pi/8., 100)
        self.wrange = linspace(-9.*pi/2., 9.*pi/2., 100)
        self.F = 0.
        self.Otrack = [ ]
        self.wtrack = [ ]
        self.xtrack = [ ]
        self.vtrack = [ ]
        self.Ftrack = [ ]

        # Frame Inicialization
        QFrame.__init__(self, *cnf)
        self.app = app
        self.setWindowTitle("Inverted Pendulum")

        # Graphic Elements
        self.ipview = PendulumView(l, m)
        self.graph = PlotWindow(5)
        self.ctrl_frame = ControlFrame(self)
        self.redef_frame = RedefFrame(self)
        self.config_frame = ConfigFrame(self)
        self.rule_label = QLabel("Show Rule:")
        self.rule_combo = QComboBox(self)
        self.rule_combo.addItems([
          'MGN & GN -> MMGN', 'MGN & PN -> MMGN', 'MGN & Z -> MGN', 'MGN & PP -> GN', 'MGN & GP -> PN',
          'GN & GN -> MMGN', 'GN & PN -> MGN', 'GN & Z -> GN', 'GN & PP -> PN', 'GN & GP -> Z',
          'PN & GN -> MGN', 'PN & PN -> GN', 'PN & Z -> PN', 'PN & PP -> Z', 'GN & GP -> PP',
          'Z & GN -> GN', 'Z & PN -> PN', 'Z & Z -> Z', 'Z & PP -> PP', 'Z & GP -> GP',
          'PP & GN -> PN', 'PP & PN -> Z', 'PP & Z -> PP', 'PP & PP -> GP', 'PP & GP -> MGP',
          'GP & GN -> Z', 'GP & PN -> PP', 'GP & Z -> GP', 'GP & PP -> MGP', 'GP & GP -> MMGP',
          'MGP & GN -> PP', 'MGP & PN -> GP', 'MGP & Z -> MGP', 'MGP & PP -> MMGP', 'MGP & GP -> MMGP'
        ])
        self.rule_combo.setCurrentIndex(17) # Z & Z -> Z
        self.rule_combo.setEnabled(False)

        # Plots
        self.gframe = QFrame(self)
        self.Ograph = PlotWindow(8, self.gframe)
        self.Ograph.setAxisScale(Qwt.QwtPlot.xBottom, -3*pi/8, 3*pi/8)
        self.Ograph.setAxisScale(Qwt.QwtPlot.yLeft, -0.1, 1.1)
        self.Ograph.setCurveColor(-1, Qt.black)
        for i in range(7):
            self.Ograph.setCurveStyle(i, Qt.DotLine)
        self.wgraph = PlotWindow(6, self.gframe)
        self.wgraph.setAxisScale(Qwt.QwtPlot.xBottom, -9*pi/2., 9*pi/2.)
        self.wgraph.setAxisScale(Qwt.QwtPlot.yLeft, -0.1, 1.1)
        self.wgraph.setCurveColor(-1, Qt.black)
        for i in range(5):
            self.wgraph.setCurveStyle(i, Qt.DotLine)
        self.Fgraph = PlotWindow(12, self.gframe)
        self.Fgraph.setAxisScale(Qwt.QwtPlot.xBottom, -100., 100.)
        self.Fgraph.setAxisScale(Qwt.QwtPlot.yLeft, -0.1, 1.1)
        self.Fgraph.setCurveColor(0, Qt.darkGray)
        self.Fgraph.setCurveBaseline(0, 0.)
        self.Fgraph.setCurveBrush(0, QBrush(Qt.gray, Qt.SolidPattern))
        self.Fgraph.setCurveColor(1, Qt.black)
        self.Fgraph.setCurveBaseline(1, 0.)
        self.Fgraph.setCurveBrush(1, QBrush(Qt.darkGray, Qt.SolidPattern))
        self.Fgraph.setCurveColor(2, Qt.red)
        glayout = QGridLayout(self.gframe)
        glayout.addWidget(self.Ograph, 0, 0)
        glayout.addWidget(self.wgraph, 1, 0)
        glayout.addWidget(self.Fgraph, 0, 1, 2, 1)
        glayout.setRowStretch(0, 1)
        glayout.setRowStretch(1, 1)
        glayout.setColumnStretch(0, 1)
        glayout.setColumnStretch(1, 2)
        self.__drawO()
        self.__draww()
        self.__drawF()
        self.gframe.setLayout(glayout)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.ipview, 'Pendulum')
        self.tabs.addTab(self.graph, 'Graphics')
        self.tabs.addTab(self.gframe, 'Membership')

        layout = QGridLayout(self)
        layout.addWidget(self.tabs, 0, 0, 5, 1)
        layout.addWidget(self.ctrl_frame, 0, 1)
        layout.addWidget(self.redef_frame, 1, 1)
        layout.addWidget(self.config_frame, 2, 1)
        layout.addWidget(self.rule_label, 3, 1)
        layout.addWidget(self.rule_combo, 4, 1)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 0)
        layout.setRowStretch(3, 0)
        layout.setRowStretch(4, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        self.feedback(O=0., w=0., x=0., v=0., F=0.)

        # Connects the events
        self.connect(self.ctrl_frame.go_button, SIGNAL("clicked()"), self.on_go_button)
        self.connect(self.ctrl_frame.stop_button, SIGNAL("clicked()"), self.on_stop_button)
        self.connect(self.ctrl_frame.step_button, SIGNAL("clicked()"), self.on_step_button)
        self.connect(self.redef_frame.redef_button, SIGNAL("clicked()"), self.on_redef_button)
        self.connect(self.config_frame.logic_combo, SIGNAL("currentIndexChanged(int)"), self.on_logic_combo)
        self.connect(self.config_frame.infer_combo, SIGNAL("currentIndexChanged(int)"), self.on_infer_combo)
        self.connect(self.config_frame.defuzzy_combo, SIGNAL("currentIndexChanged(int)"), self.on_defuzzy_combo)
        self.connect(self.tabs, SIGNAL("currentChanged(int)"), self.on_change_tab)
        self.connect(self.rule_combo, SIGNAL("currentIndexChanged(int)"), self.on_rule_combo)

        # Exibe o frame
        self.set_state(pi/8., 0., 0., 0., 0.)
        self.show()


    def enable(self):
        self.ctrl_frame.enable()
        self.redef_frame.enable()
        self.config_frame.enable()

    def disable(self):
        self.ctrl_frame.disable()
        self.redef_frame.disable()
        self.config_frame.disable()


    def __drawO(self):
        x = self.Orange
        self.Ograph.setMultiData( [
          (x, ip.Ovbn(x)), (x, ip.Obn(x)), (x, ip.Osn(x)), (x, ip.Oz(x)),
          (x, ip.Osp(x)), (x, ip.Obp(x)), (x, ip.Ovbp(x)), ([ 0. ], [ 0. ])
        ] )


    def __draww(self):
        x = self.wrange
        self.wgraph.setMultiData( [
          (x, ip.wbn(x)), (x, ip.wsn(x)), (x, ip.wz(x)),
          (x, ip.wsp(x)), (x, ip.wbp(x)), ([ 0. ], [ 0. ])
        ] )


    def __drawF(self):
        x = ip.F
        self.Fgraph.setMultiData( [
          ( [ 0. ], [ 0. ] ), ( [ 0. ], [ 0. ] ), ( [ 0., 0. ], [ -0.025, -0.1 ] ),
          (x, ip.Fvvbn(x)), (x, ip.Fvbn(x)), (x, ip.Fbn(x)), (x, ip.Fsn(x)),
          (x, ip.Fz(x)), (x, ip.Fsp(x)), (x, ip.Fbp(x)), (x, ip.Fvbp(x)),
          (x, ip.Fvvbp(x))
        ] )


    def set_state(self, O, w, x, v, F):
        self.Otrack = [ O ]
        self.wtrack = [ w ]
        self.xtrack = [ x ]
        self.vtrack = [ v ]
        self.Ftrack = [ F ]
        self.ip.set_state(O, w, x, v)
        self.feedback(O, w, x, v, F)


    def feedback(self, O, w, x, v, F):
        ci = self.tabs.currentIndex()
        if ci == 0:   # Pendulum
            self.ipview.set_state(O, w, x, v, F)
        elif ci == 1: # Plots
            t = arange(0., 2.5, self.ip.dt)
            self.graph.setMultiData( [
               (t, self.Otrack), (t, self.wtrack),
               (t, self.xtrack), (t, self.vtrack),
               (t, zeros(t.shape)) #self.Ftrack)
            ])
        elif ci == 2: # Membership
            self.Ograph.setData(-1, [ O, O ], [ 0., 1. ])
            self.wgraph.setData(-1, [ w, w ], [ 0., 1. ])
            self.Fgraph.setData(2, [ F, F ], [ -0.025, -0.1 ])
            rF = self.pc.eval_all(O, w)
            rule = self.rule_combo.currentIndex()
            _, sF = self.pc.eval(rule, (O, w))
            if sF is None:
                sF = zeros(ip.F.shape)
            self.Fgraph.setData(0, ip.F, rF)
            self.Fgraph.setData(1, ip.F, sF)
        self.redef_frame.feedback(O, w, x, v, F)


    def step(self):
        O, w, x, v = self.ip.get_state()
        F = self.pc(O, w)
        self.ip.apply(F)
        self.feedback(O, w, x, v, F)
        self.Otrack.append(O)
        self.wtrack.append(w)
        self.xtrack.append(x)
        self.vtrack.append(v)
        self.Ftrack.append(F)


    def on_go_button(self):
        self.disable()
        self.running = True
        while self.running:
            self.step()
            self.app.processEvents()
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
        O, w, x, v = self.redef_frame.get_values()
        self.Otrack = [ ]
        self.wtrack = [ ]
        self.xtrack = [ ]
        self.vtrack = [ ]
        self.Ftrack = [ ]
        self.set_state(O, w, x, v, 0)


    def on_logic_combo(self, index):
        if index == 0:      # Zadeh norms
            self.pc.set_norm(ZadehAnd)
            self.pc.set_conorm(ZadehOr)
            self.pc.set_negation(ZadehNot)
        elif index == 1:    # Probabilistic norms
            self.pc.set_norm(ProbabilisticAnd)
            self.pc.set_conorm(ProbabilisticOr)
            self.pc.set_negation(ProbabilisticNot)
        elif index == 2:    # Einstein norms
            self.pc.set_norm(EinsteinProduct)
            self.pc.set_conorm(EinsteinSum)
            self.pc.set_negation(ZadehNot)
        elif index == 3:    # Drastic norms
            self.pc.set_norm(DrasticProduct)
            self.pc.set_conorm(DrasticSum)
            self.pc.set_negation(ZadehNot)


    def on_infer_combo(self, index):
        if index == 0:      # Mamdani rules
            self.pc.set_implication(MamdaniImplication)
            self.pc.set_aglutination(MamdaniAglutination)
        elif index == 1:    # Probabilistic rules
            self.pc.set_implication(ProbabilisticImplication)
            self.pc.set_aglutination(ProbabilisticAglutination)
        elif index == 2:    # Zadeh implication and Mamdani aglutination
            self.pc.set_implication(ZadehImplication)
            self.pc.set_aglutination(MamdaniAglutination)
        elif index == 3:    # Dienes-Rescher implication and Mamdani aglutination
            self.pc.set_implication(DienesRescherImplication)
            self.pc.set_aglutination(MamdaniAglutination)
        elif index == 4:    # Lukasiewicz implication and Mamdani aglutination
            self.pc.set_implication(LukasiewiczImplication)
            self.pc.set_aglutination(MamdaniAglutination)
        elif index == 5:    # Godel implication and Mamdani aglutination
            self.pc.set_implication(GodelImplication)
            self.pc.set_aglutination(MamdaniAglutination)


    def on_defuzzy_combo(self, index):
        if index == 0:     # Centroid:
            self.pc.defuzzify = Centroid
        elif index == 1:   # Bisection
            self.pc.defuzzify = Bisector
        elif index == 2:   # SOM
            self.pc.defuzzify = SmallestOfMaxima
        elif index == 3:   # LOM
            self.pc.defuzzify = LargestOfMaxima
        elif index == 4:   # MOM
            self.pc.defuzzify = MeanOfMaxima


    def on_change_tab(self, index):
        if index == 0:   # Pendulum
            O = self.Otrack[-1]
            w = self.wtrack[-1]
            x = self.xtrack[-1]
            v = self.vtrack[-1]
            F = self.Ftrack[-1]
            self.ipview.set_state(O, w, x, v, F)
            self.rule_combo.setEnabled(False)
        elif index == 1: # Plots
            t = arange(0., 2.5, self.ip.dt)
            self.graph.setMultiData( [
               (t, self.Otrack), (t, self.wtrack),
               (t, self.xtrack), (t, self.vtrack),
               (t, zeros(t.shape)) #self.Ftrack)
            ])
            self.rule_combo.setEnabled(False)
        elif index == 2: # Membership
            O = self.Otrack[-1]
            w = self.wtrack[-1]
            x = self.xtrack[-1]
            v = self.vtrack[-1]
            F = self.Ftrack[-1]
            self.feedback(O, w, x, v, F)
            self.rule_combo.setEnabled(True)


    def on_rule_combo(self, index):
        O = self.Otrack[-1]
        w = self.wtrack[-1]
        x = self.xtrack[-1]
        v = self.vtrack[-1]
        F = self.Ftrack[-1]
        self.feedback(O, w, x, v, F)


    def closeEvent(self, event):
        self.on_stop_button()
        self.app.exit(0)


################################################################################
# Main Program
################################################################################
if __name__ == "__main__":
    q = QApplication([])
    f = IPFrame(q, None)
    q.exec_()
