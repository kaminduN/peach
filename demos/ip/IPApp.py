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
        self.goButton = QPushButton("Start", self)
        self.stopButton = QPushButton("Stop", self)
        self.stepButton = QPushButton("Step", self)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.goButton, Qt.AlignLeft)
        layout.addWidget(self.stopButton, Qt.AlignLeft)
        layout.addWidget(self.stepButton, Qt.AlignLeft)

        self.enable()
        self.show()


    def enable(self):
        self.goButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.stepButton.setEnabled(True)


    def disable(self):
        self.goButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.stepButton.setEnabled(False)


################################################################################
class RedefFrame(QGroupBox):
    '''
    This frame shows controls to reset and redefine the variables on the
    control.
    '''
    def __init__(self, *cnf):
        QGroupBox.__init__(self, *cnf)

        self.setTitle("Reset:")
        self.thetaLabel = QLabel("Theta: ")
        self.omegaLabel = QLabel("Omega: ")
        self.xLabel = QLabel("Position: ")
        self.vLabel = QLabel("Speed: ")
        self.thetaEdit = QLineEdit(self)
        self.omegaEdit = QLineEdit(self)
        self.xEdit = QLineEdit(self)
        self.vEdit = QLineEdit(self)
        self.redefButton = QPushButton("Reset", self)

        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.thetaLabel, 0, 0, 1, 1)
        layout.addWidget(self.omegaLabel, 1, 0, 1, 1)
        layout.addWidget(self.xLabel, 2, 0, 1, 1)
        layout.addWidget(self.vLabel, 3, 0, 1, 1)
        layout.addWidget(self.thetaEdit, 0, 1, 1, 1)
        layout.addWidget(self.omegaEdit, 1, 1, 1, 1)
        layout.addWidget(self.xEdit, 2, 1, 1, 1)
        layout.addWidget(self.vEdit, 3, 1, 1, 1)
        layout.addWidget(self.redefButton, 4, 0, 1, 2)

        self.enable()
        self.show()


    def enable(self):
        self.thetaEdit.setEnabled(True)
        self.omegaEdit.setEnabled(True)
        self.xEdit.setEnabled(True)
        self.vEdit.setEnabled(True)
        self.redefButton.setEnabled(True)


    def disable(self):
        self.thetaEdit.setEnabled(False)
        self.omegaEdit.setEnabled(False)
        self.xEdit.setEnabled(False)
        self.vEdit.setEnabled(False)
        self.redefButton.setEnabled(False)


    def feedback(self, O, w, x, v, F):
        self.thetaEdit.setText("%5.2f" % (O*180./pi))
        self.omegaEdit.setText("%7.4f" % w)
        self.xEdit.setText("%7.4f" % x)
        self.vEdit.setText("%7.4f" % v)


    def getValues(self):
        O = float(self.thetaEdit.text()) * pi/180.
        w = float(self.omegaEdit.text())
        x = float(self.xEdit.text())
        v = float(self.vEdit.text())
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

        self.logicLabel = QLabel("Fuzzy Logic:")
        self.logicCombo = QComboBox(self)
        self.logicCombo.addItems([ "Zadeh", "Probabilistic", "Einstein",
            "Drastic" ])
        self.inferLabel = QLabel("Inference:")
        self.inferCombo = QComboBox(self)
        self.inferCombo.addItems([ "Mamdani", "Probabilistic", "Zadeh/Mamdani",
            "Dienes-Rescher/Mamdani", "Lukasiewicz/Mamdani", "Godel/Mamdani" ])
        self.defuzzyLabel = QLabel("Defuzzification:")
        self.defuzzyCombo = QComboBox(self)
        self.defuzzyCombo.addItems([ "Centroid", "Bissector", "SOM", "LOM", "MOM" ])

        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self.logicLabel, 0, 0)
        layout.addWidget(self.logicCombo, 0, 1)
        layout.addWidget(self.inferLabel)
        layout.addWidget(self.inferCombo)
        layout.addWidget(self.defuzzyLabel, 2, 0)
        layout.addWidget(self.defuzzyCombo, 2, 1)

        self.enable()
        self.show()


    def enable(self):
        self.logicCombo.setEnabled(True)
        self.inferCombo.setEnabled(True)
        self.defuzzyCombo.setEnabled(True)


    def disable(self):
        self.logicCombo.setEnabled(False)
        self.inferCombo.setEnabled(False)
        self.defuzzyCombo.setEnabled(False)


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
        self.ctrlFrame = ControlFrame(self)
        self.redefFrame = RedefFrame(self)
        self.configFrame = ConfigFrame(self)
        self.ruleLabel = QLabel("Show Rule:")
        self.ruleCombo = QComboBox(self)
        self.ruleCombo.addItems([
          'MGN & GN -> MMGN', 'MGN & PN -> MMGN', 'MGN & Z -> MGN', 'MGN & PP -> GN', 'MGN & GP -> PN',
          'GN & GN -> MMGN', 'GN & PN -> MGN', 'GN & Z -> GN', 'GN & PP -> PN', 'GN & GP -> Z',
          'PN & GN -> MGN', 'PN & PN -> GN', 'PN & Z -> PN', 'PN & PP -> Z', 'GN & GP -> PP',
          'Z & GN -> GN', 'Z & PN -> PN', 'Z & Z -> Z', 'Z & PP -> PP', 'Z & GP -> GP',
          'PP & GN -> PN', 'PP & PN -> Z', 'PP & Z -> PP', 'PP & PP -> GP', 'PP & GP -> MGP',
          'GP & GN -> Z', 'GP & PN -> PP', 'GP & Z -> GP', 'GP & PP -> MGP', 'GP & GP -> MMGP',
          'MGP & GN -> PP', 'MGP & PN -> GP', 'MGP & Z -> MGP', 'MGP & PP -> MMGP', 'MGP & GP -> MMGP'
        ])
        self.ruleCombo.setCurrentIndex(17) # Z & Z -> Z
        self.ruleCombo.setEnabled(False)

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
        layout.addWidget(self.ctrlFrame, 0, 1)
        layout.addWidget(self.redefFrame, 1, 1)
        layout.addWidget(self.configFrame, 2, 1)
        layout.addWidget(self.ruleLabel, 3, 1)
        layout.addWidget(self.ruleCombo, 4, 1)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 0)
        layout.setRowStretch(3, 0)
        layout.setRowStretch(4, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        self.feedback(O=0., w=0., x=0., v=0., F=0.)

        # Connects the events
        self.connect(self.ctrlFrame.goButton, SIGNAL("clicked()"), self.onGoButton)
        self.connect(self.ctrlFrame.stopButton, SIGNAL("clicked()"), self.onStopButton)
        self.connect(self.ctrlFrame.stepButton, SIGNAL("clicked()"), self.onStepButton)
        self.connect(self.redefFrame.redefButton, SIGNAL("clicked()"), self.onRedefButton)
        self.connect(self.configFrame.logicCombo, SIGNAL("currentIndexChanged(int)"), self.onLogicCombo)
        self.connect(self.configFrame.inferCombo, SIGNAL("currentIndexChanged(int)"), self.onInferCombo)
        self.connect(self.configFrame.defuzzyCombo, SIGNAL("currentIndexChanged(int)"), self.onDefuzzyCombo)
        self.connect(self.tabs, SIGNAL("currentChanged(int)"), self.onChangeTab)
        self.connect(self.ruleCombo, SIGNAL("currentIndexChanged(int)"), self.onRuleCombo)

        # Exibe o frame
        self.setState(pi/8., 0., 0., 0., 0.)
        self.show()


    def enable(self):
        self.ctrlFrame.enable()
        self.redefFrame.enable()
        self.configFrame.enable()

    def disable(self):
        self.ctrlFrame.disable()
        self.redefFrame.disable()
        self.configFrame.disable()


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


    def setState(self, O, w, x, v, F):
        self.Otrack = [ O ]
        self.wtrack = [ w ]
        self.xtrack = [ x ]
        self.vtrack = [ v ]
        self.Ftrack = [ F ]
        self.ip.setState(O, w, x, v)
        self.feedback(O, w, x, v, F)


    def feedback(self, O, w, x, v, F):
        ci = self.tabs.currentIndex()
        if ci == 0:   # Pendulum
            self.ipview.setState(O, w, x, v, F)
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
            rule = self.ruleCombo.currentIndex()
            _, sF = self.pc.eval(rule, (O, w))
            if sF is None:
                sF = zeros(ip.F.shape)
            self.Fgraph.setData(0, ip.F, rF)
            self.Fgraph.setData(1, ip.F, sF)
        self.redefFrame.feedback(O, w, x, v, F)


    def step(self):
        O, w, x, v = self.ip.getState()
        F = self.pc(O, w)
        self.ip.apply(F)
        self.feedback(O, w, x, v, F)
        self.Otrack.append(O)
        self.wtrack.append(w)
        self.xtrack.append(x)
        self.vtrack.append(v)
        self.Ftrack.append(F)


    def onGoButton(self):
        self.disable()
        self.running = True
        while self.running:
            self.step()
            self.app.processEvents()
        self.enable()


    def onStopButton(self):
        self.running = False


    def onStepButton(self):
        if self.running:
            return
        self.step()


    def onRedefButton(self):
        if self.running:
            return
        O, w, x, v = self.redefFrame.getValues()
        self.Otrack = [ ]
        self.wtrack = [ ]
        self.xtrack = [ ]
        self.vtrack = [ ]
        self.Ftrack = [ ]
        self.setState(O, w, x, v, 0)


    def onLogicCombo(self, index):
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


    def onInferCombo(self, index):
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


    def onDefuzzyCombo(self, index):
        if index == 0:     # Centroide:
            self.pc.defuzzify = Centroid
        elif index == 1:   # Bisseccao
            self.pc.defuzzify = Bissector
        elif index == 2:   # SOM
            self.pc.defuzzify = SmallestOfMaxima
        elif index == 3:   # LOM
            self.pc.defuzzify = LargestOfMaxima
        elif index == 4:   # MOM
            self.pc.defuzzify = MeanOfMaxima


    def onChangeTab(self, index):
        if index == 0:   # Pendulum
            O = self.Otrack[-1]
            w = self.wtrack[-1]
            x = self.xtrack[-1]
            v = self.vtrack[-1]
            F = self.Ftrack[-1]
            self.ipview.setState(O, w, x, v, F)
            self.ruleCombo.setEnabled(False)
        elif index == 1: # Plots
            t = arange(0., 2.5, self.ip.dt)
            self.graph.setMultiData( [
               (t, self.Otrack), (t, self.wtrack),
               (t, self.xtrack), (t, self.vtrack),
               (t, zeros(t.shape)) #self.Ftrack)
            ])
            self.ruleCombo.setEnabled(False)
        elif index == 2: # Membership
            self.ruleCombo.setEnabled(True)


    def onRuleCombo(self, index):
        O = self.Otrack[-1]
        w = self.wtrack[-1]
        x = self.xtrack[-1]
        v = self.vtrack[-1]
        F = self.Ftrack[-1]
        self.feedback(O, w, x, v, F)


    def closeEvent(self, event):
        self.onStopButton()
        self.app.exit(0)


################################################################################
# Main Program
################################################################################
if __name__ == "__main__":
    q = QApplication([])
    f = IPFrame(q, None)
    q.exec_()