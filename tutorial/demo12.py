# -*- coding: utf-8 -*-

#####################################################################
# Peach - Python para Inteligência Computacional
# José Alexandre Nalon
#
# Este arquivo: demo12.py
# Métodos de defuzzificação.
#####################################################################


import numpy
from peach.fuzzy import *


# Explicação deste demo:
#
# Existem diversos métodos diferentes de fazer a defuzzificação
# de uma função de pertinência. Este demo mostra em um único
# gráfico, para uma função dada, o resultado obtido com cada
# um dos métodos disponíveis na biblioteca.


# Cria as funções a serem plotadas.
y = numpy.linspace(-30.0, 30.0, 500)
gn = Triangle(-30.0, -20.0, -10.0)(y)
pn = Triangle(-20.0, -10.0, 0.0)(y)
z = Triangle(-10.0, 0.0, 10.0)(y)
pp = Triangle(0.0, 10.0, 20.0)(y)
gp = Triangle(10.0, 20.0, 30.0)(y)

# Suposto resultado do controlador
mf = gn & 0.33 | pn & 0.67 | z & 0.25

# Pontos de defuzzificacao
centroide = Centroid(mf, y)
bissec = Bissector(mf, y)
som = SmallestOfMaxima(mf, y)
lom = LargestOfMaxima(mf, y)
mom = MeanOfMaxima(mf, y)


# Se o sistema tiver o pacote gráfico matplotlib instalado,
# então o demo tenta criar um gráfico mostrando os pontos
# de defuzzificação. O gráfico é salvo no arquivo demo11.eps.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8, 4)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])
    ll = [ 0.0, 1.0 ]

    a1.hold(True)
    a1.plot([ centroide, centroide ], ll, linewidth = 1)
    a1.plot([ bissec, bissec ], ll, linewidth = 1)
    a1.plot([ som, som ], ll, linewidth = 1)
    a1.plot([ lom, lom ], ll, linewidth = 1)
    a1.plot([ mom, mom ], ll, linewidth = 1)
    a1.fill(y, mf, 'gray')
    a1.set_xlim([ -30, 30 ])
    a1.set_ylim([ -0.1, 1.1 ])
    a1.set_xticks(linspace(-30, 30, 7.0))
    a1.set_yticks([ 0.0, 1.0 ])
    a1.legend([ u'Centróide = %7.4f' % centroide,
                 'Bissector = %7.4f' % bissec,
                 'SOM = %7.4f' % som,
                 'LOM = %7.4f' % lom,
                 'MOM = %7.4f' % mom ])
    savefig("demo12.eps")

except ImportError:
    print "Pontos de defuzzificação:"
    print "  Centróide = %7.4f" % centroide
    print "  Bissecção = %7.4f" % bissec
    print "  SOM = %7.4f" % som
    print "  LOM = %7.4f" % lom
    print "  MOM = %7.4f" % mom