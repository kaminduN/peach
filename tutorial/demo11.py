# -*- coding: utf-8 -*-
#####################################################################
# Peach - Python para Inteligência Computacional
# José Alexandre Nalon
#
# Este arquivo: demo11.py
# Normas e conormas
#####################################################################

import numpy
from peach.fuzzy import *
from peach.fuzzy.norms import *


# Explicação deste demo:
#
# Diversas normas (que implementam a operação lógica "e") e conormas
# (que implementam a operação lógica "ou") estão disponíveis para o
# uso com lógica fuzzy. Muitas delas estão disponíveis no Peach.
# Este demo mostra o seu uso.


# Cria as funcoes a serem plotadas.
x = numpy.linspace(-5.0, 5.0, 500)
a = Triangle(-3.0, -1.0, 1.0)(x)
b = Triangle(-1.0, 1.0, 3.0)(x)

# Seta as normas para as tradicionais sugeridas por Lofti Zadeh
# Este passo na verdade não é necessário, pois são as normas default.
a.set_norm(ZadehAnd)
a.set_conorm(ZadehOr)
aeb_zadeh = a & b
aoub_zadeh = a | b


# Usa as normas probabilísticas.
a.set_norm(ProbabilisticAnd)
a.set_conorm(ProbabilisticOr)
aeb_prob = a & b
aoub_prob = a | b


# Se o sistema tiver o pacote gráfico matplotlib instalado,
# então o demo tenta criar um gráfico mostrando o resultado
# da aplicação de cada uma das normas sobre os conjuntos.
# O gráfico é salvo no arquivo demo10.eps.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(8, 6)
    a1 = axes([ 0.125, 0.555, 0.775, 0.40 ])
    a2 = axes([ 0.125, 0.125, 0.775, 0.40 ])

    a1.hold(True)
    a1.plot(x, a, 'k:')
    a1.plot(x, b, 'k:')
    a1.plot(x, aeb_zadeh, 'k')
    a1.plot(x, aeb_prob, 'k-.')
    a1.set_xlim([ -5, 5 ])
    a1.set_ylim([ -0.1, 1.1 ])
    a1.set_xticks([])
    a1.set_yticks([ 0.0, 1.0 ])
    a1.legend((r'$A$', r'$B$', 'Zadeh AND', 'Prob AND'))

    a2.hold(True)
    a2.plot(x, a, 'k:')
    a2.plot(x, b, 'k:')
    a2.plot(x, aoub_zadeh, 'k')
    a2.plot(x, aoub_prob, 'k-.')
    a2.set_xlim([ -5, 5 ])
    a2.set_ylim([ -0.1, 1.1 ])
    a2.set_xticks([])
    a2.set_yticks([ 0.0, 1.0 ])
    a2.legend((r'$A$', r'$B$', 'Zadeh OR', 'Prob OR'))

    savefig("demo11.eps")

except ImportError:
    pass
