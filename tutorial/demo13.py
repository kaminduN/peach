# -*- coding: utf-8 -*-
#####################################################################
# Peach - Python para Inteligência Computacional
# José Alexandre Nalon
#
# Este arquivo: demo13.py
# Superfície de transferência.
#####################################################################


import numpy
from math import pi
from peach.fuzzy import *
from peach.fuzzy.control import *


# Explicação deste demo:
#
# Todo sistema de controle com duas variáveis pode ser visualizado
# na forma de uma superfície de transferência, que mostra a forma
# como a resposta varia em função das duas variáveis de entrada.
# Este demo plota este gráfico para o sistema de controle de um
# pêndulo invertido.

Points = 50

# Cria as funções de pertinência da variável theta, que representa
# a inclinação do pêndulo.
theta = numpy.linspace(-pi, pi, Points)
tgn = DecreasingRamp(-pi/2.0, -pi/4.0)
tpn = Triangle(-pi/2.0, -pi/4.0, 0.0)
tz = Triangle(-pi/4.0, 0.0, pi/4.0)
tpp = Triangle(0.0, pi/4.0, pi/2.0)
tgp = IncreasingRamp(pi/4.0, pi/2.0)


# Cria as funções de pertinência da variável ômega, que repre-
# senta a velocidade angular do pêndulo
omega = numpy.linspace(-pi/2.0, pi/2.0, Points)
wgn = DecreasingRamp(-pi/4.0, -pi/8.0)
wpn = Triangle(-pi/4.0, -pi/8.0, 0.0)
wz = Triangle(-pi/8.0, 0.0, pi/8.0)
wpp = Triangle(0.0, pi/8.0, pi/4.0)
wgp = IncreasingRamp(pi/8.0, pi/4.0)


# Cria as funções de pertinncia da variável F, que representa
# a força aplicada ao carro que equilibra o pêndulo.
f = numpy.linspace(-30.0, 30.0, 500)
fgn = Triangle(-30.0, -20.0, -10.0)(f)
fpn = Triangle(-20.0, -10.0, 0.0)(f)
fz = Triangle(-10.0, 0.0, 10.0)(f)
fpp = Triangle(0.0, 10.0, 20.0)(f)
fgp = Triangle(10.0, 20.0, 30.0)(f)


# Cria o controlador e as regras de decisão. As regras de
# decisão são inseridas no controlador com o uso do método
# add_table. Nesta tabela, cada linha representa um valor
# linguístico da variável theta, cada coluna representa um
# valor linguístico da variável ômega. Cada entrada na tabe-
# la é a resposta para a variável F, segundo às condições
# para aquela linha e coluna das variáveis de entrada.
c = Controller(f, [], Centroid)
c.add_table([ tgn, tpn, tz, tpp, tgp ], [ wgn, wpn, wz, wpp, wgp ],
    [ [ fgn, fgn, fgn, fpn, fz  ],
      [ fgn, fgn, fpn, fz,  fpp ],
      [ fgn, fpn, fz,  fpp, fgp ],
      [ fpn, fz,  fpp, fgp, fgp ],
      [ fz,  fpp, fgp, fgp, fgp ] ] )


# Gera a superfície
fh = numpy.zeros((Points, Points))
for i in range(0, Points):
    for j in range(0, Points):
        # Gera os valores de entrada
        t = (i - Points/2.0) / (Points / 2.0) * pi
        w = (j - Points/2.0) / Points * pi
        fh[i, j] = c(t, w)


# Se o sistema tiver o pacote gráfico matplotlib instalado,
# então o demo tenta criar um gráfico mostrando a superfície
# gerada. O matplotlib é fraco para gráficos tridimensionais,
# mas para manter adequadas as dependências da biblioteca,
# não exigimos mais que isso. O gráfico é salvo no arquivo
# demo12.eps.
try:
    import pylab as p
    import matplotlib.axes3d as p3

    fig = p.figure()
    a1 = p3.Axes3D(fig)

    theta = numpy.outer(theta, numpy.ones((Points, )))
    omega = numpy.outer(numpy.ones((Points, )), omega)
    a1.plot_surface(theta, omega, fh)
    a1.set_xlim([ -pi, pi ])
    a1.set_ylim([ -pi/2, pi/2 ])
    a1.set_xlabel(r'$\theta$')
    a1.set_ylabel(r'$\omega$')
    a1.set_zlabel(r'$F$')
    p.savefig("demo13.eps")

except ImportError:
    pass
