# -*- coding: utf-8 -*-

#####################################################################
# Peach - Python para Inteligência Computacional
# José Alexandre Nalon
#
# Este arquivo: demo16.py
# Demonstração e teste, recozimento simulado
#####################################################################

from numpy import *
from numpy.linalg import inv
import peach as p

# Explicação deste demo.
#
# Este demo mostra o processo de otimização conforme obtido por
# diversos métodos para uma função simples. A função de Rosenbrock
# para duas variáveis, dada abaixo, costuma ser utilizada para
# comparação entre diversos métodos de otimização, e é usada
# aqui para mostrar o resultado dos métodos implementados no
# Peach. A função de Rosenbrock para duas variáveis é:
#
#   f(x) = (1-x)^2 + 100(y-x^2)^2
#
# A principal característica dessa função é que ela possui um
# mínimo global em x=1, y=1 que é, no entanto, bastante difícil
# de ser encontrado devido às características da inclinação da
# curva próximo ao mínimo.
#
# Este demo utiliza os métodos de quasi-Newton implementados no
# Peach com estimativa precisa da matriz hessiana, e traça o
# gráfico da convergência da resposta para cada um deles.


# Define a função de Rosenbrock para uso
def f(xy):
    x, y = xy
    return (1-x)**2 + (y-x*x)**2

# Gradiente da função de Rosenbrock
def df(xy):
    x, y = xy
    return array( [ -2*(1-x) - 4*x*(y - x*x), 2*(y - x*x) ])


# Número máximo de iterações, haja o que houver
iMax = 2000

# Otimizador pelo método DFP
x = array([ 0.1, 0.2 ])
sa = p.ContinuousSA(f, df)
e = 0.75
xd = [ ]
yd = [ ]
i = 0
while i < iMax:
    xd.append(x[0])
    yd.append(x[1])
    i = i + 1
    x, e = sa.step(x)
xd = array(xd)
yd = array(yd)


# Contornos
x = linspace(0., 2., 250)
y = linspace(0., 2., 250)
x, y = meshgrid(x, y)
z = (1-x)**2 + (y-x*x)**2
levels = exp(linspace(0., 2., 10)) - 0.9


# Se o sistema tiver o pacote gráfico matplotlib instalado,
# então o demo tenta criar um gráfico da convergência das
# estimativas do mínimo. O gráfico é salvo no arquivo
# demo16.eps.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    figure(1).set_size_inches(6, 6)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.grid(True)
    a1.plot(xd, yd)
    a1.contour(x, y, z, levels, colors='k', linewidths=0.75)
    a1.set_xlim([ 0., 2. ])
    a1.set_xticks([ 0., 0.5, 1., 1.5, 2. ])
    a1.set_ylim([ 0., 2. ])
    a1.set_yticks([ 0.5, 1., 1.5, 2. ])
    savefig("demo19.eps")
    print (xd[-1], yd[-1])

except ImportError:
    pass
