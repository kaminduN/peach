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

# Decodifica inteiros em flutuante
def decode((x, y)):
    return (2.0*x/4294967295L , 2.0*y/4294967295L)

def fn(xy):
    x, y = decode(xy)
    return f((x, y))


# Número máximo de iterações, haja o que houver
iMax = 2000

# Otimizador pelo método do recozimento simulado
x = array([ int(0.1*4294967295L/2.0), int(0.2*4294967295L/2.0) ])
sa = p.DiscreteSA(fn, 'LL', nb=16, rt=0.995)
e = 0.75
i = 0
while i < iMax:
    i = i + 1
    x, e = sa.step(x)
print decode(x)
