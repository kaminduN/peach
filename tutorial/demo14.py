# -*- coding: utf-8 -*-

#####################################################################
# Peach - Python para Inteligência Computacional
# José Alexandre Nalon
#
# Este arquivo: demo14.py
# Demonstração e teste, minimização de funções
#####################################################################

from numpy import *
import peach as p

# Explicação deste demo.
#
# Este demo mostra o processo de otimização conforme obtido por
# diversos métodos para uma função simples. A função de Rosenbrock,
# dada abaixo, costuma ser utilizada para comparação entre diversos
# métodos de otimização, e é usada aqui para mostrar o resultado
# dos métodos implementados no Peach. A função de Rosenbrock é:
#
#   f(x) = (1-x)^2 + 100(1-x^2)^2
#
# para funções unidimensionais. A principal característica dessa
# função é que ela possui um mínimo global em x=1 que é, no entanto,
# bastante difícil de ser encontrado devido às características da
# inclinação da curva próximo ao mínimo.
#
# Este demo utiliza os métodos implementados no Peach e traça o
# gráfico da convergência da resposta e do erro para cada um
# deles.


# Define a função de Rosenbrock para minimização
# Simplificada apenas para demonstração.
def f(x):
    '''
    Função de Rosenbrock
    '''
    return (1-x)**2 + (1-x*x)**2


# Derivada para método do gradiente. Tecnicamente, se a derivada
# da função estiver disponível, ela é estimada automaticamente
# pelo algoritmo.
def df(x):
    return -2*(1-x) - 4*(1-x*x)*x


# Número máximo de iterações, haja o que houver
iMax = 100


# Cria-se o otimizador pela busca direta. O único dado de
# inicialização é a função a ser otimizada -- outros parâmetros
# estão disponíveis, mas são usados seus valores default.
linear = p.Direct1D(f)
x = 0.75               # Estimativa inicial
e = 0.75               # Erro inicial
xl = [ ]               # Retém o traço da convergência
el = [ ]
i = 0
while i < iMax:
    xl.append(x)
    el.append(e)
    i = i + 1
    x, e = linear.step(x)
xl = array(xl)
el = array(el)


# Otimizador por interpolação quadrática. O único dado de
# inicialização necessário é a função a ser otimizada.
interp = p.Interpolation(f)
x = (0., 0.75, 1.5)   # Estimativa inicial
e = 0.75              # Erro inicial
xp = [ ]              # Retém o traço da convergência
ep = [ ]
i = 0
while i < iMax:
    try:
        x0, x1, x2 = x
        q0 = x0 * (f(x1) - f(x2))
        q1 = x1 * (f(x2) - f(x0))
        q2 = x2 * (f(x0) - f(x1))
        xm = 0.5 * (x0*q0 + x1*q1 + x2*q2) / (q0 + q1 + q2)
    except ZeroDivisionError:
        xm = x0
    xp.append(xm)
    ep.append(e)
    i = i + 1
    try:
        x, e = interp.step(x)
    except ZeroDivisionError:
        break
xp = array(xp)
ep = array(ep)


# Otimizador pela seção áurea. O único dado necessário é a
# função a ser otimizada.
golden = p.GoldenRule(f)
x = (0.25, 1.25)
e = 0.75
xg = [ ]
eg = [ ]
i = 0
while i < iMax:
    xo, xh = x
    xm = 0.5 * (xo+xh)
    xg.append(xm)
    eg.append(e)
    i = i + 1
    x, e = golden.step(x)
xg = array(xg)
eg = array(eg)


# Otimizador pelo gradiente
grad = p.Gradient(f, df)
x = 0.84
e = 0.75
xd = [ ]
ed = [ ]
i = 0
while i < iMax:
    xd.append(x)
    ed.append(e)
    i = i + 1
    x, e = grad.step(x)
xd = array(xd)
ed = array(ed)


# Se o sistema tiver o pacote gráfico matplotlib instalado,
# então o demo tenta criar um gráfico da convergência das
# estimativas do mínimo. O gráfico é salvo no arquivo
# demo14.eps.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    vsize = 4
    figure(1).set_size_inches(8, 4)
    a1 = axes([ 0.125, 0.10, 0.775, 0.8 ])

    a1.hold(True)
    a1.grid(True)
    a1.plot(xl)
    a1.plot(xp)
    a1.plot(xg)
    a1.plot(xd)
    a1.legend([ "Linear", u"Interpolação", u"Seção Áurea", "Gradiente" ])
    savefig("demo14.eps")

except ImportError:
    print "Otimizador linear: ", xl[-1]
    print "Otimizador por interpolação quadrática: ", xp[-1]
    print "Otimizador pela seção áurea: ", xg[-1]
    print "Otimizador pelo gradiente: ", xd[-1]
