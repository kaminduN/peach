# -*- coding: utf-8 -*-

#####################################################################
# Peach - Python para Inteligência Computacional
# José Alexandre Nalon
#
# Este arquivo: demo07.py
# Demonstração e teste, Mapeamento de uma função não linear.
#####################################################################

from numpy import *
import random
import peach as p

# Explicação deste demo.
#
# É possível utilizar uma rede neural para fazer o mapeamento
# de uma função não linear, como uma senóide ou outra seme-
# lhante. A técnica vai exigir uma rede neural mais complexa,
# com uma entrada, mas com uma camada escondida relativamente
# complexa. A camada de saída deve ter como função de ativação
# a identidade, para somar os mapeamentos realizados.

# Criamos aqui a rede neural. N é a ordem do polinômio,
# que deixamos indicada na forma de uma variável para
# permitir fáceis adaptações. A função de ativação é a
# identidade, e o método de aprendizado é o back-propa-
# gation (por default).
# Utilizamos várias saídas, igualmente distribuídas ao
# redor do ponto de avaliação para que o erro obtido seja
# mais significativo, onde existir. Nesse caso, o ponto de
# avaliação será igual a int(inputs/2). O uso de uma vizi-
# nhança maior possibilitará melhores resultados.
inputs = 7
nn = p.FeedForward((inputs, 200, inputs), lrule=p.BackPropagation(0.01), bias=True)
nn.phi = (p.Sigmoid, p.Linear)

delta = linspace(-0.1, 0.1, inputs)
elog = [ ]
error = 1
i = 0
while i < 2000:

    # Geramos um valor de x e um valor da resposta
    # desejada. Com x, encontramos xo, que será o
    # vetor de entrada da rede neural:
    xo = random.uniform(-1.0, 1.0)
    x = xo + delta
    d = sin(pi*x)

    # Fazemos a predição, calculamos o erro e realizamos
    # o aprendizado da rede.
    y = nn(x)
    error = nn.learn(x, d)
    elog.append(error)

    # Incrementamos o contador de tentativas.
    i = i + 1


# Se o sistema tiver o pacote gráfico matplotlib instalado,
# então o demo tenta criar um gráfico da função original,
# contrastada com a função predita. O gráfico é salvo no
# arquivo demo07.eps.
try:
    from matplotlib import *
    from matplotlib.pylab import *

    x = linspace(-1, 1, 200)
    y = sin(pi*x)
    ye = [ ]
    for xo in x:
        yn = nn(delta + xo)
        ye.append(yn[int(inputs/2)])
    ye = array(ye)

    subplot(211)
    hold(True)
    grid(True)
    plot(x, y, 'b--')
    plot(x, ye, 'g')
    xlim([ -1, 1 ])
    legend([ "$y$", "$\hat{y}$" ])

    subplot(212)
    grid(True)
    plot(arange(0, 2000, 10), array(elog, dtype=float)[::10])
    savefig("demo07.eps")

except ImportError:
    pass
