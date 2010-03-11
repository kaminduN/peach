# -*- coding: utf-8 -*-
#####################################################################
# Peach - Python para Inteligência Computacional
# José Alexandre Nalon
#
# Este arquivo: demo18.py
# Demonstração do uso de algoritmos genéticos
#####################################################################

from peach import *
from numpy import *
from matplotlib import *
from pylab import *

# Explicação deste demo
#
# O objetivo desta demonstração é ilustrar o uso do pacote de
# algoritmos genéticos do Peach. Um algoritmo genético básico
# é usado para calcular o ponto de mínimo de função de Rosenbrock
# simplificada. Uma função de fitness baseada nessa função é
# criada. É importante lembrar que algoritmos genéticos sempre
# tentam *maximizar* funções, e não minimizá-las. Por isso, a
# função de fitness inverte o sinal da função a ser minimizada.
#
# O algoritmo genético passa para a função de fitness uma tupla,
# resultado da decodificação dos bits de um cromossomo. Por isso,
# a função de fitness deve estar preparada para receber uma tupla
# para o cálculo. Isso permite o uso de funções com muitos argumen-
# tos sem a necessidade de declará-los cada um. É, obviamente, res-
# ponsabilidade da função de fitness organizar os elementos do ve-
# tor, encontrar seus valores e realizar o cálculo da maneira ade-
# quada.


# Função de Rosenbrock para a otimização
def f(x, y):
    '''
    Função de Rosenbrock
    '''
    return (1-x)**2 + (y-x*x)**2


# Função de fitness para o algoritmo genético. O objetivo é
# minimizar a função de Rosenbrock, então:
def J((x, y)):
    return -f(x, y)


if __name__ == "__main__":

    # Cria e inicializa população. Para criar a população, instancie um
    # objeto da classe GA, do pacote de algoritmos genéticos do Peach.
    # A chamada básica é:
    # GA(fitness, fmt, ranges=[ ], size=50, selection=RouletteWheel,
    #    crossover=TwoPoint, mutation=BitToBit, elitist=True)
    # em que:
    #   fitness: é a função de fitness
    #   fmt: é o formato do cromossomo segundo o módulo struct da linguagem
    #        Python. Por exemplo, para criarmos um cromossomo com dois in-
    #        teiros e dois floats, usamos a string 'iiff'
    #   ranges [opcional]: intervalos nos quais cada uma das variáveis está
    #        definida. Estes valores são usados para sanear resultados in-
    #        desejados obtidos aleatoriamente. Se fornecido, deve ser dado
    #        como uma lista de duplas da forma (x0, x1), onde x0 é o limite
    #        inferior do intervalo, e x1 seu limite superior.
    #   size [opcional]: tamanho inicial da população, default=50
    #   selection [opcional]: método de seleção, default=roleta russa
    #   crossover [opcional]: método de crossover, default=crossover de
    #        dois pontos com probabilidade 0.75. Para mudar a probabilidade,
    #        utilize esse argumento como crossover=TwoPoint(p), em que p é
    #        a probabilidade de crossover
    #   mutation [opcional]: método de mutação, default=bit a bit com
    #        probabilidade de 0.05. Para mudar a probabilidade, utilize esse
    #        argumento como mutation=BitToBit(p), em que p é a probabilidade
    #        de mutação de um bit
    #   elitist [opcional]: determina se a seleção é elitista ou não,
    #        default=true
    #
    # Para o nosso caso, vamos criar uma população de 10 indivíduos, em que
    # cada cromossomo representa 2 números em ponto flutuante:
    P = GA(J, 'ff', ranges=[ (0., 2.), (0., 2.) ], size=10)

    # Inicializamos a população. Esse passo, em geral, não é necessário,
    # pois o próprio conteúdo da memória durante a alocação dos cromossomos
    # garante aleatoriedade suficiente. No entanto, um pouco de heurística
    # ajuda qualquer algoritmo de otimização.
    #
    # Para criar um cromossomo, instanciamos a classe Chromosome, passando
    # para o inicializador o formato do cromossomo. Isso não é explicitamente
    # necessário, pois criar a população já cria os cromossomos. Para setar o
    # valor codificado em um cromossomo, usamos o método .encode(). O argumen-
    # to desse método é uma tupla contendo os valores a serem codificados.
    for i in range(len(P)):
        t = uniform(0., 2*pi)
        r = 0.25 + uniform(0.25)
        x = 1 + r*cos(t)
        y = 1 + r*sin(t)
        P[i].encode((x, y))

    # Realizaremos no máximo 200 iterações, o que pode não ser o suficiente
    # para o algoritmo genético, uma vez que é um algoritmo estocástico e
    # não determinístico. Mas, em geral, com umas poucas iterações, já é
    # possível obter a convergência. Outro critério de parada é o erro co-
    # metido. A função de Rosenbrock tem mínimo em x=1, y=1, por isso, o
    # módulo do erro cometido é
    #
    #   erro = sqrt((x-1)**2 + (y-1)**2)
    #
    # Adicionalmente, temos algumas estruturas de dados para armazenar os
    # resultados parciais para o traçcado de gráficos de convergência.
    Iterations = 200    # Quantidade de iterações
    i = 0
    TrackX = [ ]
    TrackY = [ ]
    Error = [ ]
    e = 1.
    while i < Iterations and e > 0.00001:

        # Para executar um passo da convergência, usamos o método .step()
        # da população. Esse método não tem argumentos nem retorna valores.
        P.step()

        # As linhas abaixo preparam o traçado do gráfico. Uma vez que o
        # nosso exemplo trabalha com uma seleção elitista, o erro deve
        # decrescer monotonicamente.
        m = argmax(P.fitness)
        x, y = P[m].decode()
        if TrackX == [ ]:
            TrackX.append(x)
            TrackY.append(y)
        if x != TrackX[-1] or y != TrackY[-1]:
            TrackX.append(x)
            TrackY.append(y)
        e = sqrt((x-1.)**2 + (y-1.)**2)
        Error.append(e)
        i = i + 1


# Contornos
TrackX = array(TrackX)
TrackY = array(TrackY)
Error = array(Error)
x = linspace(0., 2., 250)
y = linspace(0., 2., 250)
x, y = meshgrid(x, y)
z = f(x, y)


figure(1)
a1 = axes([ 0.125, 0.125, 0.775, 0.775 ])
figure(1).set_size_inches((6, 6))

# Plota a figura
levels = exp(linspace(0., 2., 10)) - 0.9
a1.grid(True)
a1.hold(True)
a1.plot(TrackX, TrackY, 'k-')
a1.plot([ TrackX[0] ], [ TrackY[0] ], 'ko')
a1.plot([ TrackX[-1] ], [ TrackY[-1] ], 'ko')
a1.contour(x, y, z, levels, colors='k', linewidths=0.75)
a1.set_xlim([ 0., 2. ])
a1.set_xticks([ 0., 0.5, 1., 1.5, 2. ])
a1.set_ylim([ 0., 2. ])
a1.set_yticks([ 0.5, 1., 1.5, 2. ])
a1.legend([ '%d pontos' % len(TrackX) ])

#Salva a figura
savefig('AGRosenbrock.eps')


# Nova figura
figure(1).clear()
a1 = axes([ 0.125, 0.125, 0.775, 0.775 ])
figure(1).set_size_inches((8, 4))
a1.grid(True)
a1.plot(Error)

# Salva a figura
savefig('AGConvergence.eps')