import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras
from keras.models import Sequential
#Dense - funcao que define que todos os neurorios estão conectados
from keras.layers import Dense

#rede neural
classificador = Sequential()

#primeira camada oculta
#units = qtde de neuronios da camada oculta
#activation='relu' -> funçao de ativação dos neuronios
#input_dim = qtde de atributos previsores da camada de dados
classificador.add(Dense(units = 16, activation='relu', kernel_initializer='random_uniform', input_dim = 30))

#camada de saida
#units = 1 -> problema de classificação binária, logo, apenas 1 neuronio de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#compilação da rede neural
#optimizer = adam -> equivale a uma versão otimizada da descida do gradiente estocastico
#loss -> função de perda -> para problemas de classificação binária usar o binary_crossentropy
#para problemas de classificação múltipas usar o categorical_crossentropy (exemplo: classificar como cat, dog, wolf, lion, etc... )
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#treinamendo da rede
#batch_size = 10 -> calcula o erro para 10 registro e depois atualiza os pesos das ligação entre os neurônios
#epochs = 100 -> quantidade de vezes que vai rodar o treinamento da rede e ajuste dos pesos na rede
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)