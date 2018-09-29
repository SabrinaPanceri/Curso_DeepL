#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:06:10 2018

@author: sabrinapanceri
"""

import numpy, os

def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

w1 = numpy.random.rand()
w2 = numpy.random.rand()
b = numpy.random.rand()



#NN(3, 1.5, w1, w2, b) 


phrases = ['seems like its ', 'I guess ' , 'I think ', 'possibility ', 'looks like ', 'guessing...']

data = [[3, 1.5, 1], [2, 1, 0], [4, 1.5, 1], [5.5, 1, 1]]

rand_data = data[numpy.random.randint(len(data))]

m1 = rand_data[0]
m2 = rand_data[1]

prediction = NN(m1, m2, w1, w2, b)
prediction_text = ["blue", "red"][int(numpy.round(prediction))]
phrase = numpy.random.choice(phrases) + " " + prediction_text

o = os.system("say " + phrase)
print(phrase)
print(m1, m2)
print(w1, w2, b)
print("It's really " + ["blue", "red"][rand_data[2]])

