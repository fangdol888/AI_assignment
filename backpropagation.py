import math as m
from tkinter import W
import numpy as np

class layer(): 
    def __init__(self, w):
        self.w = w
        self.U = []
        self.Z = []
        
    def output(self , inp):
        out = inp
        for weight in self.w:
            out = self.forward(out, weight)
        return out 
    
    def forward(self, in_matrix, weight):
        out = weight.dot(in_matrix) # O = W I
        self.U.append(out)
        out = self.sigmoid(out) #Applying sigmoid for matrix
        self.Z.append(out)
        return np.array(out)
    
    def sigmoid(self, x):
        return 1 / (1+ np.exp(-x))
    
    def grad_sigmoid(self, x):
        return self.sigmoid(x)*self.sigmoid(1-x)
        
    def loss(self, x, y):
        error = []
        for weight in self.w:
            error.append(((y- self.output(x))**2).mean())
        return np.array(error)
    
    def grad_loss(self, x, y, w):
        return -((y-self.sigmoid(w.dot(x))).dot(x.T))
    
    def gradient_descent(self, init_x, y,lr=0.01, step_num=100):
        for num in range(step_num):
            x = init_x    
    
            de = self.grad_loss(x, y, self.w[1])
            self.w[1] -= lr * de
            
            dx = np.dot(de, self.w[0].T)
            dw = np.dot(x.T, de)
            
            self.w[0] -= lr * dx

         


#Example
i =[[0.9], [0.1], [0.8]]
w_ih = [[0.9,0.3,0.4],[0.2,0.8,0.2],[0.1,0.5,0.6]] # input -> hidden weight
w_ho = [[0.3,0.7,0.5],[0.6,0.5,0.2],[0.8,0.1,0.9]] # hidden -> output  weight
target = [[0.6],[0.8],[0.5]]

input_matrix = np.array(i)
weight_matrixs = [np.array(w_ih), np.array(w_ho)]
label = np.array(target)

l = layer(weight_matrixs)
o = l.output(input_matrix)
l.gradient_descent(input_matrix, label, lr=0.1, step_num= 10000)
output = l.output(input_matrix)

print(output)
print(label)


