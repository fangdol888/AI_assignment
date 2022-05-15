import math as m
import numpy as np

class layer():
    def __init__(self, w):
        self.w = w
        self.o = []
        
    def output(self , inp):
        out = inp
        for weight in self.w:
            self.o.append(out);
            out = self.forward(out, weight)
        return out 
    
    def forward(self, in_matrix, weight):
        out = weight.dot(in_matrix) # O = W I
        out = self.sigmoid(out) #Applying sigmoid for matrix
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
    
    def get_dw(self,x, y, lr=0.01, step_num=100):
        for num in range(step_num):
            e = y - self.output(x);
            for layer_num in range(len(self.w)-1, 0, -1):
                s = self.sigmoid(self.o[layer_num]) 
                dw = -e.dot((s*(1-s)).T)*o[layer_num-1]
                self.w[layer_num] -= lr * dw
                e = self.w[layer_num].T.dot(e)
                self.output(x)
            
                    

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

l.get_dw(input_matrix, label, lr=0.1, step_num= 1000)

output = l.output(input_matrix)
print(output)
print(label)