import math as m

class Matrix():
    def __init__(self, input_matrix):
       self.matrix = input_matrix
       self.shape = (len(input_matrix), len(input_matrix[0]))
       self.row = self.shape[0]
       self.col = self.shape[1]
    
    def matrix_multipication(self, sec_matrix): #matrix
        res =[] #결과 저장용
        for row in self.matrix: #first matrix row 추출
            tmp_row = [] #row 임시저장용
            for col in range(0, sec_matrix.col): # second matrix column 추출
                tmp = 0
                for idx in range(0, len(row)): #각 요소 곱하기
                    tmp += row[idx] * sec_matrix.matrix[idx][col]
                tmp_row.append(tmp) #1 row completed
            res.append(tmp_row) # append completed row
        return Matrix(res)
    
    def print(self):
        m = [list(map(self.format, row)) for row in self.matrix]
        print(m)
        
    def format(self, value):
        return float("%0.3f" % value)
        
class layer(): 
    def __init__(self, v, weight_matrixs):
        self.v = v
        self.weights = weight_matrixs
        
    def output(self):
        out = self.v
        for weight in self.weights:
            out = self.forward(out, weight)
        return out #Matrix class

    def forward(self, in_matrix, weight):
        out = weight.matrix_multipication(in_matrix) # O = W I
        out = self.matrix_sigmoid(out) #Applying sigmoid for matrix
        return out #Matrix class
        
    def matrix_sigmoid(self, mat):
        m = [list(map(self.sigmoid, row)) for row in mat.matrix]
        return Matrix(m)
    
    def sigmoid(self, x):
        return round(1 / (1+ m.exp(-x)),3)

#Example
i =[[0.9], [0.1], [0.8]]
w_ih = [[0.9,0.3,0.4],[0.2,0.8,0.2],[0.1,0.5,0.6]] # input -> hidden weight
w_ho = [[0.3,0.7,0.5],[0.6,0.5,0.2],[0.8,0.1,0.9]] # hidden -> output  weight
   
input_matrix = Matrix(i)
weight_matrixs = [Matrix(w_ih), Matrix(w_ho)]

l = layer(input_matrix, weight_matrixs)

output = l.output()
output.print()