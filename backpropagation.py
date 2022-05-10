import math as m

class Matrix():
    def __init__(self, input_matrix):
       self.matrix = input_matrix
       self.shape = (len(input_matrix), len(input_matrix[0]))
       self.row = self.shape[0]
       self.col = self.shape[1]
    
    def matrix_multiplication(self, sec_matrix): #matrix
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
    def nprint(self):
        print(self.matrix)
        
    def format(self, value):
        return float("%0.3f" % value)
    
    def transpose(self):
        m = []
        for row in range(len(self.matrix[0])):
            tmp = []
            for col in range(len(self.matrix)):
                tmp.append(self.matrix[col][row])
            m.append(tmp)
        return Matrix(m)
        
class layer(): 
    def __init__(self, weight_matrixs):
        self.weights = weight_matrixs
        
    def output(self , out):
        for weight in self.weights:
            out = self.forward(out, weight)
        return out #Matrix class

    def precised_forward(self,in_matrix,weight):
        for weight in self.weights:
            in_matrix = weight.matrix_multiplication(in_matrix)
        return in_matrix
    
    def forward(self, in_matrix, weight):
        out = weight.matrix_multiplication(in_matrix) # O = W I
        out = self.matrix_sigmoid(out) #Applying sigmoid for matrix
        return out #Matrix class
        
    def matrix_sigmoid(self, mat):
        m = [list(map(self.sigmoid, row)) for row in mat.matrix]
        return Matrix(m)
    
    def sigmoid(self, x):
        return 1 / (1+ m.exp(-x))
    
    def learn(self, i, label, learning_rate=0.001, repeat=1000):
        for x in range(repeat):
            
            error = self.error(i, label) # output error
            for idx in range(len(self.weights)-1,-1,-1):
                weight = self.weights[idx]  #각 weight 뒤에서부터 가져오기(matrix)
                error = self.back(weight ,error) #뒤로 한칸 옮겨서 error 계산
                sum_weight = [sum(x) for x in weight.transpose().matrix]
          
                
                self.weights[idx] = weight

        
    def back(self, weight, error):
        return weight.transpose().matrix_multiplication(error)
        
    
    def error(self, in_matrix , label): 
        errors = []
        output = self.output(in_matrix);
        
        for row in range(len(output.matrix)):
            tmp = []
            for col in range(len(output.matrix[0])):
                tmp.append ((label.matrix[row][col] - output.matrix[row][col])**2)
            errors.append(tmp)
       
        return Matrix(errors)
        

#Example
i =[[0.9], [0.1], [0.8]]
w_ih = [[0.9,0.3,0.4],[0.2,0.8,0.2],[0.1,0.5,0.6]] # input -> hidden weight
w_ho = [[0.3,0.7,0.5],[0.6,0.5,0.2],[0.8,0.1,0.9]] # hidden -> output  weight
target = [[0.6],[0.8],[0.5]]

input_matrix = Matrix(i)
weight_matrixs = [Matrix(w_ih), Matrix(w_ho)]
label = Matrix(target)

l = layer(weight_matrixs)
o = l.output(input_matrix)
l.learn(input_matrix, label, learning_rate=0.5, repeat= 10000)
output = l.output(input_matrix)
o.print()
output.print()

