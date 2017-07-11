#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from exl import readexcel
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

'''
    4组训练样本
    输入层: 4 节点
    隐藏层: 7 节点 + 1 阈值节点(输出为1, 权值为bj)
    输出层: 1 节点

    前向传输:
        第一层传输: for A in examples for j in range(0,7) B[j] = 0 for i in range(0,4) B[j] += A[i] * A[j][i]
        第二层传输: 
    反向传播:
        隐藏到输出: for i in range(0,7) for j in range(0,1) deltw = (y-t) * y * (1-y) * output(y)

'''

# sigmoid(x) = 1/(1+exp(-x)), sigmoid'(x) = sigmoid(x)(1 - sigmoid(x))
# th(x) = (exp(x) - exp(-x)/(exp(x) + exp(-x)), th'(x) = 1 - th(x) * th(x)
def act_function(x, z):
    if z == "sigmoid" :
        return 1/(1 + np.exp(-x))
    elif z == "th":
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    elif z == "linear":
        return x
    else:
        raise ValueError('invalid value: %s' % z)

def act_function_derivatives(y, z = "sigmoid" ):
    if z == "sigmoid":
        return y * (1 - y)
    elif z == "th":
        return 1 - y * y
    elif z == "linear":
        return 1
    else:
        raise ValueError('invalid value: %s' % z)

# x = (x - mu) / sigma, 返回数据集中每一个属性的 mu 和 sigma
def Normalized(data_set):
    m = len(data_set)
    n = len(data_set[0])
    mu = [np.average([data_set[i][j] for i in range(m)]) for j in range(n)]
    sigma = [np.std([data_set[i][j] for i in range(m)]) for j in range(n)]
    normal = [[(data_set[i][j] - mu[j]) / sigma[j] for j in range(n)]  for i in range(m)]
    return normal, mu, sigma



class BP:
    
    # _input_examples_values
    # _output_examples_values
    # _hiding_nodes
    # _output_nodes
    # _input_w
    # _output_w
    # _lamda
    
    def __init__(self, input_examples_values = None, output_examples_values = None, examples = None, hiding_nodes_number = 8, lam = 0.7):
        # examples[4][0][4] [4][1][1]
        # self.examples = examples
        if examples:
            # _input_examples_nodes[4][4], 保存数据集中的所有输入数据的值
            self._input_examples_values = [p[0] + [1] for p in examples]
            # _output_examples_values[4][1], 保存数据集中的所有输出数据的值
            self._output_examples_values = [p[1] for p in examples]
            # 训练样本数
            sample_num = len(examples)
        elif input_examples_values != None and output_examples_values != None:
            # _input_examples_nodes[4][4], 保存输入数据集中的所有数据的值
            self._input_examples_values = [x + [1] for x in input_examples_values]
            # _output_examples_values[4][1], 保存输出数据集中的所有数据的值
            self._output_examples_values = output_examples_values
            # 训练样本数
            sample_num = len(input_examples_values)
        else:
            raise ValueError('没有输入 input_examples_values, output_examples_values 或 examples')
       
        # _hiding_nodes[4][8], 用于保存bp神经网络对于每个输入的隐藏层的值
        self._hiding_nodes = [[(i+1)//hiding_nodes_number for i in range(hiding_nodes_number)] for j in range(sample_num)]
        # _output_nodes[4][1], 用于保存bp神经网络的输出值
        self._output_nodes = [[0] * len(self._output_examples_values[0]) for j in range(sample_num)]
        
        # 输入层 到 隐藏层 权值 w[7][4], w[7][4] * a[4][1] = h[7][1]
        self._input_w = [[random.random() for j in range(len(self._input_examples_values[0]))] for i in range(hiding_nodes_number - 1)]
        # 隐藏层 到 输出层 权值 w[1][7], w[1][7] * h[7][1] = o[1][1]
        self._output_w =  [[random.random() for j in range(hiding_nodes_number)] for i in range(len(self._output_examples_values[0]))]
        # 设置误差导数更新步长
        self._lamda = lam

    # 根据 权值矩阵 和 输入数据 更新 BP神经网络的 输出层数据
    def Forward_propagation(self, _input_eg_values = None ):
        if _input_eg_values == None:
            _input_examples_values = self._input_examples_values
        else:
            _input_examples_values = _input_eg_values
        # 第 m 个训练样本
        for m in range(len(_input_examples_values)):
            # 输入层 到 隐藏层 加权后使用激活函数 sigmoid(x) = 1 / (1+exp(-x))
            # 第 i 个隐藏层节点
            for i in range(len(self._hiding_nodes[0])-1):
                self._hiding_nodes[m][i] = 0
                # 第 j 个输入层节点
                for j in range(len(_input_examples_values[0])):
                    self._hiding_nodes[m][i] += self._input_w[i][j] * _input_examples_values[m][j]
                self._hiding_nodes[m][i] = act_function(self._hiding_nodes[m][i], "sigmoid")
            # 隐藏层 到 输出层 加权后使用激活函数 f(x) = x
            # 第 i 个输出层节点
            for i in range(len(self._output_examples_values[0])):
                self._output_nodes[m][i] = 0
                # 第 j 个隐藏层节点
                for j in range(len(self._hiding_nodes[0])):
                    self._output_nodes[m][i] += self._output_w[i][j] * self._hiding_nodes[m][j]
                self._output_nodes[m][i] = act_function(self._output_nodes[m][i], "linear")

    def Back_Propagation_Batch(self):
        delta_w_out = [[0 for j in range(len(self._hiding_nodes[0]))] for i in range(len(self._output_examples_values[0]))]
        delta_w_in = [[0 for j in range(len(self._input_examples_values[0]))] for i in range(len(self._hiding_nodes[0])-1)]
        # 第 m 个训练样本
        for m in range(len(self._input_examples_values)):
            # 隐藏层 到 输出层 误差导数 delta_w[i][j] = -lamda * x[i] * (y[j] - t[j]) * y[j] * (1 - y[j])
            for i in range(len(self._output_examples_values[0])):
                for j in range(len(self._hiding_nodes[0])):
                    delta_w_out[i][j] += self._hiding_nodes[m][j] * (self._output_nodes[m][i] - self._output_examples_values[m][i]) * act_function_derivatives(self._output_nodes[m][i], z = "linear")
            # 输入层 到 隐藏层 误差导数
            for i in range(len(self._hiding_nodes[0])-1):
                for j in range(len(self._input_examples_values[0])):
                    for k in range(len(self._output_examples_values[0])):
                        delta_w_in[i][j] += self._input_examples_values[m][j] * (self._output_nodes[m][k] - self._output_examples_values[m][k]) * act_function_derivatives(self._output_nodes[m][k], z = "linear") * self._output_w[k][i] * act_function_derivatives(self._hiding_nodes[m][i], "sigmoid")
        # 更新每一个权值
        for i in range(len(self._output_examples_values[0])):
            for j in range(len(self._hiding_nodes[0])):
                self._output_w[i][j] = self._output_w[i][j] - self._lamda * delta_w_out[i][j] / (m + 1)
        for i in range(len(self._hiding_nodes[0])-1):
            for j in range(len(self._input_examples_values[0])):
                self._input_w[i][j] = self._input_w[i][j] - self._lamda * delta_w_in[i][j] / (m + 1)


    def Back_Propagation_Random(self, m):
        delta_w_out = [[0 for j in range(len(self._hiding_nodes[0]))] for i in range(len(self._output_examples_values[0]))]
        delta_w_in = [[0 for j in range(len(self._input_examples_values[0]))] for i in range(len(self._hiding_nodes[0])-1)]
        # 第 m 个训练样本
        # 隐藏层 到 输出层 误差导数 delta_w[i][j] = -lamda * x[i] * (y[j] - t[j]) * y[j] * (1 - y[j])
        for i in range(len(self._output_examples_values[0])):
            for j in range(len(self._hiding_nodes[0])):
                delta_w_out[i][j] = self._hiding_nodes[m][j] * (self._output_nodes[m][i] - self._output_examples_values[m][i]) * act_function_derivatives(self._output_nodes[m][i], z = "linear")
        # 输入层 到 隐藏层 误差导数
        for i in range(len(self._hiding_nodes[0])-1):
            for j in range(len(self._input_examples_values[0])):
                for k in range(len(self._output_examples_values[0])):
                    delta_w_in[i][j] = self._input_examples_values[m][j] * (self._output_nodes[m][k] - self._output_examples_values[m][k]) * act_function_derivatives(self._output_nodes[m][k], z = "linear") * self._output_w[k][i] * act_function_derivatives(self._hiding_nodes[m][i], "sigmoid")
        # 更新每一个权值
        for i in range(len(self._output_examples_values[0])):
            for j in range(len(self._hiding_nodes[0])):
                self._output_w[i][j] = self._output_w[i][j] - self._lamda * delta_w_out[i][j]
        for i in range(len(self._hiding_nodes[0])-1):
            for j in range(len(self._input_examples_values[0])):
                self._input_w[i][j] = self._input_w[i][j] - self._lamda * delta_w_in[i][j]
    
    def Mean_Squared_Error(self, mu_out, sigma_out, _output_eg_values = None, _out_nodes = None):
        if _output_eg_values != None and _out_nodes != None:
            _output_examples_values = _output_eg_values
            _output_nodes = _out_nodes
        elif _output_eg_values != None and _out_nodes == None:
            _output_examples_values = _output_eg_values
            _output_nodes = self._output_nodes
        else:
            _output_examples_values = self._output_examples_values
            _output_nodes = self._output_nodes
        e = 0
        m = 0
        # BP网络输出 和 训练集中输出值 的范数之差的平方，后将所有训练集中的样本的误差累加到一起
        for m in range(len(_output_examples_values)):
            o = 0
            ov = 0
            
            for i in range(len(_output_examples_values[0])):
                value = _output_examples_values[m][i] * sigma_out[i] + mu_out[i]
                o += value * value
            for i in range(len(_output_nodes[0])):
                value = _output_nodes[m][i] * sigma_out[i] + mu_out[i]
                ov += value * value
            o = np.sqrt(o)
            ov = np.sqrt(ov)
            e += (o - ov) * (o - ov)
        mse = e/(m + 1)
        return mse
            
            
    def train_batch(self, mu_out, sigma_out, time = 500, print_state = False):
        self.Forward_propagation()
        error = []
        print("训练次数 , Error:")
        for i in range(time):
            # error_before = self.Mean_Squared_Error( mu_out, sigma_out,)
            self.Back_Propagation_Batch()
            self.Forward_propagation()
            error_after = self.Mean_Squared_Error( mu_out, sigma_out,)
            error.append(error_after)
            if (i + 1) % 50 == 0:
                print(i + 1, ":", error_after)
        if print_state:
            print("example input\t actual output\t predicted output")
            for i in range(len(self._input_examples_values)):
                print(self._input_examples_values[i], self._output_examples_values[i], self._output_nodes[i])
        return error
 
    def train_random(self, mu_out, sigma_out, time = 500, print_state = False):
        self.Forward_propagation()
        error = []
        print("训练次数 , Error:")
        acc = 0
        for i in range(time):
            for m in range(len(self._input_examples_values)):
                
                # error_before = self.Mean_Squared_Error( mu_out, sigma_out,)
                self.Back_Propagation_Random(m)
                self.Forward_propagation()
                error_after = self.Mean_Squared_Error( mu_out, sigma_out,)
                
                if acc % 50 == 0:
                    print(acc , ":", error_after)
                acc += 1
            error.append(error_after)
            if (i + 1) % 50 == 0:
                print("iteration", i + 1, ":", error_after)
        if print_state:
            print("example input\t actual output\t predicted output")
            for i in range(len(self._input_examples_values)):
                print(self._input_examples_values[i], self._output_examples_values[i], self._output_nodes[i])
        return error 

    def test(self, mu_out, sigma_out, input_test_values = None, output_test_values = None, test_eg = None, print_state = False):
        if test_eg:
            _input_examples_values = [p[0] for p in test_eg]
            _output_examples_values = [p[1] for p in test_eg]
            test_num = len(test_eg)
        elif input_test_values != None and output_test_values != None:
            _input_examples_values = input_test_values
            _output_examples_values = output_test_values
            test_num = len(input_test_values)
        else:
            raise ValueError('没有输入 input_test_values, output_test_values 或 test_eg')
        
        self.Forward_propagation(_input_examples_values)
        error = self.Mean_Squared_Error(mu_out, sigma_out, _output_eg_values = _output_examples_values)
        
        if print_state:
            print("\n测试数据集 Error:", error)
            #print("test input\t actual output\t predicted output")
            #for i in range(test_num):
            #print(_input_examples_values[i], _output_examples_values[i], self._output_nodes[i])
        return error

    def hyber_train_batch(self, mu_out, sigma_out, time = 500, input_test_values = None, output_test_values = None, test_eg = None, print_state = False):
        self.Forward_propagation()
        error_train = []
        error_test = []
        print("训练次数 , Error:")
        for i in range(time):
            # error_before = self.Mean_Squared_Error( mu_out, sigma_out,)
            self.Back_Propagation_Batch()
            error_test.append(self.test(mu_out, sigma_out, input_test_values, output_test_values))
            self.Forward_propagation()
            error_after = self.Mean_Squared_Error( mu_out, sigma_out,)
            error_train.append(error_after)
            if (i + 1) % 50 == 0:
                print(i + 1, ":", error_after)
        if print_state:
            print("example input\t actual output\t predicted output")
            for i in range(len(self._input_examples_values)):
                print(self._input_examples_values[i], self._output_examples_values[i], self._output_nodes[i])
        return error_train, error_test

def import_data():
    dir = [x for x in os.listdir('.') if os.path.isdir(x) or '.xls' in os.path.splitext(x)[1]]
    print(dir)
    filebox = "./" + input("输入导入的文件/文件所在文件夹")
    while os.path.isdir(filebox):
        print([x for x in os.listdir(filebox) if os.path.isdir(x) or '.xls' in os.path.splitext(x)[1]])
        filebox += "/"+input("输入导入的文件/文件所在文件夹")
    if os.path.isfile(filebox):
        return readexcel(filebox)

def show_error(error_train, iteration, unit = 1, error_test = None):
    plt.figure(1)
    x = [x*unit for x in range(iteration//unit)]
    y = [error_train[y*unit] for y in range(iteration//unit)]
    plt.plot(x, y, 'r')
    if error_test != None:
        x = [x*unit for x in range(iteration//unit)]
        y = [error_test[y*unit] for y in range(iteration//unit)]
        plt.plot(x, y, 'g')
    
    
    plt.axis([0, iteration, 0, np.max(y)])
    plt.xlabel("iteration")
    plt.ylabel("MSE")
    
    plt.show()

def main():
    examples_1 = [
                  [[1, -1, 0, 0], [1,]],
                  [[0, 1, 0, -1], [-1,]],
                  [[1, 1, -1, 0], [2,]],
                  [[-1, -1, -1, -1], [-2,]],
                  ]
    test_1 = None

    examples = examples_1

    print("import 输入数据集")
    l_in = import_data()
    print("import 输出数据集")
    l_out = import_data()
    
    
    print("训练集样本数：", len(l_in), len(l_out))
    
    if len(l_in) < 5:
        raise ValueError("数据集样本太少")
    
    if len(l_in) != len(l_out):
        raise ValueError("数据集样本数不匹配")
    
    
    while True:
        train_num = int(input("输入训练样本数（其他样本用于测试）："))
        if abs(train_num) < len(l_in) :
            break
        else:
            print("输入有误, len(l_in) = ", len(l_in))

    hiding_number = 8
    iteration = 100
    unit = 5

    l_in = Normalized(l_in)[0]
    l_out, mu_out, sigma_out = Normalized(l_out)

    l_in_eg = l_in[:train_num]
    l_out_eg = l_out[:train_num]
    l_in_test = l_in[train_num:]
    l_out_test = l_out[train_num:]

    bp = BP(input_examples_values = l_in_eg, output_examples_values = l_out_eg, hiding_nodes_number = hiding_number)
    error = bp.train_batch(mu_out, sigma_out, time = iteration, print_state = False)
    bp.test(mu_out, sigma_out, l_in_test, l_out_test)
    show_error(error, iteration, unit)

def demo():
    l_in = readexcel("/Users/Welkin/Documents/人工智能/BackwardPropagation/data4/chemicalInputs.xlsx")
    l_out = readexcel("/Users/Welkin/Documents/人工智能/BackwardPropagation/data4/chemicalTargets.xlsx")
    print("训练集样本数：", len(l_in), len(l_out))
    
    l_in_nor = Normalized(l_in)[0]
    l_out_nor, mu_out, sigma_out = Normalized(l_out)
    
    hiding_number = 8
    train_num = 400
    iteration = 200
    unit = 5
    
    bp_1 = BP(input_examples_values = l_in_nor[:train_num], output_examples_values = l_out_nor[:train_num], hiding_nodes_number = hiding_number, lam = 0.6)
    '''
    error_train = bp_1.train_batch(mu_out, sigma_out, time = iteration, print_state = False)
    bp_1.test(mu_out, sigma_out, l_in_nor[train_num:], l_out_nor[train_num:], print_state = True)
    show_error(error_train, iteration, unit)
    '''
    error_train, error_test = bp_1.hyber_train_batch(mu_out, sigma_out, time = iteration, input_test_values = l_in_nor[train_num:], output_test_values = l_out_nor[train_num:])
    bp_1.test(mu_out, sigma_out, l_in_nor[train_num:], l_out_nor[train_num:], print_state = True)
    show_error(error_train, iteration, unit, error_test)
    '''
    bp_2 = BP(input_examples_values = l_in_nor[:train_num], output_examples_values = l_out_nor[:train_num], hiding_nodes_number = hiding_number, lam = 0.01)
    error_train = bp_2.train_random(mu_out, sigma_out, time = 50, print_state = False)
    bp_2.test(mu_out, sigma_out, l_in_nor[train_num:], l_out_nor[train_num:], print_state = False)

    show_error(error_train, iteration = 50, unit = 1)
    '''




if __name__=="__main__":
    hiding_number = 8
    train_num = 120
    iteration = 100
    unit = 5
    #main()
    demo()

