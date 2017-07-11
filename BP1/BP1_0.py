#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import math, random
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

def sigmoid(x):
    return 1/(1 + math.exp(-x))


# sigmoid(x) = 1/(1+exp(-x)), sigmoid'(x) = sigmoid(x)(1 - sigmoid(x))
# th(x) = (exp(x) - exp(-x)/(exp(x) + exp(-x)), th'(x) = 1 - th(x) * th(x)


def sigmoid_derivatives(a):
    return sigmoid(a) * (1 - sigmoid(a))

def Squared_error(output_examples_value, outputnode):
    e = 0
    o = 0
    ov = 0
    for value in output_examples_value:
        o += value * value
    for value in outputnode:
        ov += value * value
    o = math.sqrt(o)
    ov = math.sqrt(ov)
    e = (o - ov) * (o - ov)
    return e

class BP:
    _hide_num = 8
    _hiding_nodes = [(i+1)//8 for i in range(8)]
    # outputnode = (0, 0)
    # _input_w
    # _output_w
    # _lamda
    
    def __init__(self, lam, input, output):
        # 权值 w[7][4]
        self._input_w = [[random.random() for j in range(len(input))] for i in range(len(self._hiding_nodes) - 1)]
        self._output_w =  [[-random.random() for j in range(len(self._hiding_nodes))] for i in range(len(output))]
        self._lamda = lam
        self.outputnode = [0 for i in range(len(output))]
    
    # 激活函数为 f(x) = 1/(1+exp(-x)), f'(x) = f(x)(1 - f(x))
    def Forward_propagation(self, input_examples_value):
        # 输入层 到 隐藏层 加权后使用激活函数 sigmoid(x) = 1 / (1+exp(-x))
        for i in range(len(self._hiding_nodes)-1):
            self._hiding_nodes[i] = 0
            for j in range(len(input_examples_value)):
                self._hiding_nodes[i] += self._input_w[i][j] * input_examples_value[j]
            self._hiding_nodes[i] = sigmoid(self._hiding_nodes[i])
            
        # 隐藏层 到 输出层 加权后使用激活函数 f(x) = x
        for i in range(len(self.outputnode)):
            self.outputnode[i] = 0
            for j in range(len(self._hiding_nodes)):
                self.outputnode[i] += self._output_w[i][j] * self._hiding_nodes[j]
            # self.outputnode[i] = self.outputnode[i]
            
        return self.outputnode

    def Backward_propagation(self, input_examples_value, output_examples_value):
        delta_w = [[0 for j in range(len(self._hiding_nodes))] for i in range(len(self.outputnode))]
        # 隐藏层 到 输出层 误差导数, 使用 f(x) = x 作为激活函数，所以导数为 1
        # delta_w[i][j] = -lamda * x[i] * (y[j] - t[j])
        for i in range(len(self.outputnode)):
            for j in range(len(self._hiding_nodes)):
                delta_w[i][j] = self._hiding_nodes[j] * (self.outputnode[i] - output_examples_value[i]) # * self.outputnode[i] * (1 - self.outputnode[i])
                self._output_w[i][j] = self._output_w[i][j] - self._lamda * delta_w[i][j]

        delta_w = [[0 for j in range(len(input_examples_value))] for i in range(len(self._hiding_nodes)-1)]
        # 输入层 到 隐藏层 误差导数，使用sigmoid 作为激活函数，所以导数为 sigmoid (1- sigmoid)
        # delta_w[i][j] = -lamda * x[i] * (y[k] - t[k]) * y[k] * (1-y[k]) * _output_w[i][k] * mid_y[i] * (1-mid_y[i])
        for i in range(len(self._hiding_nodes)-1):
            for j in range(len(input_examples_value)):
                delta_w[i][j] = 0
                for k in range(len(self.outputnode)):
                    delta_w[i][j] += input_examples_value[j] * (self.outputnode[k] - output_examples_value[k]) * self.outputnode[k] * (1 - self.outputnode[k]) * self._output_w[k][i] * self._hiding_nodes[i] * (1 - self._hiding_nodes[i])
                self._input_w[i][j] = self._input_w[i][j] - self._lamda * delta_w[i][j]

def add(x, y):
    return [x[i] + y[i] for i in range(len(x))]


if __name__=="__main__":

    test = dict()
    examples_2 = dict()

    examples_1 = {(1, -1, 0, 0) : [1,]
            , (0, 1, 0, -1) : [-1,]
            , (1, 1, -1, 0) : [2,]
            , (-1, -1, -1, -1) : [-2,]}
    

    A = [(random.random(), random.random()) for y in range(0,20)]
    B = [(-random.random(), random.random()) for y in range(0,20)]
    C = [(-random.random(), -random.random()) for y in range(0,20)]
    D = [(random.random(), -random.random()) for y in range(0,20)]

    for i in A[:-5] + B[:-5]:
        examples_2[i] = [-1, ]
    for i in D[:-5] + C[:-5]:
        examples_2[i] = [1, ]

    for i in A[-5:] + B[-5:]:
        test[i] = [-1, ]
    for i in D[-5:] + C[-5:]:
        test[i] = [1, ]
            
    examples = examples_2
    
    bp = BP(0.3, list(examples.keys())[0], list(examples.values())[0])

    error_before = 0
    for key, value in examples.items():
        outputvalue_before = bp.Forward_propagation(key)
        print(key, value, outputvalue_before)
        error_before += Squared_error(key, outputvalue_before)
    print( error_before)

    for a in range(100):
        error_before = 0
        error_after = 0
        for key, value in examples.items():
            outputvalue_before = bp.Forward_propagation(key)
            error_before += Squared_error(key, outputvalue_before)
            bp.Backward_propagation(key, value)
            outputvalue_after = bp.Forward_propagation(key)
            error_after += Squared_error(key, outputvalue_after)
        
        if error_after - error_before > -0.001 and error_after - error_before < 0.001:
            break

        print(a+1, error_after - error_before, error_before, error_after)

    print("example input\texpected output\tactual output")
    for key, value in examples.items():
        print(key, value, bp.Forward_propagation(key))

    print("\ntest input\texpected output\tactual output")
    for key, value in test.items():
        print(key, value, bp.Forward_propagation(key))
    
    plt.figure(1)
    plt.plot([x[0] for x in examples if examples[x][0]>0.5],[x[1] for x in examples if examples[x][0]>0.5], 'ro', color = "red")
    plt.plot([x[0] for x in examples if examples[x][0]<0.5],[x[1] for x in examples if examples[x][0]<0.5], 'ro', color = "blue")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("x")
    plt.ylabel("y")

    plt.figure(2)
    plt.plot([x[0] for x in examples if bp.Forward_propagation(x)[0]>0],[x[1] for x in examples if bp.Forward_propagation(x)[0]>0], 'ro', color = "red")
    plt.plot([x[0] for x in examples if bp.Forward_propagation(x)[0]<0],[x[1] for x in examples if bp.Forward_propagation(x)[0]<0], 'ro', color = "blue")
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

