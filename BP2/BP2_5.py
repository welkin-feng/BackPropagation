#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from bplib_2 import exl, lin, show

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

func_list = ["sigmoid", "th", "linear"]

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

# normalized_type = 1, x = (x - mu) / sigma, 返回数据集中每一个属性的 mu 和 sigma
# normalized_type = 2, x = (x - min) / (max - min)
def Normalized(data_set, normalized_type = 1):
    m = len(data_set)
    n = len(data_set[0])
    if normalized_type == 1:
        mu = [np.average([data_set[i][j] for i in range(m)]) for j in range(n)]
        sigma = [np.std([data_set[i][j] for i in range(m)]) for j in range(n)]
    elif normalized_type == 2:
        mu = [np.min([data_set[i][j] for i in range(m)]) for j in range(n)]
        sigma = [np.max([data_set[i][j] for i in range(m)]) - np.min([data_set[i][j] for i in range(m)]) for j in range(n)]
    normal = [[(data_set[i][j] - mu[j]) / sigma[j] for j in range(n)]  for i in range(m)]
    return normal, mu, sigma

def De_Normalized(normal, mu, sigma):
    m = len(normal)
    n = len(normal[0])
    data_set = [[normal[i][j] * sigma[j] + mu[j] for j in range(n)] for i in range(m)]
    return data_set


class BP:
    
    # _input_examples_values
    # _output_examples_values
    # _hiding_nodes
    # _output_nodes
    # _input_w
    # _output_w
    # _lamda
    
    def __init__(self, input_examples_values, output_examples_values, hiding_nodes_number = 8, lam = 0.5, lam_M = 0.1, hiding_func = "sigmoid", ouput_func = "linear"):
        if input_examples_values != None and output_examples_values != None:
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
        self._lamda_momentum = lam_M
        if hiding_func in func_list:
            self.hiding_func = hiding_func
        else:
            self.hiding_func = "sigmoid"
        if ouput_func in func_list:
            self.ouput_func = ouput_func
        else:
            self.ouput_func = "linear"

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
                self._hiding_nodes[m][i] = act_function(self._hiding_nodes[m][i], self.hiding_func)
            # 隐藏层 到 输出层 加权后使用激活函数 f(x) = x
            # 第 i 个输出层节点
            for i in range(len(self._output_examples_values[0])):
                self._output_nodes[m][i] = 0
                # 第 j 个隐藏层节点
                for j in range(len(self._hiding_nodes[0])):
                    self._output_nodes[m][i] += self._output_w[i][j] * self._hiding_nodes[m][j]
                self._output_nodes[m][i] = act_function(self._output_nodes[m][i], self.ouput_func)

    def Back_Propagation_Batch(self):
        delta_w_out = [[0 for j in range(len(self._hiding_nodes[0]))] for i in range(len(self._output_examples_values[0]))]
        delta_w_in = [[0 for j in range(len(self._input_examples_values[0]))] for i in range(len(self._hiding_nodes[0])-1)]
        M_w_out = [[0 for j in range(len(self._hiding_nodes[0]))] for i in range(len(self._output_examples_values[0]))]
        M_w_in = [[0 for j in range(len(self._input_examples_values[0]))] for i in range(len(self._hiding_nodes[0])-1)]
        
        # 第 m 个训练样本
        for m in range(len(self._input_examples_values)):
            # 隐藏层 到 输出层 误差导数 delta_w[i][j] = -lamda * x[i] * (y[j] - t[j]) * y[j] * (1 - y[j])
            for i in range(len(self._output_examples_values[0])):
                for j in range(len(self._hiding_nodes[0])):
                    delta_w_out[i][j] += self._hiding_nodes[m][j] * (self._output_nodes[m][i] - self._output_examples_values[m][i]) * act_function_derivatives(self._output_nodes[m][i], z = self.ouput_func)
            # 输入层 到 隐藏层 误差导数
            for i in range(len(self._hiding_nodes[0])-1):
                for j in range(len(self._input_examples_values[0])):
                    for k in range(len(self._output_examples_values[0])):
                        delta_w_in[i][j] += self._input_examples_values[m][j] * (self._output_nodes[m][k] - self._output_examples_values[m][k]) * act_function_derivatives(self._output_nodes[m][k], z = self.ouput_func) * self._output_w[k][i] * act_function_derivatives(self._hiding_nodes[m][i], self.hiding_func)
    
        # 更新每一个权值
        gradient = 0
        for i in range(len(self._output_examples_values[0])):
            for j in range(len(self._hiding_nodes[0])):
                self._output_w[i][j] = self._output_w[i][j] - (self._lamda * delta_w_out[i][j] + self._lamda_momentum * M_w_out[i][j]) / (m + 1)
                M_w_out[i][j] = delta_w_out[i][j]
                gradient += M_w_out[i][j] * M_w_out[i][j]
        gradient = np.sqrt(gradient)
        
        for i in range(len(self._hiding_nodes[0])-1):
            for j in range(len(self._input_examples_values[0])):
                self._input_w[i][j] = self._input_w[i][j] - (self._lamda * delta_w_in[i][j] + self._lamda_momentum * M_w_in[i][j]) / (m + 1)
                M_w_in[i][j] = delta_w_in[i][j]
        return gradient


    def Back_Propagation_Random(self, m, w = 1000):
        delta_w_out = [[0 for j in range(len(self._hiding_nodes[0]))] for i in range(len(self._output_examples_values[0]))]
        delta_w_in = [[0 for j in range(len(self._input_examples_values[0]))] for i in range(len(self._hiding_nodes[0])-1)]
        M_w_out = [[0 for j in range(len(self._hiding_nodes[0]))] for i in range(len(self._output_examples_values[0]))]
        M_w_in = [[0 for j in range(len(self._input_examples_values[0]))] for i in range(len(self._hiding_nodes[0])-1)]
        gradient = 0
        # 第 m 个训练样本,
        # 隐藏层 到 输出层 误差导数 delta_w[i][j] = -lamda * x[i] * (y[j] - t[j]) * y[j] * (1 - y[j])
        for i in range(len(self._output_examples_values[0])):
            for j in range(len(self._hiding_nodes[0])):
                delta_w_out[i][j] = self._hiding_nodes[m][j] * (self._output_nodes[m][i] - self._output_examples_values[m][i]) * act_function_derivatives(self._output_nodes[m][i], z = self.ouput_func)
                self._output_w[i][j] = self._output_w[i][j] - (self._lamda * delta_w_out[i][j] + self._lamda_momentum * M_w_out[i][j]) / w
                M_w_out[i][j] = delta_w_out[i][j]
                gradient += M_w_out[i][j] * M_w_out[i][j]
        gradient = np.sqrt(gradient)
        # 计算 输入层 到 隐藏层 误差导数，并更新每一个权值
        for i in range(len(self._hiding_nodes[0])-1):
            for j in range(len(self._input_examples_values[0])):
                for k in range(len(self._output_examples_values[0])):
                    delta_w_in[i][j] = self._input_examples_values[m][j] * (self._output_nodes[m][k] - self._output_examples_values[m][k]) * act_function_derivatives(self._output_nodes[m][k], z = self.ouput_func) * self._output_w[k][i] * act_function_derivatives(self._hiding_nodes[m][i], self.hiding_func)
                    self._input_w[i][j] = self._input_w[i][j] - (self._lamda * delta_w_in[i][j] + self._lamda_momentum * M_w_in[i][j]) / w
                    M_w_in[i][j] = delta_w_in[i][j]

        return gradient
    
    def Mean_Squared_Error(self, mu_out, sigma_out, _output_eg_values = None):
        if _output_eg_values != None:
            _output_examples_values = _output_eg_values
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
        return e / (m + 1)

    def test(self, mu_out, sigma_out, input_test_values, output_test_values, print_state = False):
        if input_test_values != None and output_test_values != None:
            _input_examples_values = input_test_values
            _output_examples_values = output_test_values
            test_num = len(input_test_values)
        else:
            raise ValueError('没有输入 input_test_values, output_test_values 或 test_eg')
        
        self.Forward_propagation(_input_examples_values)
        error = self.Mean_Squared_Error(mu_out, sigma_out, _output_examples_values)
        
        if print_state:
            print("\n测试数据集 Error:", error)
            #print("test input\t actual output\t predicted output")
            #for i in range(test_num):
            #print(_input_examples_values[i], _output_examples_values[i], self._output_nodes[i])
        return error

    # 训练的同时将测试数据带入，验证训练效果
    def hyber_train_batch(self, mu_out, sigma_out, input_test_values, output_test_values, time = 500, print_state = False):
        test_num = len(input_test_values)
        self.Forward_propagation()
        error_train = []
        error_test = []
        gradient = []
        print("训练次数 , Error:")
        for i in range(time):
            gradient_train = self.Back_Propagation_Batch()
            gradient.append(gradient_train)
            
            error_after_test = self.test(mu_out, sigma_out, input_test_values, output_test_values)
            error_test.append(error_after_test)
            
            self.Forward_propagation()
            error_after = self.Mean_Squared_Error( mu_out, sigma_out)
            error_train.append(error_after)
            if (i + 1) % 50 == 0:
                print(i + 1, ":", error_after)
            if error_after < 10 ** (-6):
                print(i + 1, ":", error_after)
                break
        iteration = i + 1
                        
        train_outputs = self._output_nodes
        self.Forward_propagation(input_test_values)
        test_outputs = self._output_nodes[:test_num]
        error = self.Mean_Squared_Error(mu_out, sigma_out, output_test_values)
        print("\n测试数据集 Error:", error)

        return error_train, error_test, train_outputs, test_outputs, gradient, iteration

    # 训练的同时将测试数据带入，验证训练效果
    def hyber_train_random(self, mu_out, sigma_out, input_test_values, output_test_values, lam_w = None, time = 200, print_state = False):
        # 测试集数量
        test_num = len(input_test_values)
        self.Forward_propagation()
        error_train = []
        error_test = []
        gradient = []
        print("训练次数 , Error:")
        sam_num = len(self._input_examples_values)
        
        if lam_w == None:
            lam_w = sam_num
        bstate = False
        for i in range(time):
            for m in random.sample(range(sam_num), sam_num):
                gradient_train = self.Back_Propagation_Random(m, lam_w)
            
            gradient.append(gradient_train)
            
            error_after_test = self.test(mu_out, sigma_out, input_test_values, output_test_values)
            error_test.append(error_after_test)
            
            self.Forward_propagation()
            error_after = self.Mean_Squared_Error( mu_out, sigma_out)
            error_train.append(error_after)

            if (i + 1) % 50 == 0:
                print(i + 1, ":", error_after)
            if error_after < 10 ** (-6):
                print(i + 1, ":", error_after)
                break
            
        iteration = i + 1
        
        train_outputs = self._output_nodes
        self.Forward_propagation(input_test_values)
        test_outputs = self._output_nodes[:test_num]
        error = self.Mean_Squared_Error(mu_out, sigma_out, output_test_values)
        print("\n测试数据集 Error:", error)
        
        return error_train, error_test, train_outputs, test_outputs, gradient, iteration

def import_data():
    dir = [x for x in os.listdir('.') if os.path.isdir(x) or '.xls' in os.path.splitext(x)[1]]
    print(dir)
    filebox = "./" + input("输入导入的文件/文件所在文件夹")
    while os.path.isdir(filebox):
        print([x for x in os.listdir(filebox) if os.path.isdir(x) or '.xls' in os.path.splitext(x)[1]])
        filebox += "/"+input("输入导入的文件/文件所在文件夹")
    if os.path.isfile(filebox):
        return exl.readexcel(filebox)


# BP2.5
def main():
    examples_1_in = [[1, -1, 0, 0],
                     [0, 1, 0, -1],
                     [1, 1, -1, 0],
                     [-1, -1, -1, -1]]
    examples_1_out = [[1,],
                      [-1,],
                      [2,],
                      [-2,]]

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

    unit = int(input("输入步长(每 unit 次迭代显示一次误差):"))
    hiding_number = int(input("输入隐藏层节点数:"))
    iteration = int(input("输入迭代次数:"))
    func_type = input("输入训练方式, 1 : 批量梯度下降算法, 2 : 随机梯度下降算法")
    n_type = int(input("输入归一化规则，1 : 最大最小值法, 2 : 统计概率法"))
    lam_ = float(input("输入主学习率，推荐0.7: "))
    lam_M_ = float(input("输入动量学习率，推荐0.1: "))

    # 归一化输入数据集
    l_in_normal = Normalized(l_in, normalized_type = n_type)[0]
    # 归一化输出数据集，并返回输出样本集的 均值 和 标准差
    l_out_normal, mu_out, sigma_out = Normalized(l_out, normalized_type = n_type)
    # 随机从样本集中选取元素用于训练和测试
    l_in_eg, l_in_test, l_out_eg, l_out_test = lin.randlist(l_in_normal, l_out_normal, train_num)
    # 初始化BP神经网络
    bp = BP(l_in_eg, l_out_eg, hiding_nodes_number = hiding_number, lam = lam_, lam_M = 0.1, hiding_func = "sigmoid", ouput_func = "linear")

    # 训练并测试数据集， 返回训练集误差，测试集误差，和神经网络对训练和测试集的输出
    if func_type == "2":
        error_train, error_test, train_outputs_normal, test_outputs_normal, gradient, iteration = bp.hyber_train_random(mu_out, sigma_out, l_in_test, l_out_test, time = iteration)
    else:
        error_train, error_test, train_outputs_normal, test_outputs_normal, gradient, iteration = bp.hyber_train_batch(mu_out, sigma_out, l_in_test, l_out_test, time = iteration)

    # 显示训练误差曲线和测试误差曲线
    show.show_error(error_train, error_test, iteration, unit )

    show.show_gradient(gradient, iteration, unit)

    # 反归一化，将输出数据集的数据还原
    train_outputs = De_Normalized(train_outputs_normal, mu_out, sigma_out)
    test_outputs = De_Normalized(test_outputs_normal, mu_out, sigma_out)
    
    train_targets = De_Normalized(l_out_eg, mu_out, sigma_out)
    test_targets = De_Normalized(l_out_test, mu_out, sigma_out)
    
    all_targets = train_targets + test_targets
    all_outputs = train_outputs + test_outputs
    
    k_train, b_train, r_train = lin.linefit([x[0] for x in train_targets], [x[0] for x in train_outputs])
    k_test, b_test, r_test = lin.linefit([x[0] for x in test_targets], [x[0] for x in test_outputs])
    k_all, b_all, r_all = lin.linefit([x[0] for x in all_targets], [x[0] for x in all_outputs])
    
    show.show_regression(train_outputs, train_targets, k_train, b_train, r_train, "Train Regression")
    show.show_regression(test_outputs, test_targets, k_test, b_test, r_test, "Test Regression")
    show.show_regression(all_outputs, all_targets, k_all, b_all, r_all, "All Regression")

# BP2.5
def demo():
    l_in = exl.readexcel("/Users/Welkin/Documents/人工智能/BackPropagation/data4/chemicalInputs.xlsx")
    l_out = exl.readexcel("/Users/Welkin/Documents/人工智能/BackPropagation/data4/chemicalTargets.xlsx")
    print("训练集样本数：", len(l_in), len(l_out))
    
    hiding_number = 8
    train_num = 400
    iteration = 200
    unit = 1
    
    batch_state = False
    lam_batch = 0.7
    lam_M_batch = 0.1
    lam_random = 0.5
    lam_M_random = 0.1
    
    n_type = 1
    hiding_type = "th"
    output_type = "linear"
    
    
    
    # 归一化输入数据集
    l_in_normal = Normalized(l_in, normalized_type = n_type)[0]
    # 归一化输出数据集，并返回输出样本集的 均值 和 标准差
    l_out_normal, mu_out, sigma_out = Normalized(l_out, normalized_type = n_type)
   
    # 随机从样本集中选取元素用于训练和测试
    l_in_eg, l_in_test, l_out_eg, l_out_test = lin.randlist(l_in_normal, l_out_normal, train_num)
    
    
    if batch_state:
        # 初始化BP神经网络
        bp = BP(l_in_eg, l_out_eg, hiding_nodes_number = hiding_number, lam = lam_batch, lam_M = lam_M_batch, hiding_func = hiding_type, ouput_func = output_type)
        # 训练并测试数据集， 返回训练集误差，测试集误差，和神经网络对训练和测试集的输出
        error_train, error_test, train_outputs_normal, test_outputs_normal, gradient, iteration = bp.hyber_train_batch(mu_out, sigma_out, l_in_test, l_out_test, time = iteration)
    else:
        # 初始化BP神经网络
        bp = BP(l_in_eg, l_out_eg, hiding_nodes_number = hiding_number, lam = lam_random, lam_M = lam_M_random, hiding_func = hiding_type, ouput_func = output_type)
        # 训练并测试数据集， 返回训练集误差，测试集误差，和神经网络对训练和测试集的输出
        error_train, error_test, train_outputs_normal, test_outputs_normal, gradient, iteration = bp.hyber_train_random(mu_out, sigma_out, l_in_test, l_out_test, time = iteration)
    
    # 显示训练误差曲线和测试误差曲线
    show.show_error(error_train, error_test, iteration, unit )
    
    show.show_gradient(gradient, iteration, unit)
    
    # 反归一化，将输出数据集的数据还原
    train_outputs = De_Normalized(train_outputs_normal, mu_out, sigma_out)
    test_outputs = De_Normalized(test_outputs_normal, mu_out, sigma_out)
    
    train_targets = De_Normalized(l_out_eg, mu_out, sigma_out)
    test_targets = De_Normalized(l_out_test, mu_out, sigma_out)
    
    all_targets = train_targets + test_targets
    all_outputs = train_outputs + test_outputs
    
    k_train, b_train, r_train = lin.linefit([x[0] for x in train_targets], [x[0] for x in train_outputs])
    k_test, b_test, r_test = lin.linefit([x[0] for x in test_targets], [x[0] for x in test_outputs])
    k_all, b_all, r_all = lin.linefit([x[0] for x in all_targets], [x[0] for x in all_outputs])

    show.show_regression(train_outputs, train_targets, k_train, b_train, r_train, "Train Regression")
    show.show_regression(test_outputs, test_targets, k_test, b_test, r_test, "Test Regression")
    show.show_regression(all_outputs, all_targets, k_all, b_all, r_all, "All Regression")


if __name__=="__main__":
    # hiding_number = 8
    # train_num = 120
    # iteration = 100
    # unit = 5
    # train_outputs = De_Normalized(train_outputs, mu_out, sigma_out)
    # k_train, b_train, r_train = lin.linefit([x[0] for x in l_out[:train_num]], [x[0] for x in train_outputs])
    #main()
    demo()
    plt.show()
