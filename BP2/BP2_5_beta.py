#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from bplib_3 import *

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
        return (1 - np.exp(-2 * x))/(1 + np.exp(-2 * x))
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
    if normalized_type == 1:
        mu = np.average(data_set, axis = 0)
        sigma = np.std(data_set, axis = 0)
    elif normalized_type == 2:
        mu = np.min(data_set, axis = 0)
        sigma = np.max(data_set, axis = 0) - np.min(data_set, axis = 0)
    normal = (data_set - mu) / sigma
    return normal, mu, sigma

def De_Normalized(normal, mu, sigma):
    data_set = normal * sigma + mu
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
        if len(input_examples_values) != 0 and len(output_examples_values) != 0:
            # _input_examples_nodes[m][n+1], 保存输入数据集中的所有数据的值
            # 包括 全部输入项 和 1个阈值项
            t = np.array([[1] for i in range(len(input_examples_values))])
            self._input_examples_values = np.concatenate((input_examples_values,t), axis = 1)
            # _output_examples_values[m][1], 保存输出数据集中的所有数据的值
            self._output_examples_values = output_examples_values
            # 训练样本数
            sample_num = len(input_examples_values)
        else:
            raise ValueError('没有输入 input_examples_values, output_examples_values 或 examples')
       
        # _hiding_nodes[m][8], 用于保存bp神经网络对于每个输入的隐藏层的值, 全部初始化为1
        self._hiding_nodes = np.ones( (sample_num, hiding_nodes_number) )
        # _output_nodes[m][1], 用于保存bp神经网络的输出值，全部初始化为0
        self._output_nodes = np.zeros( (sample_num, len(self._output_examples_values[0])) )
        
        # Xavier初始化方法，适合th激活函数
        input_num = len(self._input_examples_values[0])
        output_num = len(self._output_examples_values[0])
        a_ = np.sqrt(6/(input_num + hiding_nodes_number - 1))
        b_ = 1
        # 输入层 到 隐藏层 权值 w[7][n+1], w[7][n+1] * a[n+1][1] = h[7][1]，初始化成随机值
        self._input_w = np.random.rand(hiding_nodes_number - 1, len(self._input_examples_values[0]) ) * a_ * 2 - a_
        # 隐藏层 到 输出层 权值 w[1][8], w[1][8] * h[8][1] = o[1][1]，初始化成随机值
        self._output_w = np.random.rand(len(self._output_examples_values[0]), hiding_nodes_number )
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
        if _input_eg_values is not None:
            t = np.array([[1] for i in range(len(_input_eg_values))])
            _input_examples_values = np.concatenate((_input_eg_values,t), axis = 1)
        else:
            _input_examples_values = self._input_examples_values
        # 第 m 个训练样本
        for m in range(len(_input_examples_values)):
            # 输入层 到 隐藏层 加权后使用激活函数 sigmoid(x) = 1 / (1+exp(-x))
            # (7*n+1) x (n+1*1) = (7*1)
            
            self._hiding_nodes[m][:-1] = np.dot(self._input_w, _input_examples_values[m].reshape(-1,1)).reshape(1,-1)
            #
            #print(self._input_w, _input_examples_values[m], self._hiding_nodes[m])
            # 隐藏层 到 输出层 加权后使用激活函数 f(x) = x
            # (2*8) x (8*1) = (2*1)
            self._output_nodes[m] = np.dot(self._output_w, self._hiding_nodes[m].reshape(-1,1)).reshape(1,-1)
        
            
        self._hiding_nodes = act_function(self._hiding_nodes, self.hiding_func)
        self._output_nodes = act_function(self._output_nodes, self.ouput_func)

    def Back_Propagation_Batch(self):
        delta_w_out = np.zeros( (len(self._output_examples_values[0]), len(self._hiding_nodes[0])) )
        delta_w_in = np.zeros( (len(self._hiding_nodes[0])-1, len(self._input_examples_values[0])) )
        M_w_out = delta_w_out.copy()
        M_w_in = delta_w_in.copy()
        
        train_num = len(self._input_examples_values)
        # 第 m 个训练样本
        for m in range(train_num):
            # 隐藏层 到 输出层 误差导数 delta_w[i][j] = -lamda * x[i] * (y[j] - t[j]) * y[j] * (1 - y[j])
            # (2*1) x (1*8) = (2*8)
            delta_w_out += np.dot( ( (self._output_nodes[m] - self._output_examples_values[m]) * act_function_derivatives(self._output_nodes[m], z = self.ouput_func)).reshape(-1,1),  self._hiding_nodes[m].reshape(1,-1)) / train_num
            
            # 输入层 到 隐藏层 误差导数
            # (1*2) x (2*7) = (1*7), (1*7) * (1*7) = (1*7)
            yita = np.dot(((self._output_nodes[m] - self._output_examples_values[m]) * act_function_derivatives(self._output_nodes[m], z = self.ouput_func)).reshape(1,-1), self._output_w[:,-1]) * act_function_derivatives(self._hiding_nodes[m][:-1], self.hiding_func).reshape(1,-1) / train_num
            # (7*1) x (1*4) = (7*4)
            delta_w_in += np.dot( yita.reshape(-1,1), self._input_examples_values[m].reshape(1,-1))

        gradient = 0
        # 更新每一个权值
        self._output_w = self._output_w - ((1 - self._lamda_momentum) * self._lamda * delta_w_out + self._lamda_momentum * self._lamda * M_w_out)
        M_w_out = delta_w_out.copy()
        gradient = np.sqrt(np.sum(M_w_out ** 2))

        self._input_w = self._input_w - ((1 - self._lamda_momentum) * self._lamda * delta_w_in + self._lamda_momentum * self._lamda * M_w_in)
        M_w_in = delta_w_in.copy
        
        return gradient


    def Back_Propagation_Random(self, m, w = 1000):
        delta_w_out = np.zeros( (len(self._output_examples_values[0]), len(self._hiding_nodes[0])) )
        delta_w_in = np.zeros( (len(self._hiding_nodes[0])-1, len(self._input_examples_values[0])) )
        M_w_out = delta_w_out.copy()
        M_w_in = delta_w_in.copy()

        # 第 m 个训练样本
        # 隐藏层 到 输出层 误差导数 delta_w[i][j] = -lamda * x[i] * (y[j] - t[j]) * y[j] * (1 - y[j])
        # (2*1) x (1*8) = (2*8)
        delta_w_out = np.dot( ((self._output_nodes[m] - self._output_examples_values[m]) * act_function_derivatives(self._output_nodes[m], z = self.ouput_func)).reshape(-1,1),  self._hiding_nodes[m].reshape(1,-1))


        # 输入层 到 隐藏层 误差导数
        # (1*2) x (2*7) = (1*7), (1*7) * (7,) = (1*7)
        yita = np.dot(((self._output_nodes[m] - self._output_examples_values[m]) * act_function_derivatives(self._output_nodes[m], z = self.ouput_func)).reshape(1,-1), self._output_w) * act_function_derivatives(self._hiding_nodes[m], self.hiding_func)
        # (7*1) x (1*4) = (7*4)
        delta_w_in = np.dot( yita.reshape(-1,1), self._input_examples_values[m].reshape(1,-1))

        gradient = 0
        # 更新每一个权值
        self._output_w = self._output_w - (self._lamda * delta_w_out + self._lamda_momentum * M_w_out) / w
        M_w_out = delta_w_out.copy()
        gradient = np.sqrt(np.sum(M_w_out ** 2))

        self._input_w = self._input_w - (self._lamda * delta_w_in + self._lamda_momentum * M_w_in) / w
        M_w_in = delta_w_in.copy
        
        return gradient

    def Mean_Squared_Error(self, mu_out, sigma_out, _output_eg_values = None):
        if _output_eg_values is not None:
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
            o = np.sqrt(np.sum((_output_examples_values[m] * sigma_out + mu_out) ** 2))
            ov = np.sqrt(np.sum((_output_nodes[m] * sigma_out + mu_out) ** 2))
            e += (o - ov) * (o - ov)
        return e / (m + 1)

    def test(self, mu_out, sigma_out, input_test_values, output_test_values, print_state = False):
        if len(input_test_values) != 0 and len(output_test_values) != 0:
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
    #######
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
            error_after = self.Mean_Squared_Error(mu_out, sigma_out)
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
    examples_1_in = np.array([[1, -1, 0, 0],
                              [0, 1, 0, -1],
                              [1, 1, -1, 0],
                              [-1, -1, -1, -1]])
    examples_1_out = np.array([[1,],
                               [-1,],
                               [2,],
                               [-2,]])

    print("import 输入数据集")
    l_in = np.array(import_data())
    print("import 输出数据集")
    l_out = np.array(import_data())
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
            print("输入有误, 样本总数为 ", len(l_in))

    unit = int(input("输入步长(每 unit 次迭代显示一次误差):"))
    hiding_number = int(input("输入隐藏层节点数:"))
    iteration = int(input("输入迭代次数:"))
    func_type = input("输入训练方式, 1 : 批量梯度下降算法, 2 : 随机梯度下降算法")
    n_type = int(input("输入归一化规则，1 : 最大最小值法, 2 : 统计概率法"))

    # 归一化输入数据集
    l_in_normal = Normalized(l_in, normalized_type = n_type)[0]
    # 归一化输出数据集，并返回输出样本集的 均值 和 标准差
    l_out_normal, mu_out, sigma_out = Normalized(l_out, normalized_type = n_type)
    # 随机从样本集中选取元素用于训练和测试
    l_in_eg, l_in_test, l_out_eg, l_out_test = lin.randlist(l_in_normal, l_out_normal, train_num)
    # 初始化BP神经网络
    bp = BP(l_in_eg, l_out_eg, hiding_nodes_number = hiding_number, lam = 0.5, lam_M = 0.1, hiding_func = "sigmoid", ouput_func = "linear")

    # 训练并测试数据集， 返回训练集误差，测试集误差，和神经网络对训练和测试集的输出
    if func_type == "2":
        error_train, error_test, train_outputs_normal, test_outputs_normal, gradient, iteration = bp.hyber_train_random(mu_out, sigma_out, l_in_test, l_out_test, time = iteration)
    else:
        error_train, error_test, train_outputs_normal, test_outputs_normal, gradient, iteration = bp.hyber_train_batch(mu_out, sigma_out, l_in_test, l_out_test, time = iteration)

    # 显示训练误差曲线和测试误差曲线
    show.show_error(error_train, error_test, iteration, unit )

    show.show_gradient(gradient, iteration, unit)

    # 反归一化，将输出数据集的数据还原
    train_outputs = De_Normalized(train_outputs, mu_out, sigma_out)
    test_outputs = De_Normalized(test_outputs, mu_out, sigma_out)
    
    train_targets = De_Normalized(l_out_eg, mu_out, sigma_out)
    test_targets = De_Normalized(l_out_test, mu_out, sigma_out)

    all_targets = np.concatenate((train_targets, test_targets))
    all_outputs = np.concatenate((train_outputs, test_outputs))

    if len(train_targets[0]) == 1:
        k_train, b_train, r_train = lin.linefit(train_targets, train_outputs)
        k_test, b_test, r_test = lin.linefit(test_targets, test_outputs)
        k_all, b_all, r_all = lin.linefit(all_targets, all_outputs)
        
        show.show_regression(train_outputs, train_targets, k_train, b_train, r_train, "Train Regression")
        show.show_regression(test_outputs, test_targets, k_test, b_test, r_test, "Test Regression")
        show.show_regression(all_outputs, all_targets, k_all, b_all, r_all, "All Regression")

# BP2.5
def demo():
    l_in = exl.readexcel("/Users/Welkin/Documents/人工智能/BackPropagation/data4/chemicalInputs.xlsx")
    l_out = exl.readexcel("/Users/Welkin/Documents/人工智能/BackPropagation/data4/chemicalTargets.xlsx")
    l_in = np.array(l_in)
    l_out = np.array(l_out)
    print("训练集样本数：", len(l_in), len(l_out))
    
    hiding_number = 8
    train_num = 400
    iteration = 100
    unit = 1
    
    batch_state = True
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
    # (train_num * 1), (test_num * 1)
    train_outputs = De_Normalized(train_outputs_normal, mu_out, sigma_out)
    test_outputs = De_Normalized(test_outputs_normal, mu_out, sigma_out)
    
    train_targets = De_Normalized(l_out_eg, mu_out, sigma_out)
    test_targets = De_Normalized(l_out_test, mu_out, sigma_out)
    
    all_targets = np.concatenate((train_targets, test_targets))
    all_outputs = np.concatenate((train_outputs, test_outputs))
    
    if len(train_targets[0]) == 1:
        k_train, b_train, r_train = lin.linefit(train_targets, train_outputs)
        k_test, b_test, r_test = lin.linefit(test_targets, test_outputs)
        k_all, b_all, r_all = lin.linefit(all_targets, all_outputs)
        
        show.show_regression(all_outputs, all_targets, k_all, b_all, r_all, "All Regression")
        show.show_regression(test_outputs, test_targets, k_test, b_test, r_test, "Test Regression")
        show.show_regression(train_outputs, train_targets, k_train, b_train, r_train, "Train Regression")


if __name__=="__main__":
    # hiding_number = 8
    # train_num = 120
    # iteration = 100
    # unit = 5
    # train_outputs = De_Normalized(train_outputs, mu_out, sigma_out)
    # k_train, b_train, r_train = lin.linefit([x[0] for x in l_out[:train_num]], [x[0] for x in train_outputs])
    # main()
    demo()
    plt.show()
