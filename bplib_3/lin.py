#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

# 一元线性回归，用于输出数据拟合
def linefit(x , y):
    N = len(x)
    sx,sy,sx2,sy2,sxy=0,0,0,0,0
    
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x*x)
    sy2 = np.sum(y*y)
    sxy = np.sum(x*y)
    
    k = (sxy - sx * sy / N)/(sx2 - sx * sx / N)
    b = (sy - k * sx)/N
    r = abs(sy*sx/N-sxy)/np.sqrt((sx2-sx*sx/N)*(sy2-sy*sy/N))

    return k,b,r

# 随机生成
def randlist(list_in, list_out, number):
    n = len(list_in)
    b_list = range(n)
    # 先随机生成用于训练的数据的索引号数组，然后剩下的是用于测试的数据
    blist_eg = random.sample(b_list, number)
    blist_test = list(filter(lambda x:x not in blist_eg , b_list))
    # 再根据
    l_in_eg = np.array([list_in[i] for i in blist_eg])
    l_in_test = np.array([list_in[i] for i in blist_test])
    l_out_eg = np.array([list_out[i] for i in blist_eg])
    l_out_test = np.array([list_out[i] for i in blist_test])
    return l_in_eg, l_in_test, l_out_eg, l_out_test

if __name__ == '__main__':
    X=np.array([ 1 ,2  ,3 ,4 ,5 ,6]).reshape(-1,1)
    Y=np.array([ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,7.51]).reshape(-1,1)
    #z1 = np.polyfit(X, Y, 1)  #一次多项式拟合，相当于线性拟合
    #p1 = np.poly1d(z1)
    a,b,r=linefit(X,Y)
    #print(z1)
    #print(p1)
    print("X=",X)
    print("Y=",Y)
    print("拟合结果: y = %10.5f x + %10.5f , r=%10.5f" % (a,b,r) )
