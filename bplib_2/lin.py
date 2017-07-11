#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import random

def linefit(x , y):
    N = len(x)
    sx,sy,sxx,syy,sxy=0,0,0,0,0

    
    for i in range(N):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    
    a = (sxy - sx * sy / N)/(sxx - sx * sx /N)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))

    return a,b,r

def randlist(list_in, list_out, number):
    n = len(list_in)
    b_list = range(n)
    blist_eg = random.sample(b_list, number)
    blist_test = list(filter(lambda x:x not in blist_eg , b_list))
    l_in_eg = [list_in[i] for i in blist_eg]
    l_in_test = [list_in[i] for i in blist_test]
    l_out_eg = [list_out[i] for i in blist_eg]
    l_out_test = [list_out[i] for i in blist_test]
    return l_in_eg, l_in_test, l_out_eg, l_out_test

if __name__ == '__main__':
    X=[ 1 ,2  ,3 ,4 ,5 ,6]
    Y=[ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,7.51]
    z1 = np.polyfit(X, Y, 1)  #一次多项式拟合，相当于线性拟合
    p1 = np.poly1d(z1)
    a,b,r=linefit(X,Y)
    print(z1)
    print(p1)
    print("X=",X)
    print("Y=",Y)
    print("拟合结果: y = %10.5f x + %10.5f , r=%10.5f" % (a,b,r) )
