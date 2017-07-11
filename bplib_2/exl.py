#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#读取execel使用(支持07)
from openpyxl import Workbook
#写入excel使用(支持07)
from openpyxl import load_workbook
import tkinter.filedialog as filedialog
from tkinter import *
import os

def readexcel(path):
    wb=load_workbook(path)
    #print(wb)
    #print(wb.get_sheet_names())
    ws = wb['工作表1']
    l = []
    
    for n in ws.values:
        l.append(list(n))
    
    return l

def callback(entry):
    entry.delete(0,END) #清空entry里面的内容
    #listbox_filename.delete(0,END)
    #调用filedialog模块的askdirectory()函数去打开文件夹
    filepath = filedialog.askopenfilename()
    if filepath:
        entry.insert(0,filepath) #将选择好的路径加入到entry里面
    print (filepath)
    return filepath
#getdir(filepath)

def getdir(filepath=os.getcwd()):
    """
        用于获取目录下的文件列表
    """
    cf = os.listdir(filepath)
    for i in cf:
        listbox_filename.insert(END,i)

if __name__ == "__main__":
    root = Tk()
    root.title("测试版本")
    root.geometry("600x400")
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    
    entry1 = Entry(root, width=60)
    entry1.grid(sticky=W+N, row=0, column=0, columnspan=4, padx=5, pady=5)
    
    button1 = Button(root,text="选择文件",command = lambda: callback(entry1))
    button1.grid(sticky=W+N, row=1, column=0, padx=5, pady=5)
    
    entry2 = Entry(root, width=60)
    entry2.grid(sticky=W+N, row=2, column=0, columnspan=4, padx=5, pady=5)
    
    button2 = Button(root,text="选择文件",command = lambda: callback(entry2))
    button2.grid(sticky=W+N, row=3, column=0, padx=5, pady=5)
    #创建loistbox用来显示所有文件名
        
    root.mainloop()
    
    '''
    readpath = "./data2/octane.xlsx"
    a = readexcel(readpath)
    print(a)
    '''
