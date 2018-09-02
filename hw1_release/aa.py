#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : aa.py
# @Author: Xiong
# @Date  : 2018/7/14
# @Desc  :

import numpy


'''实现卷积运算'''
def ImgConvolve(input,kernel):
    '''获取图片和卷积核的宽高'''
    WImg = input.shape[0]
    HImg = input.shape[1]
    Wkernel = kernel.shape[0]
    Hkernel = kernel.shape[1]
    AddW = (Wkernel-1)//2
    AddH = (Hkernel-1)//2

    '''在原图的宽高外侧分别扩充卷积核的一半'''
    ImgTemp = numpy.zeros([WImg + AddW*2,HImg + AddH*2])
    '''将原图拷贝到临时图片的中间'''
    ImgTemp[AddW:AddW+WImg,AddH:AddH+HImg] = input[:,:]
    '''初始化一张同样大小的图片作输出的图片'''
    output = numpy.zeros_like(a=ImgTemp)
    '''将扩充图和卷积核做卷积运算'''
    for i in range(AddW,AddW+WImg):
        for j in range(AddH,AddH+HImg):
            output[i][j] = int(numpy.sum(ImgTemp[i-AddW:i+AddW+1,j-AddW:j+AddW+1]*kernel))#计算平均值

    return output[AddW:AddW+WImg,AddH:AddH+HImg]

'''实现均值过滤'''
def myAverage(input,kernel):
    '''均值过滤的卷积核大小大于1，需要进行规格化'''
    return ImgConvolve(input,kernel)*(1.0/numpy.sum(kernel))


'''生成高斯矩阵'''
def Gaussian(sigma):#传入方差
    width = heigh = 2*sigma+1
    gaussianKernel = numpy.zeros([width,width])

    for x in range(-sigma,sigma+1):
        for y in range(-sigma,sigma+1):
            gaussianKernel[x+sigma][y+sigma] = numpy.exp(-0.5*(x**2+y**2)/(sigma**2));#高斯函数求值
    return  gaussianKernel


'''Sobel算子'''
def Sobel(img,style):
    ''' Sobel算子横向和纵向的卷积核'''
    Gx = numpy.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    Gy = numpy.array([[-1,-2,-1],
                      [ 0, 0, 0],
                      [ 1, 2, 1]])

    sobelX = ImgConvolve(img,Gx)
    sobelY = ImgConvolve(img,Gy)

    if(style == 0):
        return sobelX
    if(style == 1):
        return sobelY
    if(style == 2):
        return abs(sobelX)+abs(sobelY)

I = numpy.array(
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
)

print(Sobel(I,0))
print(Sobel(I,1))