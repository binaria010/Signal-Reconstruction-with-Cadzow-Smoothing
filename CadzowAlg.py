#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 19:17:42 2020

@author: Juliana
"""


import numpy as np
from numpy import *
from numpy.linalg import inv
import scipy.linalg
from scipy.linalg import *
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import random
np.random.seed(0)


def avrg_Hankel(A) :
    """
    this function returns a Hankel matrix formed by averaging across the antidiagonals of A. This is part
    of the denoising process in the Cadzow Algorithm
    inputs: A --> 2d numpy array
    """
    dim = A.shape
    flip_A = np.fliplr(A) # flip A to get its anti diagonals c[0]= A[0,0]
    col = zeros(dim[0],dtype = complex)
    row = zeros(dim[1], dtype = complex) # col and row are the vectors of the averages to create the new Hankel matrix
    for k in range(dim[0]):
        col[k] = mean(diag(flip_A, dim[0]-1-k))
    # create r    
    for k in range(dim[1]):
        row[k] = mean(diag(flip_A,-k))
    
    H = hankel(col, row)
    #print(col,row)
    return H


def Cadzow(Y_tilde, kmax, d, sigma, r = 0) :
    """
    this function denoises the meassurements matrix Ytilde using the Cadzow algorithm.
    Inputs : Y_tilde --> Jx(L+1) array: noisy meassurements for all the snapshots. 
             kmax --> int: indicating max number of iterations to denoise
    """
    J = Y_tilde.shape[0]
    L = Y_tilde.shape[1] - 1

    
    Y_denoi = zeros(Y_tilde.shape, dtype = complex)
    for j in range(J) :
                
        # Form Hankel matrix X = H(j)
        col = np.array(Y_tilde[j,0: L//2 + 1], dtype = complex)
        row = np.array(Y_tilde[j, L//2 : L+1], dtype = complex)
        X = np.array(hankel(col, row), dtype = complex)
        
        #denoising:
        for k in range(kmax):
            U, Sigma, Vh = linalg.svd(X, full_matrices = True) # non econ!
                        
            #truncating the svd to chose rank r
            if not (r == 0) :
                S = Sigma[0:r]
                Ur = U[:,0:r]
                Vh_r = Vh[0:r,:]
                X = Ur @ diag(S) @ Vh_r
            else :
                #truncatig using the threshold tol= 4/sqrt(3)sqrt(L//2+1)sigma
                tol = (4/np.sqrt(3))*np.sqrt(col.size)*sigma
                Sigma[Sigma < tol] = 0
                X = U @ diag(Sigma) @ Vh

            # the new Hankel matrix obtained by averaging X across the anti diagonals
            Hnew = avrg_Hankel(X)
            X = Hnew
            
        # update row j of Y_tilde have in mind that col_new[-1] = row_new[0]
        Y_denoi[j,0:L//2 + 1] =  np.array(Hnew[:,0], dtype = complex)
        Y_denoi[j, L//2+1: L+1] = np.array(Hnew[-1,1:], dtype = complex)
    
    
    return Y_denoi
