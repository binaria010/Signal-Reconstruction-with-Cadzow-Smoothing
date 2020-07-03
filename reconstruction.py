# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

def Omega_sampler(x, Omega):
    """
    returns a vector y = [x(m_1),x(m_2),x(m_3),...,x(m_k)]' with m_i in Omega
    inputs: x in C^d.
            Omega --> list of integers, the sampling nodes. Omega < {0,1,2,..,d-1}
    """
    J = len(Omega)
    y = zeros(J)

    for j,m in enumerate(Omega) :
        y[j] = x[m]
        
    return y

def Obs_op (Omega,d) :
    """
    This function computes the Obervability operator for a given set Omega of sampling nodes.
    inputs: Omega --> List. The list of the sampling nodes.
            d --> int. Dimension of the state space
    Return: S --> len(Omega)xd array (fat matrix). This matrix is such that the observation is y = Sx
    """
    J = len(Omega)
    S = zeros((J,d))
    for k,j in enumerate(Omega) :
         S[k,j] = 1
    return S
    
    

def Samples(x_0, A, Omega, L):
    """
    Returns the matrix Y whose columns are the samples y_l= SmB^lx for l=0,1,2,..., L >= 2m-1 
    inputs: 
    x --> ndarray of dim (d,1)
    A--> dxd matrix or ndarray, the evolution operator.
    Omega--> Sampling set
    L --> positive integer. Number of time levels L>= 2m-1
    
    """
    J = len(Omega)
    x = x_0
    Y = zeros((J, L+1))
    for l in range(L+1):
        y = Omega_sampler(x, Omega)
        Y[:,l] = y
        x = A@x
        
    return Y

def NoisySamples(Y, sigma) :
    """
    Returns noisy samples from the matrix of samples Y
    """
    Y_tilde = zeros(Y.shape)
    for l in range(Y.shape[1]):
        Y_tilde[:,l] = Y[:,l] + np.random.normal(loc = 0. , scale = sigma, size = Y.shape[0])
        
    return Y_tilde



def all_States(x_0, A, L):
    """
    computes all states from t = 0 to t = L:
    """
    d = x_0.size
    X = zeros((d,L+1))
    x = x_0
    for l in range(L+1):
        X[:,l] = x
        x = A@x
    return X





def recover_signal(A, S, Y):
    """
    this function recovers the initial state of a system with evolution operator A and sampling matrix S 
    from the meassurements Y (Y =[y0 | y1|..|yL])
    """
    L = Y.shape[1] - 1
    b = Y[:,0]
    A_bold = S    
    Q, R = np.linalg.qr(A_bold, mode = 'reduced')
    b_tilde = Q.conj().transpose()@ b
    # first attemp to f
    f = np.linalg.pinv(R) @ b_tilde
    
    for i in range(1,L+1):
        b = Y[:,i]
        A_bold = A_bold @ A
        A_cal = vstack((R,A_bold))
        Q, R = np.linalg.qr(A_cal, mode = 'reduced')
        b_tilde = (Q.conj().transpose()) @ hstack((b_tilde,b))
        f = np.linalg.pinv(R) @ b_tilde
    
    return f
    
#### To reconstruct the spectrum of the operator

def DMD(X, X_prime, r, compute_Op = False):
    """
    assumes X, X_prime are nxm arrays with m<n (skinny). r<=m
    Returns: 
    """
    U,S,Vh = linalg.svd(X,full_matrices = False) # full_matrices= False: computes the econ svd
    V = Vh.conj().T
    Ur = U[:,0:r] # first r columns of U, V and S
    Vr = V[:,0:r]
    Sr = diag(S[0:r])
    Atilde = Ur.conj().T@X_prime@Vr@inv(Sr)
    Lambda, W = linalg.eig(Atilde) # computes eigenvals and eigenvects of Atilde
    Lambda = diag(Lambda)
    Phi = X_prime@(Vr@inv(Sr))@W  # matrix with the eigenvectors of A
    alpha1 = Sr@(Vr[0,:].conj().T)
    b = inv(W@Lambda)@alpha1
    
    if compute_Op == True:
        A = X_prime @ Vr @ inv(Sr) @ Ur.conj().T
        return A, Phi, Lambda, b
    
    return Phi, Lambda, b