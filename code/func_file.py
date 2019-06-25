#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:35:07 2019

@author: kj22643
"""
from numpy import linalg as LA
import numpy as np
import pandas as pd
# This is the equivalent of your find_eigendigits function in Matlab
def find_eigenvectors_fxn(A):

# It starts by finding the mean vector
    m = A.mean(axis =1)
    return m
    

    # Next, make our small covariance matrix (ntrain x ntrain)
    # subtract the mean from each column of the  the training matrix (ngenes x ntrain)
    X = A.sub(m, axis=0) # we subtract from each column
    
    # convert the subtracted gene expression matrix to a matrix 
    # Xmat should be ngenes x ntrain
    Xmat= X.as_matrix()
    # transpose it 
    # this should be ntrain x ngenes
    XmatT = Xmat.T
    # since we have less cells than genes, we can use the trick to make a smaller cov matrix
    # which is the (ntrain x ngenes * ngenes x ntrain = ntrain x ntrain square small covariance matrix)
    smallcov = np.matmul(XmatT,Xmat) # Now you have your ntrain x ntrain small covariance matrix

    # should get a vector of ntrain lambdas, and a square matrix ntrain x ntrain of the eigenvectors
    lambdas, Vsmall= LA.eig(smallcov)
    ind_arr = np.argsort(-abs(lambdas))

    # Now your ind_arr should start with the highest eigenvalue and work its way down
    # print the indices to check that the first is the highest lambda and the last is the lowest
 
    ordered_lambdas = lambdas[ind_arr]
    return ordered_lambdas

    # Reorder your lambdas and eigenvectors (Vsmall)
    Vnew = Vsmall[ind_arr]
    # Now apply to the big system
    # Since XX'x=mux
    # Let x = Xv
    # XX'Xv = mu Xv
    # XX' = big cov matrix
    # X'X = small cov matrix

    # These are the eigenvectors of the big covariance matrix 
    # ngenes x ntrain x ntrain x ntrain gives ngenes x ntrain
    Vbig = np.matmul(Xmat, Vnew)
    # These vectors of length ntrain are the eigenvectors, in order of importance (eigenvalue)
    # Vbig is now an ngenes by ntrain matrix of eigenvectors (each column =1 eigenvector)

    # Renormalize: Now that you have your ngenes x ntraining cells matrix of 
    #eigenvectors, make sure the big covariance matrix is normalized
    norms = LA.norm(Vbig, axis = 0)
    Vnorm = Vbig/norms # divides each column by i
    return Vnorm