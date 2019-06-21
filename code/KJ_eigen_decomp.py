#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:27:36 2019

@author: kj22643
"""


# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:29:10 2019

@author: kj22643
"""

%reset

import numpy as np
import pandas as pd
import os
import scanpy as sc
import seaborn as sns
from plotnine import *
path = '/Users/kj22643/Documents/Documents/231_Classifier_Project/data'
#path = '/stor/scratch/Brock/231_10X_data/'
os.chdir(path)
sc.settings.figdir = 'KJ_plots'
sc.set_figure_params(dpi_save=300)
sc.settings.verbosity = 3
#%% Load in pre and post data
adata_pre = sc.read('KJ_adatapre.h5ad')
adata_post = sc.read('KJ_adatapost.h5ad')
# Make ann data objects into pandas data frames
dfpre = pd.concat([adata_pre.obs['survivor'], adata_pre.obs['sample'],pd.DataFrame(adata_pre.raw.X,index=adata_pre.obs.index,
                            columns=adata_pre.var_names),],axis=1) 
dfpost= pd.concat([adata_post.obs['sample'],pd.DataFrame(adata_post.raw.X,index=adata_post.obs.index,
                            columns=adata_post.var_names),],axis=1) 
#%% Try making a UMAP of the first sample only

sc.pl.umap(adata_pre,color=['survivor'],wspace=0.3,
           save='pre_treat_res_sens.png')

sc.pl.umap(adata_post,color=['sample'],wspace=0.3,
           save='post_treat_umap.png')

#%% See if we cam plot the first two PCs
sc.pl.pca(adata_pre, color=['survivor'],wspace=0.3,
          save= 'pre_treat_pca.png')
#%%
print(dfpre) # 22192 columns corresponding to 22191 genes
#%% Make series that label the pre-treatment cells as res/sens and label the 
# label the post treatment cells by their sample
labelsdfpre = dfpre['survivor']
print(labelsdfpre)
samplabspost = dfpost['sample']
print(samplabspost)
#%% Make matrices (data frames) of just the cell-gene matrix for the pre treatment and 
# post treatment samples
genematpre1 = dfpre.loc[:, dfpre.columns !='survivor']
genematpre= genematpre1.loc[:, genematpre1.columns !='sample']

genematpost = dfpost.loc[:, dfpost.columns !='sample']
print(genematpre)
# Now genematpre and genemat post are your ncells rows x ngenes columns gene 
# expression matrices.
#%% Now try to emulate your matlab code... 
# Start with just your pre-treatment time point
# In Matlab we have an x by k where x would be all the genes and k are the indivdual 
# cells (so each column is a cell and each row is a gene)

# let's see if we can make that in python and call it Adf
npost =genematpost.shape[0]
npre = genematpre.shape[0] # this gets the number of rows in the df (number of cells)
ntrain = 8*round(ncells/10)  # start by setting the number of training cells to 1/10th 
ntest = npre-ntrain
#%% Make your full data frames (including the testing and training data set from the pre-treatment time point)
# Call these Adf
AdfT= genematpre
Adf = AdfT.T
print(Adf)
ApoT= genematpost
Apost= ApoT.T
#%% Make susbets of the full data frame for training and testing
# Training data set (first 1:ntrain cells)
AtrT= AdfT.iloc[0:ntrain]
Atr = AtrT.T
print(Atr)
# Testing data set (ntrain to end cells)
AtestT= AdfT.iloc[ntrain:]
Atest = AtestT.T
# now we have each cell as a column and each row as a gene.
# We are going to use the training subset to find the eigenvectors and then 
# get the coordinates of each training cell in eigenspace
#%% First we want to find the mean gene expression level vector 
m = Atr.mean(axis =1) # axis = 1 means we find the mean along each row
print(m)

# Next, make our small covariance matrix (ntrain x ntrain)
# subtract the mean from each column of the  the training matrix (ngenes x ntrain)
X = Atr.sub(m, axis=0) # we subtract from each column
print(X)
# convert the subtracted gene expression matrix to a matrix 
# Xmat should be ngenes x ntrain
Xmat= X.as_matrix()
# transpose it 
# this should be ntrain x ngenes
XmatT = Xmat.T
# since we have less cells than genes, we can use the trick to make a smaller cov matrix
# which is the (ntrain x ngenes * ngenes x ntrain = ntrain x ntrain square small covariance matrix)
smallcov = np.matmul(XmatT,Xmat) # Now you have your ntrain x ntrain small covariance matrix

#%%Find the eigenvectors of the small covariance matrix



# in MATLAB:
# [Vsmall,D] =eig(smallcov)
# lamdas = diag(D)
# [orderedvals, ind] = sort(lambdas, 'descend');
# Vnew = Vsmall(:,ind);
#Vbig = A0*Vnew;
# Want to normalize each eigenvector
#k = length(lambdas);
#for j = 1:k
    ## find the 2-norm of each column
    #norms(j) = norm(Vbig(:,j));
    
    #V(:,j) = Vbig(:,j)./norms(j);
#end
    
#%% Find eigenvectors and sort them in order of eigenvalues   
from numpy import linalg as LA
# should get a vector of ntrain lambdas, and a square matrix ntrain x ntrain of the eigenvectors
lambdas, Vsmall= LA.eig(smallcov)
# returns the eigenvalues (lambdas) and eigenvectors that are normalized (Euclidean norms are 1)
print(lambdas)

#%% Want to sort the lambdas in descending order
# arg sort naturally sorts in ascending order, so make it negative
ind_arr = np.argsort(-abs(lambdas))

#%% Now your ind_arr should start with the highest eigenvalue and work its way down
# print the indices to check that the first is the highest lambda and the last is the lowest
print(ind_arr) 
ordered_lambdas = lambdas[ind_arr]
print(ordered_lambdas)
#%% Reorder your lambdas and eigenvectors (Vsmall)
Vnew = Vsmall[ind_arr]
#%% Now apply to the big system
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

#%% Renormalize: Now that you have your ngenes x ntraining cells matrix of 
#eigenvectors, make sure the big covariance matrix is normalized
norms = LA.norm(Vbig, axis = 0)
Vnorm = Vbig/norms # divides each column by its norm
testnorms = LA.norm(Vnorm, axis = 0) # want to ensure these are all essentially 1
# Vnorm is your normalized, ordered, eigenvectors. They are ordered by the value of the 
# ordered eigenvalue 
#%% Now you have the Eigenspace, defined by Vnorm, which contains normalized eigenvectors of
# the length of the number of genes, in order of importance.

# Now we need to do the projection of the individual cells (in A) onto the eigenvector space

#Mcol = reshape(M, [x,1]);
#b = (Mcol-m)'*V; %(1 by x) *(x*k) Gives a vector of length k

# Project onto eigenspace
#recon = V*b'; % (x by k) * (k *1) Gives x by 1

# Declare the number of eigenvectors we want to use from the Vnorm matrix 
neigs = 100
#eventually we will loop through the number of eigenvectors and assess the accuracy as afunction of neigs
trainmat = np.zeros((neigs, ntrain))
trainvec= np.zeros((neigs,1))

# First make your Omat which contains the coordinates (the columns of each training image
# and has the corresponding assignment (sens or res) ))
for i in range(ntrain):
    Mcol = Xmat[:,i] # start right after the last training column in Amat 
    McolT= Mcol.T # to get the first testing cell as a row
    # 1xngene x ngene x ntrain = 1 x ntrain
    b=np.matmul(McolT,Vnorm)

    trainvec= b[:neigs]
    trainmat[:,i]= trainvec
#%% Should generate a neigs x ntrain matrix where each column is the coordinates
    # of the cell in eigenspace
print(trainmat)

print(labelsdfpre[:ntrain])
trainlabels = labelsdfpre[:ntrain]
testlabels = labelsdfpre[ntrain:]

#%% Project the testing cells into eigenspace
ntest = ncells-ntrain
Xte = Atest.sub(m, axis=0) # we subtract from each column
Xtest = Xte.as_matrix()
testmat = np.zeros((neigs, ntest))
testvec = np.zeros((neigs,1))
for i in range(ntest):
    Mcol = Xtest[:,i]
    McolT = Mcol.T
    b = np.matmul(McolT, Vnorm)
    testvec = b[:neigs]
    testmat[:,i] = testvec

#%% Compare the coordinates of the testing set with the coordinates of the training set cells
# Now you have a test mat which has class labels corresponding to each column
# For each column of your test mat, you want to find the k training mat columns it is closest to
# make a matrix that stores the ordered indices with the top being the lowest and the bottom the highest
# Euclidean distance
ordered_inds = np.zeros((ntrain, ntest))
dist = np.zeros((ntrain,ntest))
#for i in range(ntest):
for i in range(ntest):
    testvec = testmat[:,i]
    #for j in range(ntrain):
    for j in range(ntrain):
        trainvec = trainmat[:,j]
        # testvec of length neigs and train vec of length neigs
        # find Euclidean distance between the two vectors
        dist_j = np.linalg.norm(testvec-trainvec)
        # fill in your distance vector
        dist[j, i]=dist_j

#%% Now you have a distance matrix where each column is a testing cell
# for each column, we want to output the indices of the training vector distance
# in order from least to greatest
lab_est = [None]*ntest
for i in range(ntest):
    distcol = dist[:,i]
    ind = np.argsort(distcol)
    ordered_inds[:,i] = ind

    
#%% Use the ordered ind to make sense/res matrix
    # Using k=1 nearest neighbors classifier. Need to figure out how to extend this 

for i in range(ntest):
    index = ordered_inds[0,i]
    lab_est[i] = trainlabels[int(index)]
    print(trainlabels[int(index)])
#%%  Make a data frame with your actual and predicted classes
 df = pd.DataFrame({'actual class':testlabels,'predicted class':lab_est})

cnf_matrix = pd.crosstab(df['predicted class'],df['actual class'])

hm = sns.heatmap(cnf_matrix,annot=True,fmt='d',robust=True,
            linewidths=0.1,linecolor='black')   
#%% Calculate some metrics of accuracy from the confusion matrix
TPR=cnf_matrix.iloc[0,0]/ (sum(cnf_matrix.iloc[:,0]))
print(TPR)
TNR = cnf_matrix.iloc[1,1]/(sum(cnf_matrix.iloc[:,1]))
print(TNR)
PPV = cnf_matrix.iloc[0,0]/sum(cnf_matrix.iloc[0,:])
print(PPV)
NPV = cnf_matrix.iloc[1,1]/sum(cnf_matrix.iloc[1,:])
print(NPV)
Acc = (cnf_matrix.iloc[0,0]+ cnf_matrix.iloc[1,1,])/ntest
print(Acc)
prevalance = sum(cnf_matrix.iloc[:,0])/ntest
print(prevalance)
#%% Vary the number of eigenvectors and then compare the accuracy metrics
neigvec = [10, 50, 75, 100, 150, 200, 250,]
TPRi=np.zeros((len(neigvec),1))
TNRi=np.zeros((len(neigvec),1))
PPVi=np.zeros((len(neigvec),1)) 
NPVi=np.zeros((len(neigvec),1)) 
Acci=np.zeros((len(neigvec),1))


#%%
for k in range(len(neigvec)):
    neigi=neigvec[k] # set the number of eigenvectors
    
    trainmat = np.zeros((neigi, ntrain))
   
    trainvec= np.zeros((neigi,1))
    
# First make your Omat which contains the coordinates (the columns of each training image
# and has the corresponding assignment (sens or res) ))
    for i in range(ntrain):
        Mcol = Xmat[:,i] # start right after the last training column in Amat 
        McolT= Mcol.T # to get the first testing cell as a row
        # 1xngene x ngene x ntrain = 1 x ntrain
        b=np.matmul(McolT,Vnorm)

        trainvec= b[:neigi]
        trainmat[:,i]= trainvec
        # We want to save all of these reduced coordinate spaces
        #eigenmatrices.append(trainmat)
    #Project the testing cells into eigenspace
    testmat = np.zeros((neigi, ntest))
    testvec = np.zeros((neigi,1))
    for i in range(ntest):
        Mcol = Xtest[:,i]
        McolT = Mcol.T
        b = np.matmul(McolT, Vnorm)
        testvec = b[:neigi]
        testmat[:,i] = testvec
# Compare the coordinates of the testing set with the coordinates of the training set cells


    dist = np.zeros((ntrain,ntest))
    #for i in range(ntest):
    for i in range(ntest):
        testvec = testmat[:,i]
        #for j in range(ntrain):
        for j in range(ntrain):
            trainvec = trainmat[:,j]
        # testvec of length neigs and train vec of length neigs
        # find Euclidean distance between the two vectors
            dist_j = np.linalg.norm(testvec-trainvec)
        # fill in your distance vector
            dist[j, i]=dist_j
    # Now you have a distance matrix where each column is a testing cell
    # for each column, we want to output the indices of the training vector distance
    # in order from least to greatest
    lab_est = [None]*ntest
    for i in range(ntest):
        distcol = dist[:,i]
        ind = np.argsort(distcol)
        ordered_inds[:,i] = ind
    # Use the ordered ind to make sense/res matrix
    # Using k=1 nearest neighbors classifier. Need to figure out how to extend this 

    for i in range(ntest):
        index = ordered_inds[0,i]
        lab_est[i] = trainlabels[int(index)]

    dfi = pd.DataFrame({'actual class':testlabels,'predicted class':lab_est})

    cnf_matrixi = pd.crosstab(dfi['predicted class'],dfi['actual class'])

    hm = sns.heatmap(cnf_matrixi,annot=True,fmt='d',robust=True,
            linewidths=0.1,linecolor='black')   
# Calculate some metrics of accuracy from the confusion matrix
    TPRi[k] = cnf_matrixi.iloc[0,0]/ (sum(cnf_matrixi.iloc[:,0]))

    TNRi[k] = cnf_matrixi.iloc[1,1]/(sum(cnf_matrixi.iloc[:,1]))

    PPVi[k] = cnf_matrixi.iloc[0,0]/sum(cnf_matrixi.iloc[0,:])

    NPVi[k] = cnf_matrixi.iloc[1,1]/sum(cnf_matrixi.iloc[1,:])

    Acci[k] = (cnf_matrixi.iloc[0,0]+ cnf_matrixi.iloc[1,1,])/ntest

#%% 
print(TPRi)
print(cnf_matrixi)
print(k)
print(cnf_matrixi.iloc[0,0]/ (sum(cnf_matrixi.iloc[:,0])))

#%% Plot the accuracy metrics versus number of eigenvectors
import matplotlib.pyplot as plt


plt.plot(neigvec, TPRi, 'b-', label='TPR')
plt.plot(neigvec, TNRi, 'g-', label='TNR')
plt.plot(neigvec, PPVi, 'r-', label='PPV')
plt.plot(neigvec, NPVi, 'y-', label='NPV')
plt.plot(neigvec, Acci, 'k-', label='Accuracy')
plt.xlabel('Number of eigenvectors')
plt.ylabel('Metrics of Accuracy')
plt.legend()
loc= 'lower right'


#%% BIG GAP FOR Applying the classifier to the post treatment samples!
# Here we are doing it to just the 107 Aziz sample (one of the very late post treatment samples). 
# We will combine this with the other sample, and we should also do the 30 hour time point.
    
    
    
    
    
    
 #%% Project the post treatment cells into eigenspace
dfpost107=dfpost[dfpost['sample'].str.contains("Aziz")]
mpost = Apost.mean(axis =1)
print(npost)
#%%
neigs=100
X107 = Apost.sub(mpost, axis=0) # we subtract from each column
Xpost = X107.as_matrix()
postmat = np.zeros((neigs, npost))
postvec = np.zeros((neigs,1))
for i in range(npost):
    Mcol = Xpost[:,i]
    McolT = Mcol.T
    b = np.matmul(McolT, Vnorm)
    postvec = b[:neigs]
    postmat[:,i] = postvec

#%% Compare the coordinates of the testing set with the coordinates of the training set cells
# Now you have a test mat which has class labels corresponding to each column
# For each column of your test mat, you want to find the k training mat columns it is closest to
# make a matrix that stores the ordered indices with the top being the lowest and the bottom the highest
# Euclidean distance
ordered_inds = np.zeros((ntrain, npost))
dist = np.zeros((ntrain,npost))
#for i in range(ntest):
for i in range(npost):
    postvec = postmat[:,i]
    #for j in range(ntrain):
    for j in range(ntrain):
        trainvec = trainmat[:,j]
        # testvec of length neigs and train vec of length neigs
        # find Euclidean distance between the two vectors
        dist_j = np.linalg.norm(postvec-trainvec)
        # fill in your distance vector
        dist[j, i]=dist_j

#%% Now you have a distance matrix where each column is a testing cell
# for each column, we want to output the indices of the training vector distance
# in order from least to greatest
lab_est_post = [None]*npost
for i in range(npost):
    distcol = dist[:,i]
    ind = np.argsort(distcol)
    ordered_inds[:,i] = ind

    
#%% Use the ordered ind to make sense/res matrix
    # Using k=1 nearest neighbors classifier. Need to figure out how to extend this 

for i in range(npost):
    index = ordered_inds[0,i]
    lab_est_post[i] = trainlabels[int(index)]
#%% Now that you have lab_est post, quantify the proportion.
ct_sens=0
for i in range(npost):
    if lab_est_post[i]=='sens':
        ct_sens+=1

phi_est=ct_sens/npost
print(phi_est)
phi_est0=1-prevalance
print(phi_est0)
    
    
    
    