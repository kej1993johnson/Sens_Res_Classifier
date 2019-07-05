#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:48:27 2019

@author: kj22643
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:22:02 2019

@author: kj22643
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:11:10 2019

@author: kj22643
"""

%reset

import numpy as np
import pandas as pd
import os
import scanpy as sc
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib as plt
import random
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import plotnine as gg
import scipy as sp
import scipy.stats as stats
import sklearn as sk
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
import sklearn.feature_selection as feature_selection
import sklearn.linear_model as linear_model
import sklearn.pipeline as pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
os.chdir('/Users/kj22643/Documents/Documents/231_Classifier_Project/code')
from func_file import find_mean_AUC
from func_file import find_mean_AUC_SVM



path = '/Users/kj22643/Documents/Documents/231_Classifier_Project/data'
#path = '/stor/scratch/Brock/231_10X_data/'
os.chdir(path)
sc.settings.figdir = 'KJ_plots'
sc.set_figure_params(dpi_save=300)
sc.settings.verbosity = 3

#%% Load in pre and post treatment 231 data
adata = sc.read('HD_tpo-fm_filtered_scran_cc-regressed_fcounts-4k_7.2.h5ad')
adata.obs.head()
# current samples:
#TP0
# FM1
# FM7
# We will change these to time points


#%% Assign survivor category in adata.obs
longTreatLins = adata.obs.loc[(adata.obs['sample'].isin(['FM1','FM7']))&(adata.obs.lineage!='nan'),'lineage'].unique().tolist()

adata.obs.loc[adata.obs.lineage.isin(longTreatLins)==False,'survivor'] = 'sens'
adata.obs.loc[adata.obs.lineage.isin(longTreatLins)==True,'survivor'] = 'res'

# %%try to rename the samples by time point
samps= adata.obs['sample'].unique()
timepoint = np.array(['t=0d', 't=29d'])

adata.obs.loc[adata.obs['sample']==samps[0], 'timepoint']='t=0d'
adata.obs.loc[adata.obs['sample']==samps[1], 'timepoint']='t=29d'
adata.obs.loc[adata.obs['sample']==samps[2], 'timepoint']='t=29d'


print(adata.obs['timepoint'].unique())



#%% Separately make dataframes for the pre-treatment, intermediate, and post treatment samples
# t=0 hr (pre-treatment), 2003 pre treatment cells, 21317 genes measured

# We want to keep the info about the lineage so we can potentially
# use it to make evenly divided testing and training data sets
adata_pre = adata[adata.obs['timepoint']=='t=0d', :]
dfpre = pd.concat([adata_pre.obs['survivor'], adata_pre.obs['lineage'],
               pd.DataFrame(adata_pre.raw.X,index=adata_pre.obs.index,
                            columns=adata_pre.var_names),],axis=1) 
# t = 8 weeks, 6972 post-treatment cells
adata_post = adata[adata.obs['timepoint']=='t=29d', :]
dfpost = pd.concat([adata_post.obs['lineage'],
                   pd.DataFrame(adata_post.raw.X, index=adata_post.obs.index, 
                                columns = adata_post.var_names),], axis=1)


#%% Use sklearn to do principle component analysis on the  entire pre-treatment sample
#X = dfpre.loc[:, dfpre.columns !='survivor', dfpre.columns !='lineage']
X = dfpre.drop(columns= ['survivor', 'lineage'])

y= pd.factorize(dfpre['survivor'])[0] 
ncells = len(y)

mu_pre = sum(y)/len(y)
# X is your cell gene matrix, y is your class labels
#%% Set up cross validation where your test set is not contained in your training set at all
# Split into train/test
kCV = 5
# use this function to ensure that the class balance is maintained for each of your test sets
skf = StratifiedKFold(n_splits=kCV, shuffle= True)
Atrain = {}
Atest = {}
ytest = {}
ytrain = {}
mu_true_test = {}
ntest = {}

folds_dict = {'trainmat':{}, 'trainlabel':{}, 'V':{}, 'lambdas':{}, }
for i in range(kCV):    
    for train_index, test_index in skf.split(X, y):
        Atrain[i] = X.iloc[train_index, :]
        Atest[i] = X.iloc[test_index, :]
        ytest[i]= y[test_index]
        ytrain[i]= y[train_index]
        mu_true_test[i] = sum(ytest[i])/len(ytest[i])
        ntest[i]=len(ytest[i])
         
# Save all of your stratified folds into a single dictionary. 
folds_dict['trainmat'] = Atrain
folds_dict['trainlabel']= ytrain
folds_dict['testmat'] = Atest
folds_dict['testlabel'] = ytest
folds_dict['prevtest'] = mu_true_test
folds_dict['ntest'] = ntest




n_classes = len(np.unique(y))
      


# %%Assign the optimal parameters (foudn from KJ_classify_sklearn.py) for building your prediction model 
# p(x|Sj) where Sj is your training set and x is any new cell (in your test set or in future cells)
n_neighbors = 15
n_components = 100
random_state = 0

Copt = 1000
basis = 'rbf'


knn = KNeighborsClassifier(n_neighbors=n_neighbors)

pca=PCA(copy=True, iterated_power='auto', n_components=n_components, random_state=0,
            svd_solver='auto', tol=0.0, whiten=False)

clf = svm.SVC(kernel=basis, C=Copt)

#%% Build a new model for each fold and apply it to the test set to generate mu_hat
# where mu_hat_j is the average value of the test set predictions (sens=0, res =1) from the training set model
y_PCA = {}
mu_PCA = {}
sigmasq_PCA = {}
#y_SVM = {}
#mu_SVM = {}
#sigmasq_SVM = {}
V_train = {}


for i in range(kCV):
    X_train = folds_dict['trainmat'][i]
    y_train = folds_dict['trainlabel'][i]
    X_test = folds_dict['testmat'][i]
    y_test = folds_dict['testlabel'][i]
    
    # PCA MODEL OUTPUTS FOR EACH FOLD
    pca.fit(X_train, y_train)
    V_train[i] = pca.fit_transform(X_train)
# Fit a nearest neighbor classifier on the model built on the training data set
    knn.fit(pca.transform(X_train), y_train)
    
# Use your knn and the PCA transform to predict the class of the test set cells
# Assume that knn.predict uses majority rule, potentially we will want to change this 
# Likely we want to classify something as resistant more often 
#(since our frac res is consistently undershooting the true resistant fraction)
# For now, we leave as the default
    y_PCA[i] = knn.predict(pca.transform(X_test))
# Compute the nearest neighbor accuracy on the embedded test set
    mu_PCA[i] = sum(y_PCA[i])/ folds_dict['ntest'][i]
    sigmasq_PCA[i] = mu_PCA[i]*(1-mu_PCA[i])/folds_dict['ntest'][i]
    # SVM MODEL OUTPUTS FOR EACH FOLD
    #clf.fit(X_train, y_train)
    #y_SVM[i]= clf.predict(X_test)
    #mu_SVM[i] = sum(y_SVM[i])/folds_dict['ntest'][i]
    #sigmasq_SVM[i] = mu_SVM[i]*(1-mu_SVM[i])/folds_dict['ntest'][i]
    
#%%  Put into folds_dict
folds_dict['V_train'] = V_train
folds_dict['y_PCA']= y_PCA
folds_dict['mu_PCA'] = mu_PCA
folds_dict['sigmasq_PCA'] = sigmasq_PCA
#folds_dict['y_SVM'] = y_SVM
#folds_dict['mu_SVM'] = mu_SVM
#folds_dict['sigmasq_SVM'] = sigmasq_SVM

   
    
#%%Compare the mu_SVM and mu_PCA test set estimates of the expectation to the known
    # prevalence of the test set
df = pd.DataFrame()
dfprevtest = pd.DataFrame(folds_dict['prevtest'], index=[0])
dfmu_PCA= pd.DataFrame(folds_dict['mu_PCA'], index = [0])
#dfmu_SVM = pd.DataFrame(folds_dict['mu_SVM'], index = [0])

npprevtest=np.array(dfprevtest)
npmu_PCA = np.array(dfmu_PCA)
#npmu_SVM = np.array(dfmu_SVM)
mu_pre = np.mean(npprevtest)
mu_hat_PCA = np.mean(npmu_PCA)
#mu_hat_SVM = np.mean(npmu_SVM)
ntest = folds_dict['ntest'][0]
print(mu_hat_PCA)
#print(mu_hat_SVM)
                 
sigmasq_PCA =(1-mu_hat_PCA)*mu_hat_PCA/ntest
#sigmasq_SVM = (1-mu_hat_SVM)*mu_hat_SVM/ntest
print(sigmasq_PCA)
#print(sigmasq_SVM)


#%% Next step, apply the models to the subsequent time points!

# Make your cell gene matrices
Xpost = dfpost.drop(columns =['lineage'])
#%%
# Already defined PCA and SVM hyperparameters above

# Start by builing model using all the data from the pre-treatment time point
pca.fit(X, y)
# Compute the eigenvector space 
V = pca.fit_transform(X)
# Fit a nearest neighbor classifier on the model built on the training data set
knn.fit(pca.transform(X), y)
    
y_pre = knn.predict(pca.transform(X))
mu_pre_PCA = sum(y_pre)/len(X)

# Now that you have the PC model, apply it to post treatment samples.
y_post = knn.predict(pca.transform(Xpost))
mu_post_PCA= sum(y_post)/len(Xpost)
sigmasq_post_PCA = mu_post_PCA*(1-mu_post_PCA)/len(Xpost)
#%% Print the PCA estimate outputs
print(mu_pre_PCA)
print(mu_post_PCA)
#%% Project the intermediate and post treatment samples into the principal component space
# This is mostly for visualiation purposes, but also to evaluate if the post-treatment 
# cells are more different in some principal components than others




PCspre=pca.transform(X)
PCspost = pca.transform(Xpost)

#%%
PC_df = pd.DataFrame(PCspre)
PC_df.reset_index(drop=True, inplace=True)
PC_df['classlabel'] = y
PC_df['time'] = 0



PCpostdf = pd.DataFrame(PCspost)
PCpostdf.reset_index(drop=True, inplace=True)
PCpostdf['classlabel'] = y_post
PCpostdf['time'] = 29


PCall = pd.concat([PC_df, PCpostdf], axis=0)


#%%
sns.set_style('white')
fig = plt.figure(figsize=(6,6))
ax=sns.scatterplot(PCall[0], PCall[1], hue = PCall['time'])
#%%
ax=sns.scatterplot(PC_df[0], PC_df[1])
ax = sns.scatterplot(PCpostdf[0], PCpostdf[1])

ax.set(xlabel ='PC1', ylabel ='PC2') 

#%% PC1 PC2 and PC3
fig = plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection='3d')

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[0])
Ys=np.asarray(PC_dfsens[1])
Zs=np.asarray(PC_dfsens[2])

Xr= np.asarray(PC_dfres[0])
Yr=np.asarray(PC_dfres[1])
Zr=np.asarray(PC_dfres[2])

Xpost = np.asarray(PCpostdf[0])
Ypost = np.asarray(PCpostdf[1])
Zpost = np.asarray(PCpostdf[2])

PCpostsens = PCpostdf[PCpostdf['classlabel']==0]
PCpostres = PCpostdf[PCpostdf['classlabel']==1]

Xposts= np.asarray(PCpostsens[0])
Yposts=np.asarray(PCpostsens[1])
Zposts=np.asarray(PCpostsens[2])

Xpostr= np.asarray(PCpostres[0])
Ypostr=np.asarray(PCpostres[1])
Zpostr=np.asarray(PCpostres[2])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0d res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0d sens')

#post_cells = ax.scatter(Xpost, Ypost, Zpost, c='g', marker = 'o', alpha = 0.3, label = 't=29 d')
post_sens_cells = ax.scatter(Xposts, Yposts, Zposts, c='k', marker = 'o', alpha = 0.9, label = 't=29d sens ')
post_res_cells = ax.scatter(Xpostr, Ypostr, Zpostr, c='c', marker = '^', alpha = 0.1, label = 't=29d res ')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(loc=2, prop={'size': 25})
#plt.title('Pre and post-treatment cells in PC space',fontsize= 20)

#ax.legend([sens_cells, res_cells], ['t=0 hr sens', 't=0 hr res'])

ax.azim = 100
ax.elev = -50
#%% PC2 PC3 and PC4
fig = plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection='3d')

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[1])
Ys=np.asarray(PC_dfsens[2])
Zs=np.asarray(PC_dfsens[3])

Xr= np.asarray(PC_dfres[1])
Yr=np.asarray(PC_dfres[2])
Zr=np.asarray(PC_dfres[3])


Xpost= np.asarray(PCpostdf[1])
Ypost=np.asarray(PCpostdf[2])
Zpost=np.asarray(PCpostdf[3])


Xposts= np.asarray(PCpostsens[1])
Yposts=np.asarray(PCpostsens[2])
Zposts=np.asarray(PCpostsens[3])

Xpostr= np.asarray(PCpostres[1])
Ypostr=np.asarray(PCpostres[2])
Zpostr=np.asarray(PCpostres[3])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0d res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0d sens')

#post_cells = ax.scatter(Xpost, Ypost, Zpost, c='g', marker = 'o', alpha = 0.3, label = 't=29 d')
post_sens_cells = ax.scatter(Xposts, Yposts, Zposts, c='k', marker = 'o', alpha = 0.9, label = 't=29d sens ')
post_res_cells = ax.scatter(Xpostr, Ypostr, Zpostr, c='c', marker = '^', alpha = 0.1, label = 't=29d res ')

ax.set_xlabel('PC2')
ax.set_ylabel('PC3')
ax.set_zlabel('PC4')
plt.legend(loc=2, prop={'size': 25})
#plt.title('Pre and post-treatment cells in PC space',fontsize= 20)

#ax.legend([sens_cells, res_cells], ['t=0 hr sens', 't=0 hr res'])

ax.azim = 100
ax.elev = 50

#%% PC1 PC3 and PC4
fig = plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection='3d')

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[0])
Ys=np.asarray(PC_dfsens[2])
Zs=np.asarray(PC_dfsens[3])

Xr= np.asarray(PC_dfres[0])
Yr=np.asarray(PC_dfres[2])
Zr=np.asarray(PC_dfres[3])




Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[2])
Zpost=np.asarray(PCpostdf[3])

Xposts= np.asarray(PCpostsens[0])
Yposts=np.asarray(PCpostsens[2])
Zposts=np.asarray(PCpostsens[3])

Xpostr= np.asarray(PCpostres[0])
Ypostr=np.asarray(PCpostres[2])
Zpostr=np.asarray(PCpostres[3])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0d res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0d sens')

#post_cells = ax.scatter(Xpost, Ypost, Zpost, c='g', marker = 'o', alpha = 0.3, label = 't=29 d')
post_sens_cells = ax.scatter(Xposts, Yposts, Zposts, c='k', marker = 'o', alpha = 0.9, label = 't=29d sens ')
post_res_cells = ax.scatter(Xpostr, Ypostr, Zpostr, c='c', marker = '^', alpha = 0.1, label = 't=29d res ')


ax.set_xlabel('PC1')
ax.set_ylabel('PC3')
ax.set_zlabel('PC4')
plt.legend(loc=2, prop={'size': 25})
#plt.title('Pre and post-treatment cells in PC space',fontsize= 20)

#ax.legend([sens_cells, res_cells], ['t=0 hr sens', 't=0 hr res'])

ax.azim = 100
ax.elev = -50

#%% PC1 PC2 and PC4
fig = plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection='3d')

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[0])
Ys=np.asarray(PC_dfsens[1])
Zs=np.asarray(PC_dfsens[3])

Xr= np.asarray(PC_dfres[0])
Yr=np.asarray(PC_dfres[1])
Zr=np.asarray(PC_dfres[3])




Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[1])
Zpost=np.asarray(PCpostdf[3])

Xposts= np.asarray(PCpostsens[0])
Yposts=np.asarray(PCpostsens[1])
Zposts=np.asarray(PCpostsens[3])

Xpostr= np.asarray(PCpostres[0])
Ypostr=np.asarray(PCpostres[1])
Zpostr=np.asarray(PCpostres[3])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0d res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0d sens')

#post_cells = ax.scatter(Xpost, Ypost, Zpost, c='g', marker = 'o', alpha = 0.3, label = 't=29 d')
post_sens_cells = ax.scatter(Xposts, Yposts, Zposts, c='k', marker = 'o', alpha = 0.9, label = 't=29d sens ')
post_res_cells = ax.scatter(Xpostr, Ypostr, Zpostr, c='c', marker = '^', alpha = 0.1, label = 't=29d res ')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC4')
plt.legend(loc=2, prop={'size': 25})
#plt.title('Pre and post-treatment cells in PC space',fontsize= 20)

#ax.legend([sens_cells, res_cells], ['t=0 hr sens', 't=0 hr res'])

ax.azim = 100
ax.elev = -50


#%% PC1 vs PC2

fig = plt.figure(figsize=(10,10))

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[0])
Ys=np.asarray(PC_dfsens[1])


Xr= np.asarray(PC_dfres[0])
Yr=np.asarray(PC_dfres[1])



Xint= np.asarray(PCintdf[0])
Yint=np.asarray(PCintdf[1])

Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[1])


res_cells = plt.scatter(Xr, Yr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = plt.scatter(Xs, Ys,  c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = plt.scatter(Xint, Yint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = plt.scatter(Xpost, Ypost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

plt.xlabel('PC1')
plt.ylabel('PC2')

plt.legend(loc=2, prop={'size': 25})

#%% PC1 vs PC3

fig = plt.figure(figsize=(10,10))

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[0])
Ys=np.asarray(PC_dfsens[2])


Xr= np.asarray(PC_dfres[0])
Yr=np.asarray(PC_dfres[2])



Xint= np.asarray(PCintdf[0])
Yint=np.asarray(PCintdf[2])

Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[2])


res_cells = plt.scatter(Xr, Yr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = plt.scatter(Xs, Ys,  c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = plt.scatter(Xint, Yint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = plt.scatter(Xpost, Ypost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

plt.xlabel('PC1')
plt.ylabel('PC3')

plt.legend(loc=2, prop={'size': 25})

#%% PC2 vs PC3

fig = plt.figure(figsize=(10,10))

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[1])
Ys=np.asarray(PC_dfsens[2])


Xr= np.asarray(PC_dfres[1])
Yr=np.asarray(PC_dfres[2])



Xint= np.asarray(PCintdf[1])
Yint=np.asarray(PCintdf[2])

Xpost= np.asarray(PCpostdf[1])
Ypost=np.asarray(PCpostdf[2])


res_cells = plt.scatter(Xr, Yr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = plt.scatter(Xs, Ys,  c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = plt.scatter(Xint, Yint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = plt.scatter(Xpost, Ypost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

plt.xlabel('PC2')
plt.ylabel('PC3')

plt.legend(loc=2, prop={'size': 25})
#%% PC1 vs PC4

fig = plt.figure(figsize=(10,10))

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[0])
Ys=np.asarray(PC_dfsens[3])


Xr= np.asarray(PC_dfres[0])
Yr=np.asarray(PC_dfres[3])



Xint= np.asarray(PCintdf[0])
Yint=np.asarray(PCintdf[3])

Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[3])


res_cells = plt.scatter(Xr, Yr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = plt.scatter(Xs, Ys,  c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = plt.scatter(Xint, Yint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
#post_cells = plt.scatter(Xpost, Ypost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

plt.xlabel('PC1')
plt.ylabel('PC4')

plt.legend(loc=2, prop={'size': 25})


#%% PC2 vs PC4

fig = plt.figure(figsize=(10,10))

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[1])
Ys=np.asarray(PC_dfsens[3])


Xr= np.asarray(PC_dfres[1])
Yr=np.asarray(PC_dfres[3])



Xint= np.asarray(PCintdf[1])
Yint=np.asarray(PCintdf[3])

Xpost= np.asarray(PCpostdf[1])
Ypost=np.asarray(PCpostdf[3])


res_cells = plt.scatter(Xr, Yr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = plt.scatter(Xs, Ys,  c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = plt.scatter(Xint, Yint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = plt.scatter(Xpost, Ypost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

plt.xlabel('PC2')
plt.ylabel('PC4')

plt.legend(loc=2, prop={'size': 25})

#%% PC3 vs PC4

fig = plt.figure(figsize=(10,10))

PC_dfsens = PC_df[PC_df['classlabel']==0]
PC_dfres = PC_df[PC_df['classlabel']==1]
Xs= np.asarray(PC_dfsens[2])
Ys=np.asarray(PC_dfsens[3])


Xr= np.asarray(PC_dfres[2])
Yr=np.asarray(PC_dfres[3])



Xint= np.asarray(PCintdf[2])
Yint=np.asarray(PCintdf[3])

Xpost= np.asarray(PCpostdf[2])
Ypost=np.asarray(PCpostdf[3])


res_cells = plt.scatter(Xr, Yr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = plt.scatter(Xs, Ys,  c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = plt.scatter(Xint, Yint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = plt.scatter(Xpost, Ypost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

plt.xlabel('PC3')
plt.ylabel('PC4')

plt.legend(loc=2, prop={'size': 25})



#%%
ax1=sns.scatterplot(PC_df[1], PC_df[2], hue= PC_df['classlabel'])
ax1.set(xlabel ='PC2', ylabel ='PC3') 

ax2=sns.scatterplot(PC_df[2], PC_df[3], hue= PC_df['classlabel'])
ax2.set(xlabel ='PC3', ylabel ='PC4') 

ax3=sns.scatterplot(PC_df[0], PC_df[2], hue= PC_df['classlabel'])
ax3.set(xlabel ='PC1', ylabel ='PC3') 

ax4=sns.scatterplot(PC_df[0], PC_df[3], hue= PC_df['classlabel'])
ax4.set(xlabel ='PC1', ylabel ='PC4')

ax5=sns.scatterplot(PC_df[1], PC_df[3], hue= PC_df['classlabel'])
ax5.set(xlabel ='PC2', ylabel ='PC4')

fig = plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection='3d')

PC_dfsens = PC_df[PC_df['survivor']=='sens']
PC_dfres = PC_df[PC_df['survivor']=='res']
Xs= np.asarray(PC_dfsens[0])
Ys=np.asarray(PC_dfsens[2])
Zs=np.asarray(PC_dfsens[3])

Xr= np.asarray(PC_dfres[0])
Yr=np.asarray(PC_dfres[1])
Zr=np.asarray(PC_dfres[2])

ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1)
ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3)


ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.azim = 100
ax.elev = -50

#%% Build SVM Model and apply to the subsequent time points

clf.fit(X, y)
y_pre_SVM= clf.predict(X)
mu_pre_SVM= sum(y_pre_SVM)/len(X)

y_int_SVM = clf.predict(Xint)
mu_int_SVM = sum(y_int_SVM)/len(Xint)

y_post_SVM = clf.predict(Xpost)
mu_post_SVM = sum(y_post_SVM)/len(Xpost)

print(mu_pre_SVM)
print(mu_int_SVM)
print(mu_post_SVM)


    

    
    
    
    
    
    
    