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
adata = sc.read('daylin_anndata.h5ad')
adata.obs.head()
# current samples:
#BgL1K
#30hr
#Rel-1
#Rel-2
# We will change these to time points
# Count the number of unique lineages in all the cells
uniquelins = adata.obs['lineage'].unique()
nunique = len(uniquelins)



#%% Assign survivor category in adata.obs
# Lineages that appear in either of two post-treatment samples
longTreatLins = adata.obs.loc[(adata.obs['sample'].isin(['Rel-1','Rel-2']))&(adata.obs.lineage!='nan'),'lineage'].unique().tolist()


# Find any lineages that appear in both (in this case there is only one)
resLins1 = adata.obs.loc[(adata.obs['sample'].isin(['Rel-1']))&(adata.obs.lineage!='nan'),'lineage'].unique().tolist()
resLins2 = adata.obs.loc[(adata.obs['sample'].isin(['Rel-2']))&(adata.obs.lineage!='nan'),'lineage'].unique().tolist()
resLins = list(set(resLins1) & set(resLins2))

#lineage_abundance= adata.obs['lineage'].value_counts()

#top = adata.obs.loc[adata.obs.lineage!='nan','lineage'].value_counts()[:9].index.tolist()

# %%try to rename the samples by time point
samps= adata.obs['sample'].unique()

timepoint = np.array(['t=0hr', 't=30hr', 't=1344hr'])

adata.obs.loc[adata.obs['sample']==samps[0], 'timepoint']='t=0hr'
adata.obs.loc[adata.obs['sample']==samps[1], 'timepoint']='t=30hr'
adata.obs.loc[adata.obs['sample']==samps[2], 'timepoint']='t=1344hr'
adata.obs.loc[adata.obs['sample']==samps[3], 'timepoint']='t=1344hr'

print(adata.obs['timepoint'].unique())



#%% Separately make dataframes for the pre-treatment, intermediate, and post treatment samples
# t=0 hr (pre-treatment), 3182 pre treatment cells
# We want to keep the info about the lineage so we can potentially
# use it to make evenly divided testing and training data sets
adata_pre = adata[adata.obs['timepoint']=='t=0hr', :]
dfpre = pd.concat([adata_pre.obs['lineage'], adata_pre.obs['leiden'],
               pd.DataFrame(adata_pre.raw.X,index=adata_pre.obs.index,
                            columns=adata_pre.var_names),], axis=1) 
# t = 30 hr (intermediate timepoint) 5169 int treatment cells
adata_int = adata[adata.obs['timepoint']=='t=30hr', :]
dfint = pd.concat([adata_int.obs['lineage'], adata_int.obs['leiden'],
                   pd.DataFrame(adata_int.raw.X, index=adata_int.obs.index, 
                                columns = adata_int.var_names),], axis=1)

# t=1344 hr (~roughly 8 weeks), 10332 post treatment cells
adata_post = adata[adata.obs['timepoint']=='t=1344hr', :]
dfpost = pd.concat([adata_post.obs['lineage'], adata_post.obs['leiden'],
                    pd.DataFrame(adata_post.raw.X, index=adata_post.obs.index, 
                                 columns = adata_post.var_names),],axis =1)
#%% Try to add a .obs column that records lineage abundance from the different samples
linAbundpre= adata_pre.obs['lineage'].value_counts()
linAbundint = adata_int.obs['lineage'].value_counts()
linAbundpost = adata_post.obs['lineage'].value_counts()

# Start by adding the linabundpre and lin abund post to the pre-treatment data frame
df1 = pd.DataFrame(linAbundpre)
df1['linabundpre']= df1.lineage
df1=df1.drop(['lineage'], axis=1)
df1['lineage'] = df1.index
df1=df1.drop(index='nan')


df2 = pd.DataFrame(linAbundpost)
df2['linabundpost']= df2.lineage
df2=df2.drop(['lineage'], axis=1)
df2['lineage'] = df2.index
df2=df2.drop(index='nan')
#%% Merge the linage abundance data frames from the pre and post treatment samples into dfpre
dfpre= pd.DataFrame.merge(df1, dfpre, left_on=['lineage'], 
              right_on=['lineage'], how='right')
dfpre = pd.DataFrame.merge(df2, dfpre, left_on=['lineage'],
              right_on=['lineage'], how='right') 
dfpre['linabundpost'] = dfpre['linabundpost'].fillna(0)
dfpre['linabundpre']= dfpre['linabundpre'].fillna(0)
#%% Make a column that is the logfoldchange from post to pre
import math
dfpre['foldchange'] =  (dfpre['linabundpost']-dfpre['linabundpre'])/dfpre['linabundpre']
print(dfpre['foldchange'].unique())
#%%
dfpre['logfoldchange'] = np.log(dfpre['foldchange'])
dfpre['logfoldchange']= dfpre['logfoldchange'].fillna(0)
print(dfpre['logfoldchange'])
print(dfpre['foldchange'].unique())
print(dfpre['logfoldchange'].unique())
#%%

plt.hist(dfpre['logfoldchange'], range = [3, 7], bins = 100)
plt.xlabel('log(foldchange) of lineage abundance')
plt.ylabel('number of cells')
#%%
plt.hist(dfpre['foldchange'], range = [0, 500], bins = 100)
plt.xlabel('foldchange of lineage abundance')
plt.ylabel('number of cells')

#%% Make the survivor column, but don't label it yet. 
#dfpre['survivor'] =np.where(dfpre['linabundpost'] >1000, 'res','sens')
#%% Set a log fold change cutoff
dfpre['survivor'] = np.where(dfpre['foldchange']>0, 'res', 'sens')

#%% Use sklearn to do principle component analysis on the  entire pre-treatment sample
#X = dfpre.loc[:, dfpre.columns !='survivor', dfpre.columns !='lineage']
X = dfpre.drop(columns= ['survivor', 'lineage', 'leiden', 'linabundpost', 'linabundpre', 'foldchange', 'logfoldchange'])

y= pd.factorize(dfpre['survivor'])[0] 
y ^= 1
leidlist = dfpre['leiden']
ncells = len(y)

leidlistint = dfint['leiden']
leidlistpost = dfpost['leiden']

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
y_SVM = {}
mu_SVM = {}
sigmasq_SVM = {}
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
    sigamasq_PCA[i] = mu_PCA[i]*(1-mu_PCA[i])./folds_dict['ntest'][i]
    # SVM MODEL OUTPUTS FOR EACH FOLD
    clf.fit(X_train, y_train)
    y_SVM[i]= clf.predict(X_test)
    mu_SVM[i] = sum(y_SVM[i])/folds_dict['ntest'][i]
    sigmasq_SVM[i] = mu_SVM[i]*(1-mu_SVM[i])./folds_dict['ntest'][i]
    
#%%  Put into folds_dict
folds_dict['V_train'] = V_train
folds_dict['y_PCA']= y_PCA
folds_dict['mu_PCA'] = mu_PCA
folds_dict['sigmasq_PCA'] = sigmasq_PCA
folds_dict['y_SVM'] = y_SVM
folds_dict['mu_SVM'] = mu_SVM
folds_dict['sigmasq_SVM'] = sigmasq_SVM

   
    
#%%Compare the mu_SVM and mu_PCA test set estimates of the expectation to the known
    # prevalence of the test set
df = pd.DataFrame()
dfprevtest = pd.DataFrame(folds_dict['prevtest'], index=[0])
dfmu_PCA= pd.DataFrame(folds_dict['mu_PCA'], index = [0])
dfmu_SVM = pd.DataFrame(folds_dict['mu_SVM'], index = [0])

npprevtest=np.array(dfprevtest)
npmu_PCA = np.array(dfmu_PCA)
npmu_SVM = np.array(dfmu_SVM)
mu_pre = np.mean(npprevtest)
mu_hat_PCA = np.mean(npmu_PCA)
mu_hat_SVM = np.mean(npmu_SVM)
ntest = folds_dict['ntest'][0]
print(mu_hat_PCA)
print(mu_hat_SVM)
                 
sigmasq_PCA =(1-mu_hat_PCA)*mu_hat_PCA/ntest
sigmasq_SVM = (1-mu_hat_SVM)*mu_hat_SVM/ntest
print(sigmasq_PCA)
print(sigmasq_SVM)


#%% Next step, apply the models to the subsequent time points!

# Make your cell gene matrices
Xint = dfint.drop(columns= ['lineage', 'leiden'])
Xpost = dfpost.drop(columns =['lineage', 'leiden'])
#%%
# Already defined PCA and SVM hyperparameters above

# Start by builing model using all the data from the pre-treatment time point
pca.fit(X, y)
# Compute the eigenvector space 
V = pca.fit_transform(X)
# Fit a nearest neighbor classifier on the model built on the training data set
knn.fit(pca.transform(X), y)
  

# Apply the pca and knn classifier to the pre, int, and post treatment samples. 
y_pre1 = knn.predict(pca.transform(X))
pre_prob= knn.predict_proba(pca.transform(X))
B = pre_prob[:,1]>0
y_pre = B*1
mu_pre_PCA = sum(y_pre)/len(X)
# Now that you have the PC model, apply it to intermediate and post treatment samples.
y_int = knn.predict(pca.transform(Xint))
mu_int_PCA= sum(y_int)/len(Xint)
sigmasq_int_PCA = mu_int_PCA*(1-mu_int_PCA)/len(Xint)



y_post = knn.predict(pca.transform(Xpost))
mu_post_PCA= sum(y_post)/len(Xpost)
sigmasq_post_PCA = mu_post_PCA*(1-mu_post_PCA)/len(Xpost)
#%% Print the PCA estimate outputs
print(mu_pre_PCA)
print(mu_int_PCA)
print(mu_post_PCA)
#%% Project the intermediate and post treatment samples into the principal component space
# This is mostly for visualiation purposes, but also to evaluate if the post-treatment 
# cells are more different in some principal components than others




PCspre=pca.transform(X)
PCsint = pca.transform(Xint)
PCspost = pca.transform(Xpost)

#%%
# Pre-treatment
PC_df = pd.DataFrame(PCspre)
PC_df.reset_index(drop=True, inplace=True)
PC_df['classlabel'] = y_pre
PC_df['time'] = 0
leidlist = leidlist.astype('int64')
PC_df['leiden'] = leidlist.values

# t= 30 hr
PCintdf = pd.DataFrame(PCsint)
PCintdf.reset_index(drop=True, inplace=True)
PCintdf['classlabel'] = y_int
PCintdf['time'] = 30
leidlistint = leidlistint.astype('int64')
PCintdf['leiden']= leidlistint.values

# t = 1344 hr
PCpostdf = pd.DataFrame(PCspost)
PCpostdf.reset_index(drop=True, inplace=True)
PCpostdf['classlabel'] = y_post
PCpostdf['time'] = 1344
leidlistpost = leidlistpost.astype('int64')
PCpostdf['leiden'] = leidlistpost.values


PCall = pd.concat([PC_df,PCintdf, PCpostdf], axis=0)
# %% Plot some things
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax=sns.scatterplot(PCall[0], PCall[1], hue= PCall['leiden'],s=10, palette = 'Set2')
ax.set(xlabel ='PC1', ylabel ='PC2') 
#%%
ax=sns.scatterplot(PCall[0], PCall[1], hue= PCall['classlabel'], s=10)
ax.set(xlabel ='PC1', ylabel ='PC2') 
#%%
ax=sns.scatterplot(PCall[0], PCall[1], hue= PCall['time'], s=10)
ax.set(xlabel ='PC1', ylabel ='PC2') 

#%% Plot the first two components of the data frame my leidein cluster
ax=sns.scatterplot(PC_df[0], PC_df[1], hue= PC_df['classlabel'], s=10)
ax.set(xlabel ='PC1', ylabel ='PC2') 
#%%
ax=sns.scatterplot(PC_df[0], PC_df[1], hue= PC_df['leiden'])
ax.set(xlabel ='PC1', ylabel ='PC2') 


# A
#%% PC1, PC2, & PC3
fig = plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection='3d')

PC_df4 = PC_df[PC_df['leiden']==4]
PC_df3 = PC_df[PC_df['leiden']==3]
PC_dfmin4 = PC_df[PC_df['leiden']!=4]
PC_dfother = PC_dfmin4[PC_dfmin4['leiden']!=3]
X4= np.asarray(PC_df4[0])
Y4=np.asarray(PC_df4[1])
Z4=np.asarray(PC_df4[2])

X3= np.asarray(PC_df3[0])
Y3=np.asarray(PC_df3[1])
Z3=np.asarray(PC_df3[2])


Xnot= np.asarray(PC_dfother[0])
Ynot=np.asarray(PC_dfother[1])
Znot=np.asarray(PC_dfother[2])


#Xint= np.asarray(PCintdf[0])
#Yint=np.asarray(PCintdf[1])
#Zint=np.asarray(PCintdf[2])

#Xpost= np.asarray(PCpostdf[0])
#Ypost=np.asarray(PCpostdf[1])
#Zpost=np.asarray(PCpostdf[2])

leid4_cells = ax.scatter(X4, Y4, Z4, c='y', marker='^', alpha = 1, label = 't=0 hr leiden 4')
leid3_cells = ax.scatter(X3, Y3, Z3, c='g', marker='^', alpha = 1, label = 't=0 hr leiden 3')
other_cells = ax.scatter(Xnot, Ynot, Znot, c='k', marker='o', alpha = 0.3, label = 't= 0 hr others')
#int_cells = ax.scatter(Xint, Yint, Zint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
#post_cells = ax.scatter(Xpost, Ypost, Zpost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(loc=2, prop={'size': 25})
#plt.title('Pre and post-treatment cells in PC space',fontsize= 20)

#ax.legend([sens_cells, res_cells], ['t=0 hr sens', 't=0 hr res'])

ax.azim = 100
ax.elev = -50

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


Xint= np.asarray(PCintdf[0])
Yint=np.asarray(PCintdf[1])
Zint=np.asarray(PCintdf[2])

Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[1])
Zpost=np.asarray(PCpostdf[2])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = ax.scatter(Xint, Yint, Zint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = ax.scatter(Xpost, Ypost, Zpost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

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


Xint= np.asarray(PCintdf[1])
Yint=np.asarray(PCintdf[2])
Zint=np.asarray(PCintdf[3])

Xpost= np.asarray(PCpostdf[1])
Ypost=np.asarray(PCpostdf[2])
Zpost=np.asarray(PCpostdf[3])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = ax.scatter(Xint, Yint, Zint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = ax.scatter(Xpost, Ypost, Zpost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

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


Xint= np.asarray(PCintdf[0])
Yint=np.asarray(PCintdf[2])
Zint=np.asarray(PCintdf[3])

Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[2])
Zpost=np.asarray(PCpostdf[3])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = ax.scatter(Xint, Yint, Zint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = ax.scatter(Xpost, Ypost, Zpost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

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


Xint= np.asarray(PCintdf[0])
Yint=np.asarray(PCintdf[1])
Zint=np.asarray(PCintdf[3])

Xpost= np.asarray(PCpostdf[0])
Ypost=np.asarray(PCpostdf[1])
Zpost=np.asarray(PCpostdf[3])

res_cells = ax.scatter(Xr, Yr, Zr, c='b', marker='^', alpha = 1, label = 't=0 hr res')
sens_cells = ax.scatter(Xs, Ys, Zs, c='r', marker='o', alpha = 0.3, label = 't= 0 hr sens')
#int_cells = ax.scatter(Xint, Yint, Zint, c='g', marker = 'o', alpha = 0.2, label = 't=30 hr')
post_cells = ax.scatter(Xpost, Ypost, Zpost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

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
post_cells = plt.scatter(Xpost, Ypost, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

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


    

    
    
    
    
    
    
    