

#This script identifies lineages who increased in abundance (labeled as resistant)
# and those who decreased in abundance (labeled as sensitive)

# Using just the cells from these identified lineages in the pre-treatment sample,
# we will build a classifier from this subset of data. Then, we will apply it
# to each cell in the pre, intermediate, and post-treatment data sets

# We will project the cells labeled as resistant or sensitive into the PC space 
# of both all of the cells and just the cells used for classifying.




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
import math
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
from sklearn.cluster import KMeans
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
adata = sc.read('post-cell-cycle-regress.h5ad')
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



#%% Identify lineages that have been recalled from the pre-treatment sample
# Make a column labeled recalled lin that can be use to identify the specific lineages of interest

# These correspond to AA161 and AA170
# AA170 is the highly proliferative lineage that ultimately decreases in lineage abundance
# AA161 is a lowly abundant in the pre-treatment but is one of the ones that comes
# to  be highly abundance in both post treatment samples

reslin = ['GTACATTTTCATACCTCTCG']
senslin = ['GTGTGTTGTTTTTGTACGGC']


adata.obs.loc[adata.obs.lineage.isin(reslin)==True,'recalled_lin'] = 'res'
adata.obs.loc[adata.obs.lineage.isin(reslin)==False,'recalled_lin'] = 'na'
adata.obs.loc[adata.obs.lineage.isin(senslin)==True,'recalled_lin'] = 'sens'

print(adata.obs['recalled_lin'])
#%% Add a column to rename the samples by time point
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
dfall = pd.concat([adata.obs['lineage'],adata.obs['timepoint'], 
               pd.DataFrame(adata.raw.X,index=adata.obs.index,
                            columns=adata.var_names),], axis=1) 
ncells = len(dfall)
print(ncells)
adata_pre = adata[adata.obs['timepoint']=='t=0hr', :]
dfpre = pd.concat([adata_pre.obs['lineage'], adata_pre.obs['recalled_lin'],
               pd.DataFrame(adata_pre.raw.X,index=adata_pre.obs.index,
                            columns=adata_pre.var_names),], axis=1) 
npre = len(dfpre)
print(npre)
# t = 30 hr (intermediate timepoint) 5169 int treatment cells
adata_int = adata[adata.obs['timepoint']=='t=30hr', :]
dfint = pd.concat([adata_int.obs['lineage'], adata_int.obs['recalled_lin'],
                   pd.DataFrame(adata_int.raw.X, index=adata_int.obs.index, 
                                columns = adata_int.var_names),], axis=1)
nint = len(dfint)
print(nint)
# t=1344 hr (~roughly 8 weeks), 10332 post treatment cells
adata_post = adata[adata.obs['timepoint']=='t=1344hr', :]
dfpost = pd.concat([adata_post.obs['lineage'], adata_post.obs['recalled_lin'],
                    pd.DataFrame(adata_post.raw.X, index=adata_post.obs.index, 
                                 columns = adata_post.var_names),],axis =1)
npost = len(dfpost)
print(npost)
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

dfpre['foldchange'] =  (dfpre['linabundpost']-dfpre['linabundpre'])
print(dfpre['foldchange'].unique())
foldchangevec = dfpre['foldchange']
#%% Look at fold change and log fold change for each cell and its correpsonding lineage
dfpre['logfoldchange'] = np.log(dfpre['foldchange'])
dfpre['logfoldchange']= dfpre['logfoldchange'].fillna(0)


print(dfpre['logfoldchange'].unique())

# Figures of logfold change and fold change
plt.figure()
plt.hist(dfpre['logfoldchange'], range = [3, 7], bins = 100)
plt.xlabel('log(foldchange) of lineage abundance')
plt.ylabel('number of cells')

plt.figure()
plt.hist(dfpre['foldchange'],range = [0, 500], bins = 100)
plt.xlabel('foldchange of lineage abundance')
plt.ylabel('number of cells')

#%% Make the survivor column, but don't label it yet. 
#dfpre['survivor'] =np.where(dfpre['linabundpost'] >1000, 'res','sens'
# Want to call cells that have an increase in lineage abundance resistant
dfpre.loc[dfpre.foldchange>0, 'survivor'] = 'res'

dfpre.loc[dfpre.foldchange<-200, 'survivor']= 'sens'
survivorvec = dfpre['survivor']


# Want to call cells that have an signficant in lineage abundance resistant

# Make a dataframe that only contains cells who are given a sens or resistant label

dfsr = dfpre[(dfpre['survivor']=='sens') | (dfpre['survivor'] == 'res')]
nclass = len(dfsr)
print(nclass)
y= pd.factorize(dfsr['survivor'])[0] 
y ^= 1
mu_sr = sum(y)/len(y)
print(mu_sr)
Xsr = dfsr.drop(columns= [ 'lineage', 'linabundpost', 'recalled_lin', 'linabundpre', 'foldchange', 'logfoldchange', 'survivor'])

dfpremin = dfpre[(dfpre['survivor']!='sens') & (dfpre['survivor'] != 'res')]
Xpremin = dfpremin.drop(columns = [ 'lineage', 'linabundpost', 'recalled_lin', 'linabundpre', 'foldchange', 'logfoldchange', 'survivor'])
npremin = len(dfpremin)
print(npremin)
#%% Make your gene cell matrices, for all time points and as isolated time points


Xpre= dfpre.drop(columns= [ 'lineage', 'linabundpost', 'recalled_lin', 'linabundpre', 'foldchange', 'logfoldchange', 'survivor'])
Xall = dfall.drop(columns = ['lineage', 'timepoint'])

npre = len(Xpre)
print(npre)

Xint = dfint.drop(columns= ['lineage', 'recalled_lin'])
Xpost = dfpost.drop(columns =['lineage', 'recalled_lin'])
nint = len(Xint)
npost = len(Xpost)


# X is your cell gene matrix, y is your class labels


# %%Assign the optimal parameters (foudn from KJ_classify_sklearn.py) for building your prediction model 
# p(x|Sj) where Sj is your training set and x is any new cell (in your test set or in future cells)
# Will want to redo these optimized hyperparameters with the new way of classifying cells as sensitive or resistant by cluster
n_neighbors = 15


knn = KNeighborsClassifier(n_neighbors=n_neighbors)

#pca=PCA(copy=True, iterated_power='auto', n_components=n_components, random_state=0,
            #svd_solver='auto', tol=0.0, whiten=False)

pca = PCA(n_components=0.90, svd_solver='full', random_state = 0)



#%% Perform PCA on the all cells from all sample 
# Already defined PCA and SVM hyperparameters above

# Start by builing model using all the data from the pre-treatment time point
pca.fit(Xsr, y)
# Compute the eigenvector space 
# These should all be ncells x ncomponents matrices
V = pca.fit_transform(Xsr)
PCsr = pca.transform(Xsr)
PCsall = pca.transform(Xall)
PCspremin=pca.transform(Xpremin)
PCsint = pca.transform(Xint)
PCspost = pca.transform(Xpost)
#%% Look at explained variance
var_in_PCs= pca.explained_variance_ratio_
cdf_varPCs = var_in_PCs.cumsum()
print(var_in_PCs)
print(cdf_varPCs)
#%% Fit a nearest neighbor classifier on the model built on the training data set
knn.fit(pca.transform(Xsr), y)
  

# Apply the pca and knn classifier to the pre, int, and post treatment samples. 
y_premin = knn.predict(pca.transform(Xpremin))
pre_prob= knn.predict_proba(pca.transform(Xpremin))
B = pre_prob[:,1]>0
y_pre = B*1
mu_pre_PCA = (sum(y_pre)+sum(y))/(len(Xpremin) + len(Xsr))
print(mu_pre_PCA)

y_int = knn.predict(pca.transform(Xint))
int_prob= knn.predict_proba(pca.transform(Xint))
C = int_prob[:,1]>0
y_intpr = C*1
mu_int_PCA = sum(y_intpr)/(len(y_intpr))
print(mu_int_PCA)


y_post = knn.predict(pca.transform(Xpost))
post_prob= knn.predict_proba(pca.transform(Xpost))
D= post_prob[:,1]>0
y_postpr = D*1
mu_post_PCA = sum(y_postpr)/(len(y_postpr))
print(mu_post_PCA)



#%% Make data frames for the pre, int, and post-treatment cells in PC space

# Pre-treatment
PCsrdf = pd.DataFrame(PCsr)
PCsrdf['classlabel'] = y
PCsrdf['time'] = 0
PCsrdf.reset_index(drop=True, inplace=True)
PCsrdf['recalled_lin'] = dfsr['recalled_lin']


# Pre-treatment
PCpmdf = pd.DataFrame(PCspremin)
PCpmdf['classlabel'] = y_pre
PCpmdf.reset_index(drop=True, inplace=True)
PCpmdf['time'] = 0
PCpmdf['recalled_lin'] = dfpremin['recalled_lin']


# t= 30 hr
PCintdf = pd.DataFrame(PCsint)
PCintdf['classlabel'] = y_intpr
PCintdf.reset_index(drop=True, inplace=True)
#PCintdf['kclust'] = kclustint
PCintdf['time'] = 30
PCintdf['recalled_lin'] = 'nan'


# t = 1344 hr
PCpostdf = pd.DataFrame(PCspost)
PCpostdf['classlabel'] = y_postpr
PCpostdf.reset_index(drop=True, inplace=True)
#PCpostdf['kclust'] = kclustpost
PCpostdf['time'] = 1344
PCpostdf['recalled_lin'] = 'nan'



#%% PC1 PC2 and PC3


# First just look at the cells used to build the classifier
PCsrdfs = PCsrdf[PCsrdf['classlabel']==0]
PCsrdfr = PCsrdf[PCsrdf['classlabel']==1]
# Then other breakdowns
PCsrdfls = PCsrdf[PCsrdf['recalled_lin']=='sens']
PCsrdflr = PCsrdf[PCsrdf['recalled_lin']=='res']
PCpmdfs = PCpmdf[PCpmdf['classlabel']==0]
PCpmdfr = PCpmdf[PCpmdf['classlabel']==1]
PCintdfs = PCintdf[PCintdf['classlabel']==0]
PCintdfr = PCintdf[PCintdf['classlabel'] == 1]
PCpostdfs = PCpostdf[PCpostdf['classlabel']==0]
PCpostdfr = PCpostdf[PCpostdf['classlabel'] == 1]
#%% WHY DOES RUNNING THIS CHANGE MY PC DATAFRAMES????
# Cells used for classifying
xsrs= np.asarray(PCsrdfs[0])
ysrs=np.asarray(PCsrdfs[1])
zsrs=np.asarray(PCsrdfs[2])
#
xsrr= np.asarray(PCsrdfr[0])
ysrr=np.asarray(PCsrdfr[1])
zsrr=np.asarray(PCsrdfr[2])

# Lineages that we isolated
xsrls= np.asarray(PCsrdfls[0])
ysrls=np.asarray(PCsrdfls[1])
zsrls=np.asarray(PCsrdfls[2])

xsrlr= np.asarray(PCsrdflr[0])
ysrlr=np.asarray(PCsrdflr[1])
zsrlr=np.asarray(PCsrdflr[2])

# Pre-treat samples
xp= np.asarray(PCpmdf[0])
yp=np.asarray(PCpmdf[1])
zp=np.asarray(PCpmdf[2])

#pre-treat sensitive
xps= np.asarray(PCpmdfs[0])
yps=np.asarray(PCpmdfs[1])
zps=np.asarray(PCpmdfs[2])
# pre-treat resistant
xpr= np.asarray(PCpmdfr[0])
ypr=np.asarray(PCpmdfr[1])
zpr=np.asarray(PCpmdfr[2])

# t=30 hr cells
xi= np.asarray(PCintdf[0])
yi=np.asarray(PCintdf[1])
zi=np.asarray(PCintdf[2])
#sens and res labels
xis= np.asarray(PCintdfs[0])
yis=np.asarray(PCintdfs[1])
zis=np.asarray(PCintdfs[2])
xir= np.asarray(PCintdfr[0])
yir=np.asarray(PCintdfr[1])
zir=np.asarray(PCintdfr[2])

# t=1344 hr cells
xpo= np.asarray(PCpostdf[0])
ypo=np.asarray(PCpostdf[1])
zpo=np.asarray(PCpostdf[2])
# sens and res labels
xpos= np.asarray(PCpostdfs[0])
ypos=np.asarray(PCpostdfs[1])
zpos=np.asarray(PCpostdfs[2])
xpor= np.asarray(PCpostdfr[0])
ypor=np.asarray(PCpostdfr[1])
zpor=np.asarray(PCpostdfr[2])
#%%
fig = plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection='3d')

# pre-treat cells for classifying
srs = ax.scatter(xsrs, ysrs, zsrs, c='g', marker='^', alpha = 1, label = 't=0 hr labeled sensitive')
srr = ax.scatter(xsrr, ysrr, zsrr, c='r', marker='^', alpha = 1, label = 't=0 hr labeled resistant')

#pre_cells = ax.scatter(xp, yp, zp, c='b', marker='o', alpha = 0.2, label = 't=0 hr remaining')
# classified pre-treatment cells
#ps = ax.scatter(xps, yps, zps, c='olivedrab', marker='+', alpha = 0.5, label = 't=0 hr est sensitive')
#ps = ax.scatter(xpr, ypr, zpr, c='pink', marker='+', alpha = 0.5, label = 't=0 hr est resistant')

#isolated lineages
#ls_pre = ax.scatter(xsrls, ysrls, zsrls, c='lime', marker='o', alpha = 1, label = 'sensitive lineage AA170')
#lr_pre = ax.scatter(xsrlr, ysrlr, zsrlr, c='fuchsia', marker='o', alpha = 1, label = 'resistant lineage AA161')

#int_cells = ax.scatter(xi, yi, zi, c='grey', marker = 'o', alpha = 0.2, label = 't=30 hr unclassified')
#ints = ax.scatter(xis, yis, zis, c='olivedrab', marker = '+', alpha = 0.5, label = 't=30 hr est sensitive')
#intr = ax.scatter(xir, yir, zir, c='pink', marker = '+', alpha = 0.5, label = 't=30 hr est resistant')

#post_cells = ax.scatter(xpo, ypo, zpo, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')
pos = ax.scatter(xpos, ypos, zpos, c='olivedrab', marker = '+', alpha = 0.5, label = 't=1344hr est sensitive')
por = ax.scatter(xpor, ypor, zpor, c='pink', marker = '+', alpha = 0.5, label = 't=1344 hr est resistant')



ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(loc=2, prop={'size': 25})
#plt.title('Pre and post-treatment cells in PC space',fontsize= 20)

#ax.legend([sens_cells, res_cells], ['t=0 hr sens', 't=0 hr res'])

ax.azim = 100
ax.elev = -50

#%% PC1 vs PC2

fig = plt.figure(figsize=(10,10))

srs = plt.scatter(xsrs, ysrs, c='g', marker='^', alpha = 1, label = 't=0 hr labeled sensitive')
srr = plt.scatter(xsrr, ysrr, c='r', marker='^', alpha = 1, label = 't=0 hr labeled resistant')
#pre_cells = plt.scatter(xp, yp, c='b', marker='o', alpha = 0.2, label = 't=0 hr remaining')
#xps = plt.scatter(xps, yps, c='olivedrab', marker='+', alpha = 0.5, label = 't=0 hr est sensitive')
#yps = plt.scatter(xpr, ypr, c='pink', marker='+', alpha = 0.5, label = 't=0 hr est resistant')

#int_cells = plt.scatter(xi, yi, c='grey', marker = 'o', alpha = 0.2, label = 't=30 hr unclassified')
#ints = plt.scatter(xis, yis, c='olivedrab', marker = '+', alpha = 0.5, label = 't=30 hr est sensitive')
#intr = plt.scatter(xir, yir, c='pink', marker = '+', alpha = 0.5, label = 't=30 hr est resistant')
#post_cells = plt.scatter(xpo, ypo, c='c', marker = 'o', alpha = 0.1, label = 't=1344 hr')

#ls_pre = plt.scatter(xsrls, ysrls,  c='lime', marker='o', alpha = 1, label = 'sensitive lineage AA170')
#lr_pre = plt.scatter(xsrlr, ysrlr, c='fuchsia', marker='o', alpha = 1, label = 'resistant lineage AA161')
pos = plt.scatter(xpos, ypos,  c='olivedrab', marker = '+', alpha = 0.5, label = 't=1344hr est sensitive')
por = plt.scatter(xpor, ypor,  c='pink', marker = '+', alpha = 0.5, label = 't=1344 hr est resistant')


plt.xlabel('PC1')
plt.ylabel('PC2')

plt.legend(loc=1, prop={'size': 15})

#%% PC1 vs PC3

fig = plt.figure(figsize=(10,10))

srs = plt.scatter(xsrs, zsrs, c='g', marker='^', alpha = 1, label = 't=0 hr labeled sensitive')
srr = plt.scatter(xsrr, zsrr, c='r', marker='^', alpha = 1, label = 't=0 hr labeled resistant')
pre_cells = plt.scatter(xp, zp, c='b', marker='o', alpha = 0.1, label = 't=0 hr remaining')


plt.xlabel('PC1')
plt.ylabel('PC3')

plt.legend(loc=1, prop={'size': 15})
#%% PC2 vs PC3

fig = plt.figure(figsize=(10,10))

srs = plt.scatter(ysrs, zsrs, c='g', marker='^', alpha = 1, label = 't=0 hr labeled sensitive')
srr = plt.scatter(ysrr, zsrr, c='r', marker='^', alpha = 1, label = 't=0 hr labeled resistant')
pre_cells = plt.scatter(xp, zp, c='b', marker='o', alpha = 0.2, label = 't=0 hr remaining')


plt.xlabel('PC2')
plt.ylabel('PC3')

plt.legend(loc=1, prop={'size': 15})
#%% Make a consensus dataframe

PCalldf = pd.concat([PCsrdf, PCpmdf, PCintdf, PCpostdf], axis=0)
# Use consensus data frame to look at the separation between the classlabels
ydfs = PCalldf[PCalldf['classlabel']==0]
yplots= ydfs[0]
xs = np.random.normal(0, 0.04, size=len(yplots))

plt.figure()
bp = PCalldf.boxplot(column=0, by='classlabel', grid=False)
# Add some random "jitter" to the x-axis
qu = plot(xs, yplots, 'r.', alpha=0.2)
plt.ylabel('PC1')

plt.figure()
bp = PCalldf.boxplot(column=1, by='classlabel', grid=False)

plt.ylabel('PC2')