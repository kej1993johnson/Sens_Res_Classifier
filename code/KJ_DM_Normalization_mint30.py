#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:37:22 2019

@author: kj22643
"""

# Script to normalize the data and redo the umap projection without t=30 hr time point
# Script adapted from Daylin
#%% Loading Libraries
%reset

import scanpy as sc
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import seaborn as sb

import seaborn as sns
import os
import sys

import rpy2.rinterface_lib.callbacks
import logging

from rpy2.robjects import pandas2ri
import anndata2ri
import leidenalg


path = '/Users/kj22643/Documents/Documents/231_Classifier_Project/data'
os.chdir(path)
os.chdir(path)
sc.settings.figdir = 'KJ_plots'
sc.set_figure_params(dpi_save=300)
sc.settings.verbosity = 3

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
pandas2ri.activate()
anndata2ri.activate()
%load_ext rpy2.ipython
sc.settings.verbosity = 3
sc.set_figure_params(dpi_save=300)
sc.logging.print_versions()

figDir = path+'DM_plots/'
sc.settings.figdir = figDir


import warnings
warnings.filterwarnings('ignore')
#%% R
# Load all the R libraries we will be using in the notebook
library(scran)
#%% Importing Data and lineage assignments

#set up files for loading data 
cellranger_input='filtered_feature_bc_matrix'
lineage_assignment='10x_231.lineage_assignment.clustered_lineage.singletons_only.tsv'

#read in 10x data 
adata=sc.read_10x_mtx(cellranger_input, var_names='gene_symbols', cache=True)

#add sample names based off proper cell name annotation 
adata.var_names_make_unique()
dic = {'1':'BgL1K','2':'30Hr','3':'Res-1','4':'Res-2'}
adata.obs['sample'] = adata.obs.index.str[-1].map(dic)

#read in lineage data 
lineages=pd.read_csv(lineage_assignment,sep='\t')
lineages=lineages[['proper_cell_name','clustered_lineage']]
lineages=lineages.rename(columns={'clustered_lineage':'lineage'})
lineages.set_index('proper_cell_name',inplace=True)
adata.obs = adata.obs.join(lineages)
#%% Now that we have adata object, remove the 30 hour time point
adata = adata[adata.obs['sample']!='30Hr', :]

plt.style.use('default')
plt.rcParams['figure.figsize']=(8,8)

#%%Old QC Methods
adata.obs['n_genes'] = (adata.X > 0).sum(1)
mito_genes = adata.var_names.str.startswith('MT-')
# for each cell compute fraction of counts in mito genes vs. all genes
# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
# add the total counts per cell as observations-annotation to adata
adata.obs['n_counts'] = adata.X.sum(axis=1).A1

#%% Quality control - plot QC metrics
#Sample quality plots
t1 = sc.pl.violin(adata, 'n_counts', groupby='sample', size=2, log=True, cut=0)
t2 = sc.pl.violin(adata, 'percent_mito', groupby='sample')

#%%Data quality summary plots
p1 = sc.pl.scatter(adata, 'n_counts', 'n_genes', color='percent_mito')
p2 = sc.pl.scatter(adata[adata.obs['n_counts']<10000], 'n_counts', 'n_genes', color='percent_mito')