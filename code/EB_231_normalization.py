# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os
import scanpy as sc
import seaborn as sns
from plotnine import *
path = '/Users/kj22643/Documents/Documents/231_Classifier_Project/data'
#path = '/stor/scratch/Brock/231_10X_data/'
os.chdir(path)
sc.settings.figdir = 'EB_plots'
sc.set_figure_params(dpi_save=300)
sc.settings.verbosity = 3
#%% 

#read in 10X data to create anndata object called "adata"
adata = sc.read_10x_mtx('agg_all/outs/filtered_feature_bc_matrix',
                        cache=True).copy()
#%%
#add lineage and sample info for each cell
df = pd.read_csv('lineage_analysis/10x_231.lineage_assignment.clustered_lineage.singletons_only.tsv',sep='\t')
df = df.rename(columns={'clustered_lineage':'lineage'})
df.set_index('proper_cell_name',inplace=True)
adata.obs = adata.obs.join(df)
dic = {'1':'Doxneg','2':'Doxpos','3':'107Aziz','4':'113Aziz'}
adata.obs['sample'] = adata.obs.index.str[-1].map(dic)
adata.obs['sample'] = adata.obs['sample'].astype('category').cat.reorder_categories(['Doxneg','Doxpos','107Aziz','113Aziz'])


#since we used the "filtered_feature_bc_matrix", cellranger had already done
#   filtering; this step is just a formality to add n_genes, n_counts and 
#   remove undetected genes
sc.pp.filter_cells(adata, min_genes=500)
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_genes(adata, min_cells=3)

#calculate percent mito
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1

##QC plots
#sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
#             jitter=0.4, multi_panel=True)
sc.pl.scatter(adata, x='n_counts', y='percent_mito')
#sc.pl.scatter(adata, x='n_counts', y='n_genes', color='sample')

#remove doublets and damaged cells
adata = adata[adata.obs['n_counts'] < 75000, :]
adata = adata[adata.obs['percent_mito'] < 0.14, :]
sc.pp.filter_genes(adata, min_cells=3)

#normalize, log transform, and center each gene to mean of zero
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata.X -= adata.X.mean(axis=0)

#cell cycle scoring
cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
adata.obs['cc_difference'] = adata.obs['S_score'] - adata.obs['G2M_score']
sc.pp.regress_out(adata, 'cc_difference',n_jobs=24)
adata.X -= adata.X.mean(axis=0) #rescale

#%%pca
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata)

#make nearest neighbors graph
sc.pp.neighbors(adata, n_pcs=15)

#assign cells to clusters
sc.tl.leiden(adata, resolution = 0.6)

#write to file
adata.write('EB_clustered.h5ad')
#%%
adata = sc.read('EB_clustered.h5ad')

top = adata.obs.loc[adata.obs.lineage!='nan','lineage'].value_counts()[:9].index.tolist()
adata.obs['topLineages'] = adata.obs.loc[adata.obs.lineage.isin(top),'lineage'].cat.set_categories(top+['other'])
adata.obs.loc[adata.obs.lineage.isin(top)==False,'topLineages'] = 'other'
sc.tl.umap(adata)
sc.pl.umap(adata,color='sample',palette=['#2c9e2f','#046df7', '#d604f7', '#c91212'])
sc.pl.umap(adata,color='topLineages',palette=['#a00101','#f79d02','#00c6c6','#FFFF00','#8ca8ff','#ff7aef','#0800ff','#06a803','#a200b7','#dddddd'])
sc.pl.umap(adata,color=['sample','topLineages'],save='_samples_topLineages.png',wspace=0.3)
#%%
lin = 'AGTGGGTACCGAGGCCATCC'

#sc.pl.stacked_violin(adata[adata.obs.lineage!=lin,:], ['HLA-DRB1','HLA-DRA'],groupby='sample',
#             save='_not_'+lin+'_HLA.png',swap_axes=True,figsize=(4,3))
temp = adata[adata.obs.lineage==lin,:].copy()
sc.pl.stacked_violin(temp, ['HLA-DRB1','HLA-DRA'],groupby='sample',
             save='_HLA_'+lin+'.png',swap_axes=True,figsize=(4,3))

temp2 = adata[:,['HLA-DRB1','HLA-DRA']].copy()
temp2 = temp2[temp2.obs['sample']!='113Aziz',:]
df = pd.concat([temp2.obs['lineage'],
                pd.DataFrame(temp2.X,index=temp2.obs.index,
                             columns=temp2.var_names),],axis=1)
df = df.groupby('lineage').median()
df['mean'] = df.mean(axis=1)
df.sort_values('mean',ascending=False,inplace=True)

for lineage in df.index:
    if temp2.obs.loc[temp2.obs['lineage']==lineage,'sample'].unique().shape[0]>2:
        print(lineage)
        
#%%
sc.tl.umap(adata)

sc.pl.umap(adata,color=['sample','HLA-DRB1','HLA-DRA'],wspace=0.3,
           save='_113_HLA.png')

sc.pl.umap(adata, color=list(adata.obs.drop('lineage',axis=1)),save=True)

sc.tl.rank_genes_groups(adata, 'leiden')
sc.pl.rank_genes_groups(adata,swap_axes=True)

sns.heatmap(pd.crosstab(adata.obs['leiden'], adata.obs['sample'],normalize='all'))

pos = adata[adata.obs['sample']=='doxPos',:].copy()
sc.pp.filter_genes(pos, min_cells=1)
#%% differential expression analysis
adata = sc.read('EB_clustered.h5ad')
top = adata.obs.loc[adata.obs.lineage!='nan','lineage'].value_counts()[:9].index.tolist()
adata.obs['topLineages'] = adata.obs.loc[adata.obs.lineage.isin(top),'lineage'].cat.set_categories(top+['other'])
adata.obs.loc[adata.obs.lineage.isin(top)==False,'topLineages'] = 'other'

for var in ['sample','topLineages','leiden']:
    sc.tl.rank_genes_groups(adata, var)
    sc.pl.rank_genes_groups_matrixplot(adata,n_genes=3,dendrogram=True,
                                       swap_axes=True,standard_scale='var',
                                       save='_'+var+'_markers.png')
    
#%%
sc.pp.filter_cells(adata, min_counts=1)
sc.pl.violin(adata,['n_counts'],groupby='sample',stripplot=False,
             save='_pre-filter_nCounts_by_sample.png',rotation=20)

adata.obs.loc[adata.obs.n_counts<5000,'UMI_category'] = 'lowUMI_cells'
adata.obs.loc[adata.obs.n_counts>=5000,'UMI_category'] = 'highUMI_cells'
sc.tl.rank_genes_groups(adata,'UMI_category')
sc.pl.rank_genes_groups(adata)

#%%








#%%
for samp in adata.obs['sample'].unique():
    print(adata.obs.loc[(adata.obs['sample']==samp)&(adata.obs.lineage!='nan'),'lineage'].value_counts()[:6])
#%%
longTreatLins = adata.obs.loc[(adata.obs['sample'].isin(['107Aziz','113Aziz']))&(adata.obs.lineage!='nan'),'lineage'].unique().tolist()
for lin in longTreatLins:
    if lin not in adata.obs.loc[adata.obs['sample']=='Doxneg','lineage'].tolist():
        print(lin+' not in Doxneg')
    if lin not in adata.obs.loc[adata.obs['sample']=='Doxpos','lineage'].tolist():
        print(lin+' not in Doxpos')

#%%
import pysam
samfile = pysam.AlignmentFile('/stor/work/Brock/10X_data_from_broad/TP0/outs/possorted_genome_bam.bam', 'rb')

for read in samfile.fetch():
    print(read)
    break






