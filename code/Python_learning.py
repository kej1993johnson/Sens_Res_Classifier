#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:38:22 2019

@author: kj22643
"""

%reset

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
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
os.chdir('/Users/kj22643/Documents/Documents/Python_Learning/code')


# Purpose of this script is to test out how to do some forward model evaluation
# and subsequent parameter estimation in python


#%% Forward logistic growth model
# time points
tvec = np.linspace(0,400, 26)
#  define a model function
def logisticmod (N,t,params):
    g = params[0]
    K = params[1]
    dNdt=g*N*(1-N/K)
    return dNdt
# set params
g= 0.02;
K = 5e4;
p = [g,K];
N0 = 2000;

# solve the ODE for the set growth rate and carrying capacity
Nmod = odeint(logisticmod, N0, tvec, args = (p,))
p[0] = 0.5*g
Nmod2 = odeint(logisticmod, N0, tvec, args = (p,))

# plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
 
plt.plot(tvec,Nmod, 'r-', linewidth=2, label= 'g=0.02')
plt.plot(tvec,Nmod2, 'b-', linewidth=2, label= 'g=0.01')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.show()
#%% Simulate more complex functions with multiple "species"
# Start with Didi's model from her qual.
def interlogmod (N,t,params):
    x1 = N[0];
    x2 = N[1];
    g1 = params[0] # growth rate of species 1
    g2 = params[1] # growth rate of species 2
    alpha12 = params[2] # effect of species 2 on species 1
    alpha21 = params[3] # effect of species 1 on species 2
    K = params[4]
    dN1dt=g1*x1*(1-((x1+alpha12*x2)/K))
    dN2dt = g2*x2*(1-((x1+alpha21*x2)/K))
    dNdt = [dN1dt,dN2dt]
    return dNdt

# set params
g1 = 0.02;
g2 = 0.015;
alpha12 = 0.1;
alpha21 = -0.2;
p =[g1, g2, alpha12, alpha21, K];
N0 = [20, 20]

Nmod = odeint(interlogmod, N0, tvec, args = (p,))
Ntot = Nmod[:,0] + Nmod[:,1];

# plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tvec,Nmod[:,0], 'r-', linewidth=2, label= 'x1, g=0.02')
plt.plot(tvec,Nmod[:,1], 'b-', linewidth=2, label= 'x2, g=0.015')
plt.plot(tvec, Ntot, 'k-', linewidth = 3, label = 'x1+x2')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.show()
#%% Use the function to simulate and compare the expected results observed in 
# each population and the total for a few different scenarios

# Case 1: No interactions
g1 = 0.02;
g2 = 0.015;
alpha12 = 0;
alpha21 = 0;
p =[g1, g2, alpha12, alpha21, K];
N0 = [2000, 2000]

N1 = odeint(interlogmod, N0, tvec, args = (p,))
N1tot= N1[:,0] + N1[:,1];

# plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tvec,N1[:,0], 'r-', linewidth=2, label= 'x1, g=0.02')
plt.plot(tvec,N1[:,1], 'b-', linewidth=2, label= 'x2, g=0.015')
plt.plot(tvec, N1tot, 'k-', linewidth = 3, label = 'x1+x2')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.title('Case 1: No interaction')
plt.show()

# Case 2: Producers (slower growers) help consumers grow faster
g1 = 0.02;
g2 = 0.015;
alpha12 = -0.1;
alpha21 = 0;
p =[g1, g2, alpha12, alpha21, K];
N0 = [2000, 2000]

N2 = odeint(interlogmod, N0, tvec, args = (p,))
N2tot= N2[:,0] + N2[:,1];

# plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tvec,N2[:,0], 'r-', linewidth=2, label= 'Consumers, g=0.02')
plt.plot(tvec,N2[:,1], 'b-', linewidth=2, label= 'Producers, g=0.015')
plt.plot(tvec, N2tot, 'k-', linewidth = 3, label = 'Total, alpha12 <0')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.title('Case 2: Producers aid consumers')
plt.show()

# Case 3: Mutual benefit- both cooperate to promote growth
g1 = 0.02;
g2 = 0.015;
alpha12 = -0.1;
alpha21 = -0.1;
p =[g1, g2, alpha12, alpha21, K];
N0 = [2000, 2000]

N3 = odeint(interlogmod, N0, tvec, args = (p,))
N3tot= N3[:,0] + N3[:,1];

# plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tvec,N3[:,0], 'r-', linewidth=2, label= 'x1, g=0.02')
plt.plot(tvec,N3[:,1], 'b-', linewidth=2, label= 'x2, g=0.015')
plt.plot(tvec, N3tot, 'k-', linewidth = 3, label = 'Total, alpha12 \& alpha21 $<$0')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.title('Case 3: Mutual cooperative growth')
plt.show()

# Case 4: Explicit competition between both species
g1 = 0.02;
g2 = 0.015;
alpha12 = 0.1;
alpha21 = 0.1;
p =[g1, g2, alpha12, alpha21, K];
N0 = [2000, 2000]

N4 = odeint(interlogmod, N0, tvec, args = (p,))
N4tot= N4[:,0] + N4[:,1];

# plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tvec,N4[:,0], 'r-', linewidth=2, label= 'x1, g=0.02')
plt.plot(tvec,N4[:,1], 'b-', linewidth=2, label= 'x2, g=0.015')
plt.plot(tvec, N4tot, 'k-', linewidth = 3, label = 'Total, alpha12 \& alpha21$>$0')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.title('Case 4: Competition')
plt.show()

# Are the bulk population dynamics noticeably different? 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tvec, N1tot, 'r-', linewidth = 3, label = 'Case 1: no interaction')
plt.plot(tvec, N2tot, 'c-', linewidth = 3, label = 'Case 2: consumers benefit')
plt.plot(tvec, N3tot, 'g-', linewidth = 3, label = 'Case 3: cooperative growth')
plt.plot(tvec, N4tot, 'b-', linewidth = 3, label = 'Case 4: competition')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.title('Case 4: Competition')
plt.show()

# Looks to be not something we can very clearly see a difference in the bulk behavior
# Perhaps looking at the individual growth dynamics at different initial proportions
# would enable the identifiability of these different interaction relationships?
# Not really sure but this is definitely something that we will need to explore further!

#%% How do we perform parameter estimation of the interaction logistic growth model 
# with the data we have? 
path = '/Users/kj22643/Documents/Documents/Python_Learning/data'
data = pd.read_excel(r'/Users/kj22643/Documents/Documents/Python_Learning/data/ex_data.xlsx')
print(data)

# Plot the raw data
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(data.time, data.AA161_B2, 'b.', linewidth = 3, label = 'AA161 well B2')
plt.plot(data.time, data.AA161_B3, 'c.', linewidth = 3, label = 'AA161 well B3')
plt.plot(data.time, data.AA170_B10, 'm.', linewidth = 3, label = 'AA170 well B10')
plt.plot(data.time, data.AA170_B11, 'r.', linewidth = 3, label = 'AA170 well B11')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.legend()
plt.title('Growth data of isolated lineages')
plt.show()
#%% Fit indidual wells first to the simple logistic model- set K and fit for g
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
#import pymc3 as pm3
#import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


# Put data into a smaller data frame with jsut the first well
df = pd.DataFrame({'N': data.iloc[:,1], 't':data.time})
# plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(df.t, df.N, '.', linewidth=2)
plt.xlabel('time')
plt.ylabel('N(t)')
plt.title('AA161 well B2 growth data')
plt.show()
#%% Start by making a objective function- simplest one is just minimize the sum of the squared errors

# sum-of-squared error
def objfunlogistic(params, t, N):
    g, K = params[0], params[1] # inputs are our guesses at parameters
    N0 = N[0]
    Nmodel = odeint(logisticmod, N0, t, args = (params,))
    Nmod = np.array(Nmodel)

    # find the sum of the squared error between the data and the model
    sumsqerr = np.sum((Nmodel-N)**2)
    return (sumsqerr)

# Now that we have a cost function, let's initialize and minimzie it:
# Let's start with some random coeffieicent guesses and optimize
   
guess = np.array([0.005, 100]) # represents our initial guess for g (per day) and K

pbest = minimize(objfunlogistic,(0.003, 100), args = (df.t.values, df.N.values), method = 'BFGS')
options = ({'disp':True})
# Does not appear to be minimizing... 
Nbest = odeint(logisticmod, df.N[0], df.t, args = (pbest.x,))
Ntest = odeint(logisticmod, df.N[0], df.t, args = (guess,))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(df.t, df.N, '.', linewidth=2, label='data')
plt.plot(df.t, Ntest, '-', linewidth=2, label = 'guess')
plt.plot(df.t, Nbest, '-', linewidth=2, label = 'model fit')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.title('AA161 well B2 growth data')
plt.legend()
plt.show()
#%% Other objective functions- how about Maximum Likelihood?
# To do this, we will need to have the mean and standard deviation of the data
# Instead of fitting just one well individually, we will use the mean and 
# spread in the data over the replicate wells.
meanNAA161 = np.mean([data.AA161_B2, data.AA161_B3, data.AA161_C2, data.AA161_C3, data.AA161_D2,
                      data.AA161_D3, data.AA161_E2, data.AA161_E3, data.AA161_F2,
                      data.AA161_F3, data.AA161_G2, data.AA161_G2, data.AA161_G3], axis=0)
stdAA161 = np.std([data.AA161_B2, data.AA161_B3, data.AA161_C2, data.AA161_C3, data.AA161_D2,
                      data.AA161_D3, data.AA161_E2, data.AA161_E3, data.AA161_F2,
                      data.AA161_F3, data.AA161_G2, data.AA161_G2, data.AA161_G3], axis=0)
df['N_AA161'] = meanNAA161
df['SD_AA161'] = stdAA161
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.plot(df.t, data.AA161_B3,'.',linewidth=1)
#plt.plot(df.t, data.AA161_C2,'.',linewidth=1)
#plt.plot(df.t, data.AA161_C3,'.',linewidth=1)
#plt.plot(df.t, data.AA161_E2,'.',linewidth=1)
#plt.plot(df.t, data.AA161_E3,'.',linewidth=1)
#plt.plot(df.t, data.AA161_F2,'.',linewidth=1)
#plt.plot(df.t, data.AA161_F3,'.',linewidth=1)
#plt.plot(df.t, data.AA161_G2,'.',linewidth=1)
#plt.plot(df.t, data.AA161_G3,'.',linewidth=1)
plt.plot(df.t, df.N_AA161, 'b-', linewidth=4, label='mean data AA161')
plt.plot(df.t, meanNAA161+1.96*stdAA161, 'b-', linewidth=1, label = 'CI')
plt.plot(df.t, meanNAA161-1.96*stdAA161, 'b-', linewidth=1)
plt.xlabel('time')
plt.ylabel('N(t)')
plt.title('AA161 growth data')
plt.legend()
plt.show()

# Use the mean cell number and standard deviation in cell number at every time point to perform ML parameter estimation

# define likelihood function
def MLElogistic(params, Ndata, t, sd):
    g, K = params[0], params[1] # inputs are our guesses at parameters
    N0 = Ndata[0]
    Nmodel = odeint(logisticmod, N0,t, args = (params,))
    yhat = np.array(Nmodel)
    
    # Flip the Bayesian question: compute PDF of observed values normally 
    # distributed around mean (yhat- your model predicted values) with a 
    # standard deviation of sd
    
    negLL = -np.sum(stats.norm.logpdf(Ndata, loc=yhat, scale = sd))
    
    # return negative LL
    return(negLL)

results = minimize(MLElogistic, guess, method = 'Nelder-Mead', args = (df.N_AA161.values, df.t.values, df.SD_AA161.values))
options = ({'disp':True})

Nbest = odeint(logisticmod, df.N_AA161[0], df.t, args = (results.x,))
Nguess = odeint(logisticmod, df.N_AA161[0], df.t, args = (guess,))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(df.t, df.N_AA161, 'b-', linewidth=4, label='mean data AA161')
plt.plot(df.t, meanNAA161+1.96*stdAA161, 'b-', linewidth=1, label = 'CI')
plt.plot(df.t, meanNAA161-1.96*stdAA161, 'b-', linewidth=1)
plt.plot(df.t, Nbest, 'r-', linewidth=1, label = 'model fit')
plt.plot(df.t, Nguess, 'g-', linewidth=1, label = 'guess')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.title('AA161 growth data')
plt.legend()
plt.show()