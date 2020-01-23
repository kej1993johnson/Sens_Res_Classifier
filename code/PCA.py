#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:28:13 2019

@author: kj22643
"""

import matplotlib.pyplot as plt

import MaclearnUtilities
from MaclearnUtilities import safeFactorize, ggpca

plt.ion()

import RestrictedData
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys



## -----------------------------------------------------------------
plt.close()
ggpca(xnorms['shen'], annots['shen']['System'],
      rlab=True,
      cshow=0, colscale=['firebrick', 'goldenrod', 'lightseagreen',
                          'darkorchid', 'darkslategray', 'dodgerblue'])


plt.close()
ggpca(xnorms['shen'], annots['shen']['System'],
      rlab=True, clab=True,
      cshow=25, clightalpha=0.1,
      colscale=['firebrick', 'goldenrod', 'lightseagreen',
                'darkorchid', 'darkslategray', 'dodgerblue', 'gray'])


plt.close()
ggpca(xnorms['patel'], ys['patel'],
      rlab=False, clab=True, cshow=10, clightalpha=0)
© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
