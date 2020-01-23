#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:24:40 2019

@author: kj22643
"""

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.base import BaseEstimator, TransformerMixin

class PcaExtractor(BaseEstimator, TransformerMixin):
    """Transforms data set into first k principal components."""

    def __init__(self, k=2, center="col", scale="none", small=1e-10):
        self.k = k
        self.center = center
        self.scale = scale
        self.small = small

    def fit(self, X, y=None):
        xhere = pd.DataFrame(X.copy())
        if self.center in ['row', 'both']:
            xRowAvs = xhere.mean(axis=1)
            xhere = xhere.add(-xRowAvs, axis=0)
        if self.center in ['col', 'both']:
            self.colAvs_ = xhere.mean(axis=0)
            xhere = xhere.add(-self.colAvs_, axis=1)
        colSds = xhere.std(axis=0)
        xhere.loc[:, colSds==0] += (self.small *
                                    np.random.randn(xhere.shape[0],
                                                    sum(colSds==0)))
        if self.scale == 'row':
            rowSds = xhere.std(axis=1)
            xhere = xhere.divide(rowSds, axis=0)
        elif self.scale == 'col':
            self.colSds_ = xhere.std(axis=0)
            xhere = xhere.divide(self.colSds_, axis=1)
        xsvd = np.linalg.svd(xhere, full_matrices=False)
        self.v_ = np.transpose(xsvd[2])[:, 0:self.k]
        return self

    def transform(self, X):
        xhere = pd.DataFrame(X.copy())
        if self.center in ['row', 'both']:
            xRowAvs = xhere.mean(axis=1)
            xhere = xhere.add(-xRowAvs, axis=0)
        if self.center in ['col', 'both']:
            xhere = xhere.add(-self.colAvs_, axis=1)
        if self.scale == 'row':
            rowSds = xhere.std(axis=1)
            xhere = xhere.divide(rowSds, axis=0)
        elif self.scale == 'col':
            xhere = xhere.divide(self.colSds_, axis=1)
        return np.dot(xhere, self.v_)
