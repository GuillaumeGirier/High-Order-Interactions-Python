# -*- coding: utf-8 -*-
"""
@script_author: GIRIER Guillaume
@email : guillaumegirier@gmail.com
@commun_work : GIRIER Guillaume, DESROCHES Mathieu, RODRIGUEZ Serafim

COMMENTS :

This script code is used by "high_order.py".
If you want to test this code, you can find a toy in the Jupyter Notebook called "High_order_notebook.ipynb".
If you want more information, refers to "high_order.py" comments.

"""

"""
################## Importations ##################
"""

import numpy as np
from scipy.stats import norm


"""
################## Functions ##################
"""


def data2gaussian (data) :

	"""
	Transforms 'data' to Gaussian with mean = 0 and sd = 1 using empirical copulas.
	
	INPUTS :
	
	data = T samples x N variables matrix
	
	OUTPUTS :
	
	gaussian_data = T samples x N variables matrix with the gaussian copula transformed data.
	covmat = N x N covariance matrix of gaussian copula transformed data.
	
	"""
	
	T = len(data)
	sort_index = np.argsort(data, axis = 0) # Sort the data and keep the indexes.
	copdata = np.argsort(sort_index, axis = 0) # Sorting sorting indexes
	copdata += 1 # To avoid 0 because of the python indexation
	copdata = copdata/ (T+1) # Normalization.
	gaussian_data = norm.ppf(copdata, 0, 1) # PPF : Probability Density Function  => Gaussian data
	gaussian_data[~np.isfinite(gaussian_data)] = 0 # Removing -Inf
	cov_mat = np.dot(np.transpose(gaussian_data), gaussian_data ) / (T-1) # Covariance matrix
	
	return gaussian_data, cov_mat

