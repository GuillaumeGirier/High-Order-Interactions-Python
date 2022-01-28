# -*- coding: utf-8 -*-
"""
@script_author: GIRIER Guillaume
@email : guillaumegirier@gmail.com
@commun_work : GIRIER Guillaume, DESROCHES Mathieu, RODRIGUEZ Serafim

COMMENTS :

This script code is used by "high_order.py" and use "gaussian_ent_biascorr.py" script.
If you want to test this code, you can find a toy at the end of the script.
If you want more information, refers to "high_order.py" comments.

"""

"""
################## Importations ##################
"""


import cmath
import numpy as np

import gaussian_ent_biascorr as geb

"""
################## Functions ##################
"""

def ent_fun (x,y):

	"""
	Function to compute the entropy of multivariate gaussian distribution where x is
	dimensionality and y is the variables variance of the covariance matrix determinant.
	
	In order to avoid log(0), we replace the returned value as NaN value.
	"""
	
	if (( 2*np.pi*np.exp(1) ) ** x) * y == 0:
	
		return np.nan
	else:
		return 0.5 * cmath.log((( 2*np.pi*np.exp(1) ) ** x) * y)



def reduce_x (x, covmat):

	"""
	This function remove the line and row x in order to obtain a submatrix 
	len(covmat)-1x len(covmat)-1.
	"""

	covmat = np.delete(covmat, x, axis = 0)
	covmat = np.delete(covmat, x, axis = 1)
	
	return covmat





def soinfo_from_covmat (covmat, T):
	
	"""
	
	Computes the 0-information and S-information of gaussian data given their covariance 
	matrix 'covmat'.
	
	INPUTS :
	
	covmat = N x N covariance matrix
	T = lenght data
	
	OUPUTS :
	
	oinfo = O - Information
	sinfo = S - Information of the system with covariance matrix 'covmat'.
	
	"""
	
	covmat = np.array(covmat)
	N = len(covmat)
	emp_det = np.linalg.det(covmat) # Determinant
	single_vars = np.diag(covmat) # Variance of single variables (Diagonal matrix values)
	
	### Bias corrector for N, (N-1) and one gaussian variables :
	
	biascorrN = geb.gaussian_ent_biascorr(N, T)
	biascorrNmin1 = geb.gaussian_ent_biascorr(N-1, T)
	biascorr_1 = geb.gaussian_ent_biascorr(1, T)
	
	### Computing estimated measures for multi-variate gaussian variables :
	
	tc = np.sum(list(map(lambda x : ent_fun(1,x), single_vars)) - biascorr_1) - (ent_fun(N,emp_det) - biascorrN) #Total correlation
	
	Hred = 0
	
	for red in range(1, N+1):
		Hred += ent_fun((N-1), np.linalg.det(reduce_x(red-1, covmat))) - biascorrNmin1
		
	dtc = Hred - (N-1) * (ent_fun(N, emp_det)-biascorrN) # dtc = Dual Total Correlation

	oinfo = tc - dtc
	sinfo = tc + dtc
	
	return oinfo, sinfo

