# -*- coding: utf-8 -*-
"""
@script_author: GIRIER Guillaume
@email : guillaumegirier@gmail.com
@commun_work : GIRIER Guillaume, DESROCHES Mathieu, RODRIGUEZ Serafim

COMMENTS :

If you want to test this code, you can find a toy in the Jupyter Notebook called "High_order_notebook.ipynb".
If you want more information, refers to "high_order.py" comments.

"""

"""
################## Importations ##################
"""


import numpy as np
from scipy.special import digamma



"""
################## Functions ##################
"""

def gaussian_ent_biascorr(N,T):

	"""
	
	Compute the bias corrector for the entropy estimator based on covariance matrix of gaussian.
	
	INPUTS:
	
	N = Number of dimensions
	T = Sample size
	
	OUTPUTS
	
	biascorr = bias corrector value
	"""
	
	values = np.arange(1, N+1)
	
	return 0.5 * ((N * np.log(2/(T-1))) + np.sum(list(map(lambda n : digamma((T-n)/2), values))) )


