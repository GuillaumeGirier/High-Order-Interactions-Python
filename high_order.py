# -*- coding: utf-8 -*-
"""
@script_author: GIRIER Guillaume
@email : guillaumegirier@gmail.com
@commun_work : GIRIER Guillaume, DESROCHES Mathieu, RODRIGUEZ Serafim


PURPOSE :

This main script "high_order.py" computes the High Order Interaction.
It uses "data2gaussian.py" and "soinfo_from_covmat.py" functions (which use "gaussian_ent_biascorr.py" function).

The inputs of the method are:

	"data" : has to be (NxT) structure, where :
		- N is the number of module that you want to study in your data,
		- T is the time serie associated to the modules studied.
	" n " : n-plet or the number of interaction between the modules that you want to consider for the simulation. "n" has to be greater or equal to 2.
	
The outputs are :

	"Red" : (1, N) structure, it correspond to the redundancy values for a patient per module.
	"Syn" : (1, N) structure, it correspond to the synergy values for a patient per module.
	"Oinfo" : O-Information for all the n-plets.
	"Sinfo" : S-Information for all the n-plets.

This code is a translation from Matlab to Python of this github : https://github.com/brincolab/High-Order-interactions
The associated paper : https://www.biorxiv.org/content/biorxiv/early/2020/03/18/2020.03.17.995886.full.pdf

"""

"""
################## Importations ##################
"""


import numpy as np

from scipy.stats import norm
from itertools import combinations

import soinfo_from_covmat as sfc
import data2gaussian as d2g



"""
################## Functions ##################
"""

def high_order (data, n):

	""" 
	
	Function to compute S-Information, O-Information, and characterize the High Order	
	interactions among n variables governed by Redundancy or Synergy.
	
	INPUTS :
	
	data = Matrix with dimensionality (N,T), where N is the number of brain regions or 
	modules, and T is the number of samples.
	n = number of interactions or n-plets.
	
	OUTPUTS :
	Red = Matrix with dimension (1, Modules), with the redundancy values per patient 
	and per module.
	Syn = Matrix with dimension (1, Modules), with synergy values per patient 
	and per module.
	Oinfo = O-Information for all the n-plets.
	Sinfo = S-Information for all the n-plets.
	
	"""

	### INITIALISATION :
	
	Modules = len(data)
	Red = np.zeros(Modules) 
	Syn = np.zeros(Modules)
	
	vector = np.arange(0, Modules)
	nplets = []
	
		
	Oinfo = []
	Sinfo = []
	
	
	### N-PLETS CALCULATION :
	
	for x in combinations(vector, n): # n-tuples without repetition over 20 modules
		nplets.append(x)
	nplets = np.array(nplets)

	### DATA NORMALISATION :

	mean = np.mean(data, axis = 1)
	mean = mean.reshape(-1,1)
	
	dataNorm = data - mean

	gaussian_data, cov_mat = d2g.data2gaussian (np.transpose(dataNorm)) # Transformation to Copulas and Covariance Matrix Estimation
	
	i = 0
	
	### OINFO AND SINFO COMPUTATION :	

	Info = []
	Info.append(list(map(lambda x : sfc.soinfo_from_covmat(cov_mat[np.ix_(x,x)], len(dataNorm[0])), nplets)))
	Info = np.transpose(Info)
	Oinfo = np.array(Info[0])
	Sinfo = np.array(Info[1])
	
	### REDUNDANCY AND SYNERGY COMPUTATION :
	
	"""
	Here, we want to verify in each nplet if a module exist : when it is the case we compute
	the associated Redundancy and Synergy, according to the Oinfo value of these nplets.
	"""
	Values = []
	
	for module in range(Modules):
		Values, cols = np.where(nplets == module)
		
		Oinfo_module_pos = []
		Oinfo_module_neg = []
		
		for i in range(len(Values)):

			if Oinfo[Values[i]].real > 0 :
				Oinfo_module_pos.append(Oinfo[Values[i]].real)
			if Oinfo[Values[i]].real < 0 :
				Oinfo_module_neg.append(Oinfo[Values[i]].real)

		Red[module] = np.mean(np.array(Oinfo_module_pos))
		Syn[module] = np.mean(np.absolute(np.array(Oinfo_module_neg)))
		
		Values = []
		
	Red = Red.reshape(-1,1)
	Syn = Syn.reshape(-1,1)
	
	np.nan_to_num(Syn)
		
	
	return Red, Syn, Oinfo, Sinfo
	
