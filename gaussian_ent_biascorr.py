# -*- coding: utf-8 -*-
"""
@script_author: GIRIER Guillaume
@email : guillaumegirier@gmail.com
@commun_work : GIRIER Guillaume, DESROCHES Mathieu, RODRIGUEZ Serafim

LICENSE :

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 3

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


