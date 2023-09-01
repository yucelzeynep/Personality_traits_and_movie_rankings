#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:05:31 2023

@author: zeynep
"""
from importlib import reload
import tools_optim 
reload(tools_optim)
###############################################################################
OCEAN_TEST_OPTIONS = ['NEOFFI', 'TIPIJ']
###############################################################################

"""
Normalize personalities or not. Options are:
    Raw: 
        No normalization
    Scale_minmax_all: 
        Scale all personality traits between 0 and 1
    Scale_minmax_cols: 
        Scale each trait separately between 0 and 1
    Scale_by_minmax_possible:
        the points are limited in range 0-48. Divide each number in personlities by 48
"""
#NORMALIZE_OCEAN = 'Scale_minmax_cols'
#NORMALIZE_OCEAN = 'Scale_by_minmax_possible'

NORMALIZE_OCEAN_OPTIONS = ['Raw', \
                           'Scale_minmax_all', \
                           'Scale_minmax_cols', \
                           'Scale_by_minmax_possible']
###############################################################################

"""
if true, random weights 
if false, euqal weights (0.2)
"""
#INIT_GRAPH_WEIGHTS_RANDOM = False

INIT_GRAPH_WEIGHTS_RANDOM_OPTIONS = [True, False]

###############################################################################
"""
The one that works is Powell
Powell

These ones give bad results
Nelder-Mead
CG
BFGS
L-BFGS-B
TNC
SLSQP

Others do not work but because I need to give Jacobian etc

"""
#MY_OPTIMIZATION_METHOD = "TNC"
        
MY_OPTIMIZATION_METHOD_OPTIONS = ['Nelder-Mead',\
                                  'CG',\
                                  'BFGS',\
                                  'L-BFGS-B',\
                                  'TNC',\
                                  'SLSQP']
###############################################################################
"""
Constraints on weights
"""
#CONS = [{'type':'eq', 'fun': tools.constraint}]
#CONS = [{'type':'eq', 'fun': tools.constraint_real}]
CONS = [{'type':'eq', 'fun': tools_optim.constraint}, \
        {'type':'eq', 'fun': tools_optim.constraint_real}]


"""
Bounds of weights
"""
BOUNDS = ((0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0))

###############################################################################
N_TOP = 5
N_BOT = 5

N_RANDOM_RECOMMENDATIONS = 100
