#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:56:57 2023

@author: zeynep
"""
import numpy as np
import copy
import sys
import preferences


import constants

def load_raw_data(ot):
    
    if ot == 'NEOFFI':
    
    
        ###########################################################################
        """
        Raw data on ocean
        """
        ocean_raw = \
        [[22, 32, 19, 40, 38],\
        [17, 36, 32, 36, 30],\
        [22, 27, 30, 27, 32],\
        [26, 28, 35, 43, 37],\
        [23, 32, 36, 29, 31],\
        [40, 24, 38, 36, 13],\
        [14, 43, 28, 41, 34],\
        [17, 27, 30, 40, 30],\
        [21, 32, 32, 38, 36],\
        [37, 35, 24, 22, 20],\
        [33, 30, 32, 33, 21],\
        [24, 20, 25, 29, 10],\
        [35, 28, 33, 26, 32],\
        [26, 19, 34, 27, 34],\
        [27, 32, 20, 29, 26],\
        [27, 37, 15, 26, 29],\
        [32, 26, 33, 34, 21]
    ]
        
    elif ot == 'TIPIJ':
    
         ocean_raw = \
        [\
        [4, 3.5, 4, 6, 5.5],\
        [3.5, 6, 6.5, 6, 3],\
        [4, 5.5, 5, 5, 3],\
        [4, 2.5, 3.5, 7, 4],\
        [4, 5, 5, 5.5, 2],\
        [5.5, 2, 4, 5.5, 2],\
        [3.5, 6.5, 5, 6, 2],\
        [3.5, 3.5, 2, 6, 4],\
        [3.5, 3.5, 3.5, 6.5, 6],\
        [5, 6, 3, 6, 1.5],\
        [5, 4.5, 4.5, 6, 2.5],\
        [5, 2.5, 2.5, 4.5, 4],\
        [6, 5, 4.5, 2.5, 3],\
        [4, 1.5, 3.5, 5.5, 3],\
        [5, 5, 3.5, 4.5, 3.5],\
        [6, 6.5, 4, 4, 4],\
        [6, 5, 5, 5.5, 2.5]\
    ]
    else:
        print('Wrong ocean test name')
        sys.exit(0)
        
            
        
    
    ###########################################################################

    """
    Raw data on rankings
    """

    movie_rankings_raw = [\
        [3, 8, 10, 1, 7, 2, 6, 4, 5, 9 ],\
        [6, 3, 9, 2, 8, 5, 10, 7, 1, 4 ],\
        [9, 3, 8, 10, 4, 5, 7, 1, 6, 2 ],\
        [5, 7, 4, 2, 10, 8, 3, 6, 1, 9 ],\
        [2, 3, 6, 4, 8, 10, 7, 9, 1, 5 ],\
        [2, 6, 1, 4, 10, 9, 5, 8, 3, 7 ] ,\
        [2, 5, 9, 6, 1, 7, 8, 3, 10, 4],\
        [7, 2, 1, 5, 6, 4, 8, 10, 9, 3],\
        [1, 10, 6, 3, 9, 2, 8, 4, 5, 7], \
        [9, 8, 3, 2, 7, 4, 10, 1, 6, 5], \
        [3, 10, 9, 1, 6, 5, 7, 4, 8, 2], \
        [5, 9, 4, 1, 2, 3, 6, 10, 8, 7], \
        [5, 7, 8, 6, 4, 3, 9, 2, 10, 1], \
        [5, 1, 6, 2, 8, 4, 9, 3, 10, 7],\
        [1, 7, 5, 2, 4, 6, 10, 3, 9, 8],\
        [7, 6, 10, 4, 5, 1, 8, 3, 9, 2],\
        [6, 2, 9, 3, 10, 8, 4, 5, 1, 7]\
      ]
    
    return ocean_raw, movie_rankings_raw
    

def preprocess_ocean(ocean_raw, normalize_ocean, ot):
    
        
#    print('-----')
#    print('Preprocessing ocean: \t {}'.format(normalize_ocean)) 
        
    if normalize_ocean == 'Raw':
        
        ocean = ocean_raw
    
    elif normalize_ocean == 'Scale_minmax_all':
        
        emin = np.min(ocean_raw)
        emax = np.max(ocean_raw)
        
        ocean = copy.deepcopy(ocean_raw)
        
        for i, row in enumerate(ocean):
            for j, cell in enumerate(row):
                cell_norm = (cell-emin)/(emax-emin)
                ocean[i][j] = cell_norm
                
#        print('Before preprocesing: \t min {0:0.0f}\t max {1:0.0f}'.format(emin, emax))        
#        print('After preprocesing: \t min {0:0.0f}\t max {1:0.0f}'.format(np.min(ocean), np.max(ocean)))        
        
    elif normalize_ocean == 'Scale_minmax_cols':

        
        ocean = copy.deepcopy(ocean_raw)
        ocean_transpose = np.array(ocean_raw).T.tolist()
            
        for i, row in enumerate(ocean_transpose):
            row_min = np.min(row)
            row_max = np.max(row)
            for j, cell in enumerate(row):

                cell_norm = (cell-row_min)/(row_max - row_min)
                ocean[j][i] = cell_norm
        
#        # for debugging
#        print('Col# \t min \t max')
#        ocean_transpose = np.array(ocean).T.tolist()
#        for i, row in enumerate(ocean_transpose):
#            row_min = np.min(row)
#            row_max = np.max(row)
#            print('{} \t {} \t {}'.format(i, row_min, row_max))
                
                
    elif 'Scale_by_minmax_possible':
        ocean = copy.deepcopy(ocean_raw)
        
        for i, row in enumerate(ocean):
            for j, cell in enumerate(row):
                
                if cell == np.nan:
                    continue
                
                if ot == 'NEOFFI':
                    cell_norm = (cell)/(constants.NEOFFI_MAX_POSSIBLE)
                elif ot == 'TIPIJ':
                    cell_norm = (cell)/(constants.TIPIJ_MAX_POSSIBLE)
                else:
                    print('Wrong ocean test name')
                    sys.exit(0)
                    
                ocean[i][j] = cell_norm
#        print('No message')
        
    else:
        print('Unknown option')
        print('Invalid NORMALIZE_OCEAN')
        sys.exit()   
        
#    print('-----')

    return ocean

def preprocess_movie_rankings(movie_rankings_raw):

    """
    Modify such that index starts from 0 (not 1)
    """
    movie_rankings = [0]*len(movie_rankings_raw)
    for t, line in enumerate(movie_rankings_raw):
        movie_rankings[t] = list(map(lambda x: x-1, line))
        
#    print('-----')
#    print('Preprocessing movie rankings: ')
#    print('m \in [1, 10] --> m \in [0,9]')  
#    print('-----')
      
    return movie_rankings