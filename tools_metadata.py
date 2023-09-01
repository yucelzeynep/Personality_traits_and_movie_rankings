#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:42:03 2023

@author: zeynep
"""
import numpy as np
import constants

def get_metadata_ocean(ocean):
    
    """
    Get number of participants
    """
    n_persons = len(ocean)
#        
#    """"
#    Get Ocean stats
#    """    
#    ocean_transpose = np.array(ocean).T.tolist()
#    feat_names = ['o','c','e','a','n']
#            
#    print('-----')
#    print('Ocean stats')
#    print('Name \t min \t max \t mean \t std')
#    for i in range(len(feat_names)):
#        f_min = np.min(ocean_transpose[i])
#        f_max = np.max(ocean_transpose[i])
#        f_mean = np.mean(ocean_transpose[i])
#        f_std = np.std(ocean_transpose[i])
#        
#        print('{0:s}\t{1:0.2f}\t{2:0.2f}\t{3:0.2f}\t{4:0.2f}'.format(\
#              feat_names[i], f_min, f_max, f_mean, f_std))
#        
#                
#    print('-----')
    
    return n_persons
        
def get_metadata_movie_rankings(movie_rankings):    
   
    """
    Get number of movies
    """
    n_movies =  len(movie_rankings[0])

#    """"
#    Get movie_rankings stats
#    """    
#    movie_ranking_transpose = np.array(movie_rankings).T.tolist()
#            
#    print('-----')
#    print('Movie ranking stats')
#    print('Name \t min \t max \t mean \t std')
#    for i in range(len(constants.MOVIE_NAMES)):
#        f_min = np.min(movie_ranking_transpose[i])
#        f_max = np.max(movie_ranking_transpose[i])
#        f_mean = np.mean(movie_ranking_transpose[i])
#        f_std = np.std(movie_ranking_transpose[i])
#        
#        print('{0:s}\t{1:0.2f}\t{2:0.2f}\t{3:0.2f}\t{4:0.2f}'.format(\
#              constants.MOVIE_NAMES[i][0:4], f_min, f_max, f_mean, f_std))    
#        
#    print('-----')    
    
    return n_movies

        
