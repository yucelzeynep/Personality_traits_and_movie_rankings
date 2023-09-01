#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:30:30 2023

@author: zeynep
"""
from importlib import reload

import numpy as np
import tools_optim 
reload(tools_optim)
import tools_preprocess 
import tools_metadata
import copy
import time

import pickle
import preferences

from numpy import random

    
if __name__ == "__main__": 
    start_time = time.time()

    weights_opt_all = {}
    for ot in preferences.OCEAN_TEST_OPTIONS:
        weights_opt_all[ot] = {}
        for no in preferences.NORMALIZE_OCEAN_OPTIONS:
            weights_opt_all[ot][no] = {}
            for igw in preferences.INIT_GRAPH_WEIGHTS_RANDOM_OPTIONS:
                weights_opt_all[ot][no][igw] = {}
                for om in preferences.MY_OPTIMIZATION_METHOD_OPTIONS:
                    weights_opt_all[ot][no][igw][om] = []
                
    ###########################################################################
    counter = 0
    for ot in preferences.OCEAN_TEST_OPTIONS:
        ocean_raw, movie_rankings_raw = tools_preprocess.load_raw_data(ot)
        for no in preferences.NORMALIZE_OCEAN_OPTIONS:
            for igw in preferences.INIT_GRAPH_WEIGHTS_RANDOM_OPTIONS:
                for om in preferences.MY_OPTIMIZATION_METHOD_OPTIONS:
                    
                    counter += 1
                    print('{} {} {} {} {}'.format(\
                          counter, ot, no, igw, om))
                    ocean = tools_preprocess.preprocess_ocean(ocean_raw, no, ot)
                
                    movie_rankings = tools_preprocess.preprocess_movie_rankings(movie_rankings_raw)
            
                    ###########################################################################    
                    
                    
                    """
                    Check and get meta data
                    """
                    if len(ocean) != len(movie_rankings):
                        print('Number of persons do not match')
                        print('Problem with lengths of data on ocean and movie rankings')
                        exit(0)
                    else:
                        n_persons = tools_metadata.get_metadata_ocean(ocean)
                        n_movies = tools_metadata.get_metadata_movie_rankings(movie_rankings)
                        
#                        print('n_persons \t {}'.format(n_persons))
#                        print('n_movies \t {}'.format(n_movies))
                
                
                    ###########################################################################
                    """
                    Initialize graph model, i.e. weights 
                    
                    Weights:
                            It can be random or equal weights.
                            Random weights is dangerous
                    
                    """
                    if igw:
                        weights0 = random.rand(5)
                    else:            
                        weights0 = [0.2, 0.2, 0.2, 0.2, 0.2]   
                        
                    """
                    Initialize delta of recommended  movies for each person
                    """
                    deltas_recommended_by_graph_model = [np.nan] * n_persons
                    ###########################################################################
                    
                
                    """
                    Leave 1 person for test and take others for train 
                    Optimize the weights of ocean for the graph model
                    """
                    idx_all = list(np.arange(0, n_persons))
                    
                    for idx_person_test in idx_all:
                        
                        print('Training for ID {}'.format(idx_person_test), end =' ')
                            
                        idx_person_train = copy.deepcopy(idx_all)
                        idx_person_train.remove(idx_person_test)
                               
                        ocean_train = [ocean[j] for j in idx_person_train]
                        movie_rankings_train = [movie_rankings[j] for j in idx_person_train]
                        
                        ocean_test = ocean[idx_person_test]
                        movie_rankings_test = movie_rankings[idx_person_test]
                                
                        temp_weights_opt = tools_optim.train_graph_model(om,\
                                                                         ocean_train, \
                                                                         movie_rankings_train, \
                                                                         weights0)
                        
                        weights_opt_all[ot][no][igw][om] = temp_weights_opt
                        
                        filename = 'weights_opt_all.pkl'
                        with open(filename, 'wb') as handle:
                            pickle.dump(weights_opt_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)
    # Time elapsed  24132.83 sec                