#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:30:30 2023

@author: zeynep
"""

import numpy as np
#import random
import sys
from collections import Counter
from scipy.optimize import minimize
import tools_simplex as tools

from numpy import random
import copy

# inly for unreal data
#ocean_trait_minval = 0
#ocean_trait_maxval = 5

"""
Normalize personalities or not. Options are:
    Raw: 
        No normalization
    Scale_all_entries_0_to_1: 
        Scale all personality traits between 0 and 1
    Scale_each_column_max_to_1: 
        Scale each trait separately between 0 and 1
    Scale_by_minmax_possible:
        the points are limited in range 0-48. Divide each number in personlities by 48
"""
#NORMALIZE_PERSONALITIES = 'Scale_each_column_max_to_1'
NORMALIZE_PERSONALITIES = 'Raw'

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
MY_OPTIMIZATION_METHOD = "TNC"
        
N_TOP = 3
N_BOT = 2

"""
Constraints on weights
"""
#CONS = [{'type':'eq', 'fun': tools.constraint}]
#CONS = [{'type':'eq', 'fun': tools.constraint_real}]
CONS = [{'type':'eq', 'fun': tools.constraint}, {'type':'eq', 'fun': tools.constraint_real}]
#CONS = []

"""
Bounds of weights
"""
BOUNDS = ((0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0))

N_RANDOM_ESTIMATIONS = 10000

PERSONALITY_TRAIT_MAX = 48
PERSONALITY_TRAIT_MIN = 0

    
if __name__ == "__main__": 

    ###########################################################################
    """
    Real data on personalities.
    
    """
    personalities_notnormalized = \
    [[22, 32, 19, 40, 38],\
    [17, 36, 32, 36, 30],\
    [22, 27, 30, 27, 32],\
    [26, 28, 35, 43, 37],\
    [23, 32, 36, 29, 31],\
    [40, 24, 38, 36, 13],
    [14, 43, 28, 41, 34],
    [17, 27, 30, 40, 30],
    [21, 32, 32, 38, 36],
    [37, 35, 24, 22, 20],
    [33, 30, 32, 33, 21],
    [24, 20, 25, 29, 10],
    [35, 28, 33, 26, 32],
    [26, 19, 34, 27, 34],
    [27, 32, 20, 29, 26],
    [27, 37, 15, 26, 29],
    [32, 26, 33, 34, 21]
]
    
        
    if NORMALIZE_PERSONALITIES == 'Raw':
        
        personalities = personalities_notnormalized
    
    elif NORMALIZE_PERSONALITIES == 'Scale_all_entries_0_to_1':
        
        personalities = personalities_notnormalized
    
        emin = np.min(personalities_notnormalized)
        emax = np.max(personalities_notnormalized)

        print("emin == ")
        print(emin)
        print()
        print("emax == ")
        print(emax)
        
        personalities = copy.deepcopy(personalities_notnormalized)
        
        dmin = np.nanmin(personalities_notnormalized)
        dmax = np.nanmax(personalities_notnormalized)

        print("dmin == ")
        print(dmin)
        print()
        print("dmax == ")
        print(dmax)
            
        for i, row in enumerate(personalities):
            for j, cell in enumerate(row):
                if cell == np.nan:
                    continue
                cell_norm = (cell-dmin)/(dmax-dmin)
                personalities[i][j] = cell_norm
        
    elif NORMALIZE_PERSONALITIES == 'Scale_each_column_max_to_1':
        
        personalities = copy.deepcopy(personalities_notnormalized)
        
        col_mins = np.min(personalities_notnormalized,0)
        col_maxs = np.max(personalities_notnormalized,0)
            
        for i, row in enumerate(personalities):
            for j, cell in enumerate(row):
                if cell == np.nan:
                    continue
                cell_norm = (cell)/(col_maxs[j])
                personalities[i][j] = cell_norm
                
    elif 'Scale_by_minmax_possible':
        personalities = copy.deepcopy(personalities_notnormalized)
        
        for i, row in enumerate(personalities):
            for j, cell in enumerate(row):
                if cell == np.nan:
                    continue
                cell_norm = (cell)/(48)
                personalities[i][j] = cell_norm
        
    else:
        print('Unknown option')
        print('Invalid NORMALIZE_PERSONALITIES')
        sys.exit()                 
    
    ###########################################################################

    """
    Real data on rankings
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
        [6, 2, 9, 3, 10, 8, 4, 5, 1, 7]]
    
    ###########################################################################

    """
    Get number of participants and number of movies
    """
    n_persons = 0
    if len(personalities) != len(movie_rankings_raw):
        print('Number of persons do not match')
        print('Problem with lengths of data on personalities and movie rankings')
        exit(0)
    else:
        n_persons = len(personalities)



    """
    modify such that index starts from 0 (not 1)
    """
    movie_rankings = [0] * n_persons
    for t, line in enumerate(movie_rankings_raw):
        movie_rankings[t] = list(map(lambda x: x -1, line))

    """
    Get number of movies
    """
    n_movies =  len(movie_rankings[0])


    ###########################################################################
    """
    Initialize graph model, i.e. weights 
    
    Weights:
            It can be random or equal weights.
            Random weights is dangerous
    
    """
    weights0 = [0.2, 0.2, 0.2, 0.2, 0.2]   
#    weights0 = random.rand(5)

    weights_opt = [[0]*5]*n_persons
    

    
    """
    Initialize delta of recommended  movies for each person
    """
    deltas_recommended_by_graph_model = [0] * n_persons
    ###########################################################################
        
    """
    Leave 1 person for test and take others for train 
    Optimize the weights of personalities for the graph model
    """
    
    idx_all = list(np.arange(0, n_persons))
    
    for idx_person_test in idx_all:
                
        idx_person_train = copy.deepcopy(idx_all)
        idx_person_train.remove(idx_person_test)
               
        personalities_train = [personalities[j] for j in idx_person_train]
        movie_rankings_train = [movie_rankings[j] for j in idx_person_train]
        
        personality_test = personalities[idx_person_test]
        movie_rankings_test = movie_rankings[idx_person_test]
        
        """
        Train the graph model with the the training sets of personalities and 
        movie_rankings
        """
    
        if MY_OPTIMIZATION_METHOD == "Powell" or \
        MY_OPTIMIZATION_METHOD == 'Nelder-Mead' or\
        MY_OPTIMIZATION_METHOD == 'CG' or\
        MY_OPTIMIZATION_METHOD == 'BFGS':
            
            res = minimize(tools.get_cost, \
               weights0, \
               method = MY_OPTIMIZATION_METHOD,\
               args=(personalities_train, movie_rankings_train),\
                options={'maxiter': 2})

        elif MY_OPTIMIZATION_METHOD == "L-BFGS-B" or\
        MY_OPTIMIZATION_METHOD == "TNC":
            
                res = minimize(tools.get_cost, \
                   weights0, \
                   method = MY_OPTIMIZATION_METHOD,\
                   args=(personalities_train, movie_rankings_train),\
                   bounds=BOUNDS)
        else:

               res = minimize(tools.get_cost, \
                           weights0, \
                           method = MY_OPTIMIZATION_METHOD,\
                           args=(personalities_train, movie_rankings_train),\
                           bounds=BOUNDS,\
                           constraints=CONS)
            
        weights_opt[idx_person_test] = res.x         # weights that minimize obj fun
        
        ###########################################################################
        """
        Observe training performance and weights
        
        The initial cost has to be higher than the final (optimal) cost.
        
        Optimization method is default.
        It can be changed, but it does not make much difference 
        """
        cost_init = tools.get_cost(weights0,\
                                   personalities_train,\
                                   movie_rankings_train) 
            
        cost_opt = tools.get_cost(weights_opt[idx_person_test],\
                                  personalities_train,\
                                  movie_rankings_train) 
        

        
        if cost_init == cost_opt:
            """            
            print('cost_init \t{0:2.2f}'.format(cost_init))
            print('cost_opt \t{0:2.2f}'.format(cost_opt))
            
            print('')
            
            formatted_list = [ '%.4f' % elem for elem in weights0 ]
            print('weights0 \t{}'.format( formatted_list))
            
            formatted_list = [ '%.4f' % elem for elem in weights_opt ]
            print('weights_opt \t{}'.format( formatted_list))

            print('')
            
            print('Weights cannot be updated')
            sys.exit("Weights cannot be updated")
            """
        
        
        """
        Get pairwise personality distances with these optimal weights 
        
        Recommend to the test person the same movies listed by the train person 
        with closest personality (smallest delta_personality)
        
        Compute how good is this recommendation in terms of 
        delta_movie_recommended
        """
        delta_movie_rankings_pairwise = \
        tools. get_delta_movie_rankings_pairwise(movie_rankings)
        
    
        # Careful that here do not go back to train list idx
        # because the above is computed for all personalities (test and train)
        # but when test==train, delta is defined as np.nan
        idx_person_closest = np.nanargmin(delta_movie_rankings_pairwise[idx_person_test])
        
        personality_estimated = \
        personalities[idx_person_closest]
        
        
        
        delta_personality_estimated = tools.get_delta_personality(\
                                            personality_estimated, \
                                            personality_test,\
                                            weights_opt[idx_person_test]) 
        """ 
        delta_personality_estimated = tools.get_delta_euclid(\
                                            personality_estimated, \
                                            personality_test)
        """
                
        deltas_recommended_by_graph_model[idx_person_test] = delta_personality_estimated
        
     
        
    print('########################')   
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################    
    ###########################################################################
    """
    For each test person generate many random recommendations by shuffling 
    the array between 0 and 9
    (N_RANDOM_RECOMMENDATIONS)
    
    Compare each random recommendation with the one from the graph model and 
    decide whether graph model is better, worse or equal. (Win-Loss-tie)

    """    

    for idx_person_test in idx_all:
        
#        if idx_person_test != 3:
#            continue
##        
        
        # random recommendation to idx_person_test
        win_loss_tie = []
        personality_test = personalities[idx_person_test]
        
        for i in range(N_RANDOM_ESTIMATIONS):
            
            random_estimation = [random.randint(0, 48) for i in range(5) ]


                        
            delta_random_estimation = tools.get_delta_personality(\
                                    personality_test, \
                                    random_estimation,\
                                    weights_opt[idx_person_test])
            """
            delta_random_estimation = tools.get_delta_euclid(\
                personality_test,\
                    random_estimation)
            """
            

            
            delta_estim_graph_model = deltas_recommended_by_graph_model[idx_person_test]
        
            # print("{} {}".format(delta_random_estimation, delta_estim_graph_model))
            
            if (delta_random_estimation) < (delta_estim_graph_model) :
                # loss
                win_loss_tie.append(-1)
            elif (delta_estim_graph_model) < (delta_random_estimation) :
                # win
                win_loss_tie.append(1)
            else:
                # tie
                win_loss_tie.append(0)
                


                
        print('Win {0:2.2f} \tLose {1:2.2f} \tTie {2:2.2f}'.format(\
              win_loss_tie.count(1)/N_RANDOM_ESTIMATIONS, \
              win_loss_tie.count(-1)/N_RANDOM_ESTIMATIONS, \
              win_loss_tie.count(0)/N_RANDOM_ESTIMATIONS))
    
                
            