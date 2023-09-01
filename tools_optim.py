#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:49:18 2023

@author: zeynep
"""
import numpy as np
import sys

from scipy.optimize import minimize


import preferences


def constraint(t):
    """
    Optimization constraint 
    Weights sum up to 1
    """
    return np.sum(t)-1

def constraint_real(t):
    """
    Optimization constraint 
    Weights are real (no imaginary part)
    """
    return np.sum(np.iscomplex(t))

def get_delta_ocean_weighted_Euc(p1, p2, weights):
    """
    This function returns the weighted Euclidean distance between inputs p1 and 
    p2, which are arrays of size 5 (o, c, e, a, n)

    """
    d2 = 0
    for e1, e2, w in zip(p1, p2, weights):
        d2 += w * (e1 - e2)**2
        
    if d2 > 0:
        d = np.sqrt(d2)
    else:
        d = 0
    return d


def get_delta_movie_ranking_RBSC(mr1, mr2):
    """
    Kendall's tau between mr1 and mr2, which are rankings of movies of 
    person 1 and 2.

	tau \in [-1, 1]
    """
    # number of conforming, disconforming evidence
    c, d, rho  = 0, 0, 0
    
    movs1 = sorted(np.unique(mr1))
    movs2 = sorted(np.unique(mr2))
    
    if movs1 == movs2:
        movs = movs1
    else:
        print('Ranked moview of these people are not the same')
        sys.exit(0)
    
    for m1 in movs:
        for m2 in movs:
            # if same movie you do not need to check relative ranking (no such thing)
            if m1 == m2:
                continue

            rel_rank_p1 = np.sign(mr1.index(m1) - mr1.index(m2))
            rel_rank_p2 = np.sign(mr2.index(m1) - mr2.index(m2))
            
			# if they have same sign (1,1) or (-1, -1) --> confirming
			# if they have opposite signs --> disconforming
 			# if they have same rank, the above sign is 0 --> omit
            if rel_rank_p1 * rel_rank_p2 > 0:
                c += 1
            elif rel_rank_p1 * rel_rank_p2 < 0:
                d += 1
                
            
    """
    correlation tau \in [-1,1]
    """
    rho = (c-d) / (c+d)
    
    """
    new correlation 
    (tau+1)/2 \in [0,1]
    
    turn it into distance by subtracting from 1
    d = 1-(tau+1)/2 \in [0,1]
    """
    d = 1-(rho+1)/2
          
    return d


         
def get_cost(weights0,\
             ocean_train,\
             movie_rankings_train)  :
    """
    This is the objective function (cost) to minimize!
    
    The cost is equal to the minus of the correlation (r) between 
    personality difference and 
    movie taste difference.
    
    Maximizing r is equalivalent to minimizig -r. So I return -r.
    """
  
    n_persons = len(ocean_train)
    
    delta_oceans = [ [0]*n_persons for i in range(n_persons)]
    for i in range(n_persons):
        for j in range(n_persons):
            if i == j:
                delta_oceans[i][j] = np.nan
                continue
     
            delta_oceans[i][j] = get_delta_ocean_weighted_Euc(\
                                      ocean_train[i], \
                                      ocean_train[j], 
                                      weights0)
    """ 
    # for debugging
    if np.allclose(delta_oceans, np.array(delta_oceans).T.tolist(), rtol=1e-08, atol=1e-08):
        print('delta_oceans is symmetric --> OK')
    """
    
    delta_movie_rankings = [ [0]*n_persons for i in range(n_persons)]
    for ii in range(n_persons):
        for jj in range(n_persons):
            if ii == jj:
                delta_movie_rankings[ii][jj] = np.nan
                continue
            
            mr1 = movie_rankings_train[ii]
            mr2 = movie_rankings_train[jj]
            delta_movie_rankings[ii][jj] = get_delta_movie_ranking_RBSC(\
                                      mr1, \
                                      mr2)

#    dp_flat = [j for sub in delta_oceans for j in sub]
#    dmr_flat = [j for sub in delta_movie_rankings for j in sub ]
    
    # upper triangle entries
    dp_flat  = np.asarray(delta_oceans)[np.triu_indices(n_persons, k=1)]
    dmr_flat = np.asarray(delta_movie_rankings)[np.triu_indices(n_persons, k = 1)]
    
  
    
    # The values of r are between -1 and 1, inclusive.
    r = np.corrcoef(dp_flat, dmr_flat)    
    
    return -r[0,1]


def train_graph_model(om, ocean_train, movie_rankings_train, weights0):
    """
    Train the graph model with the training sets of ocean and movie_rankings
    """

    if om == "Powell" or \
    om == 'Nelder-Mead' or\
    om == 'CG' or\
    om == 'BFGS':
        
        res = minimize(get_cost, \
           weights0, \
           method = om,\
           args=(ocean_train, movie_rankings_train),\
           options={'maxiter': 10})

    elif om == "L-BFGS-B" or\
    om == "TNC":
        
            res = minimize(get_cost, \
               weights0, \
               method = om,\
               args=(ocean_train, movie_rankings_train),\
               bounds=preferences.BOUNDS)
    else:

           res = minimize(get_cost, \
                       weights0, \
                       method = om,\
                       args=(ocean_train, movie_rankings_train),\
                       bounds=preferences.BOUNDS,\
                       constraints=preferences.CONS)
        
    weights_opt = res.x         # weights that minimize obj fun
    
    
    ###########################################################################
    """
    Observe training performance and weights
    
    The initial cost has to be higher than the final (optimal) cost.
    
    Optimization method is default.
    It can be changed, but it does not make much difference 
    """
    cost_init = get_cost(weights0,\
                               ocean_train,\
                               movie_rankings_train) 
        
    cost_opt = get_cost(weights_opt,\
                              ocean_train,\
                              movie_rankings_train) 
    
    print('cost_init: {0:2.2f} \t cost_opt: {1:2.2f}'.format(cost_init, cost_opt))

    
    if cost_init == cost_opt:
                    
        print('cost_init \t{0:2.2f}'.format(cost_init))
        print('cost_opt \t{0:2.2f}'.format(cost_opt))
        
        print('')
        
        formatted_list = [ '%.10f' % elem for elem in weights0 ]
        print('weights0 \t{}'.format( formatted_list))
        
        formatted_list = [ '%.10f' % elem for elem in weights_opt ]
        print('weights_opt \t{}'.format( formatted_list))

        print('')
        
        print('Weights cannot be updated')
#        sys.exit("Weights cannot be updated")
        
    return weights_opt        
    

#def get_delta_euc(p1, p2):
#    """
#    This function returns the euclidean distance between inputs p1 and p2, which
#    are integer arrays.
#    
#    They can be of size 5 (o, c, e, a, n --> personality) or
#    of size 10 (movie ratings)
#
#    """
#    d2 = 0
#    for e1, e2 in zip(p1, p2):
#        d2 +=  (e1 - e2)**2
#        
#    return np.sqrt(d2)

#def get_delta_euc_pairwise(vs):
#    """
#    This function returns the eucklidean distance between each pair of array of vectors 
#    (ie vectors can be personalities or rankings).
#    """
#    delta_pairwise = []
#        
#    for i, v1 in enumerate(vs):
#        temp = []
#        for j, v2 in enumerate(vs):
#            
#            if i == j:
#
#                temp.append( np.nan )
#                
#                continue
#            
#                
#            temp.append(\
#            get_delta_euc(v1, v2)\
#            )
#            
#        delta_pairwise.append(temp)    
#        
#    return delta_pairwise


#def get_delta_oceans_pairwise(personalities, weights):
#    """
#    This function returns the weighted distance between each pair of people 
#    (ie their personalities).
#    """
#    delta_oceans_pairwise = []
#        
#    for i, p1 in enumerate(personalities):
#        temp = []
#        for j, p2 in enumerate(personalities):
#            
#            if i == j:
#
#                temp.append( np.nan )
#                
#                continue
#            
#                
#            temp.append(\
#            get_delta_ocean(p1, p2, weights)\
#            )
#            
#        delta_oceans_pairwise.append(temp)    
#        
#def get_delta_movie_ranking_RBSCs_pairwise(movie_rankings):
#    """
#    This function returns the distance between each pair of movie rankings.
#    """
#    delta_movie_rankings_pairwise = []
#        
#    for i, mr1 in enumerate(movie_rankings):
#        temp = []
#        for j, mr2 in enumerate(movie_rankings):
#            
#            if i == j:
#
#                temp.append( np.nan )
#                
#                continue
#            
#                
#            temp.append(\
#            get_delta_movie_ranking_RBSC(mr1, mr2)\
#            )
#            
#        delta_movie_rankings_pairwise.append(temp)    
#        
#    return delta_movie_rankings_pairwise   #    return delta_oceans_pairwise