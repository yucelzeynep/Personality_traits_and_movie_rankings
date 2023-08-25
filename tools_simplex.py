#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:49:18 2023

@author: zeynep
"""
import numpy as np
import sys

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

def get_delta_personality(p1, p2, weights):
    """
    This function returns the weighted distance between inputs p1 and p2, which
    are integer arrays of size 5 (o, c, e, a, n)

    """
    d = 0
    for e1, e2, w in zip(p1, p2, weights):
        d += w * np.abs(e1 - e2)
    return d

def get_delta_movie_rankings_pairwise(movie_rankings):
    """
    This function returns the distance between each pair of movie rankings.
    """
    delta_movie_rankings_pairwise = []
        
    for i, mr1 in enumerate(movie_rankings):
        temp = []
        for j, mr2 in enumerate(movie_rankings):
            
            if i == j:

                temp.append( np.nan )
                
                continue
            
                
            temp.append(\
            get_delta_movie_ranking(mr1, mr2)\
            )
            
        delta_movie_rankings_pairwise.append(temp)    
        
    return delta_movie_rankings_pairwise

def get_delta_euclid(p1,p2):
    d = 0
    for e1, e2 in zip(p1,p2):
        d += np.square(e1-e2)
    return d

def get_delta_personalities_pairwise(personalities, weights):
    """
    This function returns the weighted distance between each pair of people 
    (ie their personalities).
    """
    delta_personalities_pairwise = []
        
    for i, p1 in enumerate(personalities):
        temp = []
        for j, p2 in enumerate(personalities):
            
            if i == j:

                temp.append( np.nan )
                
                continue
            
                
            temp.append(\
            #get_delta_personality(p1, p2, weights)\
            get_delta_euclid(p1,p2)
            )
            
        delta_personalities_pairwise.append(temp)    
        
    return delta_personalities_pairwise

def get_delta_movie_ranking(mr1, mr2):
    """
    Kendall's tau between mr1 and mr2, which are rankings of movies of 
    person 1 and 2.

	tau \in [-1, 1]
    """
    # number of conforming, disconforming evidence
    c, d, tau  = 0, 0, 0
    
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
    tau = (c-d) / (c+d)
    
    """
    new correlation 
    (tau+1)/2 \in [0,1]
    
    turn it into distance by subtracting from 1
    d = 1-(tau+1)/2 \in [0,1]
    """
    d = 1-(tau+1)/2
          
    return d
            
def get_cost(weights0,\
             personalities_train,\
             movie_rankings_train)  :
    """
    This is the objective function (cost) to minimize!
    
    The cost is equal to the minus of the correlation (r) between 
    personality difference and 
    movie taste difference.
    
    Maximizing r is equalivalent to minimizig -r. So I return -r.
    """
  
    n_persons = len(personalities_train)
    
    delta_personalities = [ [0]*n_persons for i in range(n_persons)]
    for i in range(n_persons):
        for j in range(n_persons):
            if i == j:
                delta_personalities[i][j] = np.nan
                continue
     
            delta_personalities[i][j] = get_delta_personality(\
                                      personalities_train[i], \
                                      personalities_train[j], 
                                      weights0)
            """
            delta_personalities[i][j] = get_delta_euclid(\
                                      personalities_train[i], \
                                      personalities_train[j])
                                      """

    
    delta_movie_rankings = [ [0]*n_persons for i in range(n_persons)]
    for ii in range(n_persons):
        for jj in range(n_persons):
            if ii == jj:
                delta_movie_rankings[ii][jj] = np.nan
                continue
            delta_movie_rankings[ii][jj] = get_delta_movie_ranking(\
                                      movie_rankings_train[ii], \
                                      movie_rankings_train[jj])

#    dp_flat = [j for sub in delta_personalities for j in sub]
#    dmr_flat = [j for sub in delta_movie_rankings for j in sub ]
    
    # upper triangle entries
    dp_flat  = np.asarray(delta_personalities)[np.triu_indices(3, k = 1)]
    dmr_flat = np.asarray(delta_movie_rankings)[np.triu_indices(3, k = 1)]
    
    # The values of r are between -1 and 1, inclusive.
    r = np.corrcoef(dp_flat, dmr_flat)    
    
    return -r[0,1]
