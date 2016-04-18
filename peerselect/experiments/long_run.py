'''
    File:   pnas_long_run.py
    Author: Nicholas Mattei (nsmattei@gmail.com)
    Date:   Jan 11 2016

  * Copyright (c) 2015, Nicholas Mattei and NICTA
  * All rights reserved.
  *
  * Developed by: Nicholas Mattei
  *               NICTA
  *               http://www.nickmattei.net
  *               http://www.preflib.org
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *     * Redistributions of source code must retain the above copyright
  *       notice, this list of conditions and the following disclaimer.
  *     * Redistributions in binary form must reproduce the above copyright
  *       notice, this list of conditions and the following disclaimer in the
  *       documentation and/or other materials provided with the distribution.
  *     * Neither the name of NICTA nor the
  *       names of its contributors may be used to endorse or promote products
  *       derived from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY NICTA ''AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL NICTA BE LIABLE FOR ANY
  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.    
    

About
--------------------
  This is a simple experiment file for the cluster that runs through
  a bunch of steps and saves the pickled file so we can load 
  the objects back up later.

        
'''


import math
import csv
import numpy as np
import random
import itertools
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from collections import Counter

from peerselect import impartial
from peerselect import profile_generator

class Impartial:
    VANILLA = "Vanilla"
    EXACT = "ExactDollarPartition"
    PARTITION = "Partition"
    DPR = "DollarPartitionRaffle"
    CREDIABLE = "CredibleSubset"
    RAFFLE = "DollarRaffle"
    ALL = (VANILLA, EXACT, PARTITION, RAFFLE, CREDIABLE, DPR)

random.seed(1112016)
s = 1000
test_n = [150]
test_k = [15, 20, 25, 30, 35]
test_m = [5, 7, 9, 11, 13, 15]
test_l = [3, 4, 5, 6]
test_p = [0.0, 0.1, 0.2, 0.35, 0.5]

# Map for all Results.
results = {}
order_results = {}
for n,k,m,l,p in itertools.product(test_n, test_k, test_m, test_l, test_p):
    agents = np.arange(0, n)
    for c_sample in range(s):
        # Generate a profile and clustering
        profile = profile_generator.generate_mallows_mixture_profile(agents, agents, [1.0], [agents], [p])
        clustering = impartial.even_partition_order(sorted(agents, key=lambda j: random.random()), l)
        
        # Uncomment one:
        # Borda
        scores = np.arange(m, 0, -1)
        #Lexicographic
        #scores = [pow(n, i) for i in np.arange(m, 0, -1)]
        
        # Generate an m-regular assignment
        m_assignment = profile_generator.generate_approx_m_regular_assignment(agents, m, clustering, randomize=True)
        score_matrix = profile_generator.strict_m_score_matrix(profile, m_assignment, scores)

        #Compute Target Set.
        target_set = impartial.vanilla(score_matrix, k)
        
        # Capture the winning sets
        ws = {}
        # Let everyone else have the same size set so they are all compareable.
        vs = [i for i,j in target_set]
        ws[Impartial.VANILLA] = vs
        
        # Let CRED, PART, and RAFFLE have bigger sets...
        ws[Impartial.EXACT] = impartial.exact_dollar_partition_explicit(score_matrix, k, clustering, normalize=True)
        ws[Impartial.PARTITION] = impartial.partition_explicit(score_matrix, k, clustering, normalize=False)
        
        ws[Impartial.CREDIABLE] = impartial.credible_subset(score_matrix, k, m, normalize=False)
        ws[Impartial.DPR] = impartial.dollar_raffle_explicit(score_matrix, k, clustering, normalize=True)
        #Call Raffle and have everyone in a cluster by themselves = Dollar.
        ws[Impartial.RAFFLE] = impartial.dollar_raffle(score_matrix, k, n, randomize=True, normalize=True)
            
        # Only want to track the size of the set intersection.
        #
        # REMEMBER TO CHANGE THE VARIABLE INDEXING DOWN HERE!!!
        #
        #
        for x in Impartial.ALL:
            key = (n, k, m, l, p, s, x)
            results[key] = results.get(key, []) + [len(set(vs) & set(ws[x]))]
        for x in Impartial.ALL:
            key = (n, k, m, l, p, s, x)
            order_results[key] = order_results.get(key, []) + [len(set(np.arange(0, k)) & set(ws[x]))]

    print("Finished: " + ",".join([str(x) for x in [n, k, m, l, p, s]]))
print("Done")


## Save the current runs
with open("./PNAS_LongPickle_versus_vanilla.pickle", 'wb') as output_file:
    pickle.dump(results, output_file)
with open("./PNAS_LongPickle_versus_base.pickle", 'wb') as output_file:
    pickle.dump(order_results, output_file)
