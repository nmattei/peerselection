'''
    File:   variance_exp.py
    Author: Nicholas Mattei (nsmattei@gmail.com)
    Date:   July 30th, 2015

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
  This runs a simple stepping experiment and saves the results to a file.
  Note that it DOES NOT TRACK the score vector etc.  This is meant 
  as a way to run a series of steps -- not a comprehensive experimental framework.
        
'''
import pickle
import numpy as np
import random
import itertools
import pandas as pd
from collections import Counter

_DEBUG = False

from peerselect import impartial
from peerselect import profile_generator

#random.seed(15)

class Impartial:
    VANILLA = "Vanilla"
    PARTITION = "Partition"
    DOLLAR = "DollarPartition"
    DPR = "DollarPartitionRaffle"
    CREDIABLE = "CredibleSubset"
    RAFFLE = "DollarRaffle"
    ALL = (VANILLA, DOLLAR, PARTITION, RAFFLE, CREDIABLE, DPR)


# Exponential.
#scores = [pow(n, 4), pow(n, 3), pow(n, 2), n, 1]
#dist = [0.1, 0.2, 0.2, 0.2, 0.3]

#Borda
#scores = [3, 2, 1, 0]
#dist = [0.25, 0.25, 0.25, 0.25]

# Psuedo 10 point Normal... \sigma~=1
grades = ["A+", "A", "B+", "B", "C+", "C", "D+", "D", "F"]
scores = [8, 7, 6, 5, 4, 3, 2, 1, 0]
# Normal...
dist = [0.03, 0.05, 0.12, 0.15, 0.30, 0.15, 0.12, 0.05, 0.03]

s = 1000

# For now assume these are singles.

test_n = [130]
test_k = [25]
test_m = [5, 10, 15]
test_l = [5]
test_p = [0.1]

# Output File name
## Save the current runs
out_name = "../notebooks/pickled_runs/variance_NSF1000s_130n_25k_5-15m_5l.pickle"


# Map for all results... We'll build a high level index out of this later...
results = {}
for n,k,m,l,p in itertools.product(test_n, test_k, test_m, test_l, test_p):
    
    # Compute some artifacts from the scoring distributions.
    agents = np.arange(0, n)
    
    #Bit Hacky but faster.... Generate a unanimous score matrix and compute some properties..
    t_profile = profile_generator.generate_mallows_mixture_profile(agents, agents, [1.0], [agents], [0.0])
    t_matrix = profile_generator.profile_classes_to_score_matrix(t_profile, scores, dist)
    
    #Determine how many of each bin we have.
    # Compute the binning from The SCORE MATRIX ---> Tally guy 0.
    t = Counter(list(t_matrix[:,0]))
    size = [t[k] for k in sorted(t.keys(), reverse=True)]
    # Determine how many of each bin we should have...
    n_from_bin = [0]*len(size)
    left = k
    for i,j in enumerate(n_from_bin):
        if left > 0: n_from_bin[i] = min(size[i], left)
        left -= size[i]
    cum_n_from_bin = list(np.cumsum(n_from_bin))
    
    # Determine what bin they should go in according to the ground truth.
    # Just take the first guys's vector and iterate over it.
    # Guy i got score v and should be in the corresponding bin as indexed by the score vector.
    in_bin = {i:scores.index(v) for i,v in enumerate(list(t_matrix[:, 0]))}
    
    # Containers for Results
    #count_results = {x:[0]*k for x in Impartial.ALL}
    #per_sample_bin_results = []
    # Multilevel dict: {METHOD} --> {SAMPLE} --> [grade]
    grade_results = {m:{x:[0]*len(size) for x in range(s)} for m in Impartial.ALL}
    for c_sample in range(s):
        #Generate a full profile and a clustering.
        profile = profile_generator.generate_mallows_mixture_profile(agents, agents, [1.0], [agents], [p])
        clustering = impartial.even_partition_order(sorted(agents, key=lambda j: random.random()), l)

        #Generate an approx-m-regular assignment.
        m_assignment = profile_generator.generate_approx_m_regular_assignment(agents, m, clustering, randomize=True)
        score_matrix = profile_generator.profile_classes_to_score_matrix(profile, scores, dist)
        score_matrix = profile_generator.restrict_score_matrix(score_matrix, m_assignment)

        #Compute Target Set.
        target_set = impartial.vanilla(score_matrix, k)

        ws = {}
        ws[Impartial.DOLLAR] = impartial.dollar_partition_explicit(score_matrix, k, clustering, normalize=True)
        size_ws = len(ws[Impartial.DOLLAR])
        # Let everyone else have the same size set so they are all compareable.
        ws[Impartial.VANILLA] = [i for i,j in impartial.vanilla(score_matrix, size_ws)]
        # Let CRED, PART, and RAFFLE have bigger sets...
        ws[Impartial.PARTITION] = impartial.partition_explicit(score_matrix, size_ws, clustering, normalize=False)
        ws[Impartial.CREDIABLE] = impartial.credible_subset(score_matrix, size_ws, m, normalize=False)
        ws[Impartial.DPR] = impartial.dollar_raffle_explicit(score_matrix, size_ws, clustering, normalize=True)
        #Call Raffle and have everyone in a cluster by themselves = Dollar.
        ws[Impartial.RAFFLE] = impartial.dollar_raffle(score_matrix, size_ws, n, randomize=True, normalize=True)
        
        # Update the per bin picking for each type.
        for x in Impartial.ALL:
            for e in ws[x]:
                grade_results[x][c_sample][in_bin[e]] += 1
        
    t = (n, k, m, l, p)
    results[t] = grade_results
    print("Finished: " + ",".join([str(x) for x in [n, k, m, l, p]]))

## Save the current runs
with open(out_name, 'wb') as output_file:
    pickle.dump(results, output_file)

print("Done")
print("Wrote to: " + out_name)
print("Score: " + str(scores))
print("Distribution: " + str(dist))
print("Size: " + str(size))
