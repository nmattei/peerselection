'''
    File:   impartial.py
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
    This file implements a number of peer peer selection techniques
    that are tested in our paper.

    All methods implemented here assume that you are not allowed to grade
    yourself... otherwise it wouldn't be impartial.

    Note that this REQUIRES PYTHON 3.0+ with the newer rounding functions.
        
'''

import math
import itertools
import random
import copy
import numpy as np
from scipy import stats

## Some Settings
_DEBUG = False

def vanilla(score_matrix, k, normalize=False):
  """
  Selects k agents from a using the vanilla method
  which is just selecting agents with the highest scores.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    The number of agents to select from a.

  Returns
  -----------
    winning_set: array of tuples (agent, score)
      A list of size k of tuples (agent, score)

  Notes
  -----------

  """
  if _DEBUG: print("Running Vanilla\n")

  score_matrix = validate_matrix(score_matrix)
  if isinstance(score_matrix, int):
    return 0
  if normalize:
    score_matrix = normalize_score_matrix(score_matrix)

  # Select the k winners with highest score..
  winning_set = select_winners(score_matrix, [k], {0: list(range(score_matrix.shape[0]))})

  if _DEBUG:
      print("\nWinners:")
      for agent,score in winning_set:
          print("Agent: " + str(agent) + " with Score: " + str(score))

  return winning_set

def partition_explicit(score_matrix, k, partition, normalize=False):
  """
  Selects elements[i] agents from partition[i] based on 
  their total scores in score_matrix not taking into account
  the scores within the partition.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    The number of elements to select from the score_matrix.

  partition: dict
    A mapping from an integer --> list(indicies) where the list of 
    inidicies are the rows of the score_matrix that contain 
    the scores. 

  normalize: Boolean
    A flag for normalizing or not normalizing the score matrix.

  Returns
  -----------
    winners: array like, tuples
      A list of length sum(elements) of winners or 0 on error.

  Notes
  -----------
  """
  score_matrix = validate_matrix(score_matrix, partition)
  if isinstance(score_matrix, int):
    return 0
  if normalize:
    score_matrix = normalize_score_matrix(score_matrix, partition)

  # Allocate a number of elements to each partition.
  l = len(partition)
  # Assign the min number
  elements = [k // l]*l
  # Allocate the remainders...
  for i in range(k % l):
    elements[i] += 1

  # Select the k winners with highest score..
  winning_set = select_winners(score_matrix, elements, partition)

  if _DEBUG:
      print("\nWinners:")
      for agent,score in winning_set:
          print("Agent: " + str(agent) + " with Score: " + str(score))

  return [agent for agent,score in winning_set]

def partition(score_matrix, k, l, randomize=True, normalize=False):
  """
  Selects k elements evenly from l partitions that are randomly
  chosen.  Remainders are allotted to the earliest of the partitions
  and scores are normalized with respect to scores given outside
  the matrix by default.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    The number of elements to select from the score_matrix.

  l: integer
    The number of partitions to make.

  randomize: Boolean
    A flag for randomizing the partitions or not. 

  normalize: Boolean
    A flag for normalizing or not normalizing the score matrix.

  Returns
  -----------
    winners: array like, tuples
      A list of length sum(elements) of winners or 0 on error.

  Notes
  -----------
  """
  index_set = list(range(score_matrix.shape[0]))
  if randomize:
      random.shuffle(index_set)
  partition = even_partition_order(index_set, l)
  score_matrix = validate_matrix(score_matrix, partition)
  if isinstance(score_matrix, int):
    return 0
  if normalize:
    score_matrix = normalize_score_matrix(score_matrix, partition)

  # Allocate a number of elements to each partition.
  # Assign the min number
  elements = [k // l]*l
  # Allocate the remainders...
  for i in range(k % l):
    elements[i] += 1

  if _DEBUG: print("\nselection elements:\n" + str(elements))
  if _DEBUG: print("score matrix:\n"+str(score_matrix))

  return partition_explicit(score_matrix, elements, partition, normalize)

def dollar_partition_explicit(score_matrix, k, partition, normalize=True):
  """
  Selects k agents from the partitions based on 
  their dollar shares in score_matrix not taking into account
  the scores within the partition.

  This implementation uses the CEIL method to do the rounding.
  Hence we can return up to k+len(partitions)-1 elements.

  By default dollar partition normalizes the scores of the agents.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    Number of total elements to select.

  partition: dict
    A mapping from an integer --> list(indicies) where the list of 
    inidicies are the rows of the score_matrix that contain 
    the scores. 

  normalize: Boolean
    A flag for normalizing or not normalizing the score matrix.
    Default is true.

  Returns
  -----------
    winners: array like, tuples
      A list of length k?+len(partition)-1 of winners or 0 on error.

  Notes
  -----------
  """
  ## Set N for convinence...
  n = score_matrix.shape[0]
  l = len(partition.keys())

  score_matrix = validate_matrix(score_matrix, partition)
  if isinstance(score_matrix, int):
    return 0
  if normalize:
    score_matrix = normalize_score_matrix(score_matrix, partition)

  #Create and Conglomerate the partition scores..
  partition_scores = np.zeros((l,l))

  #Compute the score that partition p[1] gives to p[0]..
  for p in itertools.permutations(list(range(l)), 2):
      t = 0.0
      for i in partition[p[0]]:
          for j in partition[p[1]]:
              t += score_matrix[i,j]
      partition_scores[p[0], p[1]] = t

  if _DEBUG: print("\nGroup Scores:\n" + str(partition_scores))

  #Compute a distribution from this ratio for each partition.
  #grades are a vec the same lenght as the column
  dist = np.zeros(partition_scores.shape[1])
  #Sum up each row (partition):
  dist += partition_scores.sum(axis=1)
  # Normalize this into a probability distribution.
  dist = dist / dist.sum()
  if _DEBUG: print("\nDistribution:\n" + str(dist))

  #Multiply the dist by k and take the ceil of each of these 
  #to get the elements counts for each of the partitions.
  elements = [math.ceil(x*k) for x in dist]
  if _DEBUG: print("\nnum per partition:\n" + str(elements))

  # Select the elements winners with highest score..
  winning_set = select_winners(score_matrix, elements, partition)

  if _DEBUG:
      print("\nWinners:")
      for agent,score in winning_set:
          print("Agent: " + str(agent) + " with Score: " + str(score))

  return [agent for agent,score in winning_set]

def dollar_partition(score_matrix, k, l, randomize=True, normalize=True):
  """
  Selects k agents from the generated partitions based on 
  their dollar shares in score_matrix not taking into account
  the scores within the partition.

  This implementation uses the CEIL method to do the rounding.
  Hence we can return up to k+len(partitions)-1 elements.

  By default dollar partition normalizes the scores of the agents.
  Note that dollar partition is NOT STRAGETYPROOF IF IT IS NOT NORMALIZED!!

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    Number of total elements to select.

  l:integer
    Number of partitions to use.

  randomize: Boolean
    Whether or not to permute the order (make partitions random)

  normalize: Boolean
    A flag for normalizing or not normalizing the score matrix.
    Default is true.

  Returns
  -----------
    winners: array like, tuples
      A list of length k?+len(partition)-1 of winners or 0 on error.

  Notes
  -----------
  """
  index_set = list(range(score_matrix.shape[0]))
  if randomize:
      random.shuffle(index_set)
  partition = even_partition_order(index_set, l)
  score_matrix = validate_matrix(score_matrix, partition)
  if isinstance(score_matrix, int):
    return 0
  return dollar_partition_explicit(score_matrix, k, partition, normalize)

def credible_subset(score_matrix, k, m, normalize=False):
  """
  Selects k agents from the score matrix according
  to the crediable subset mechanism detailed below.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    Number of total elements to select.

  Returns
  -----------
    winners: array like
      A list of length k of winners or the empty list.

  Notes
  -----------
  Each agent reviews $m$ other agents; and $k$ out of a total 
  of $n$ agents are selected. If each agent reviews each other 
  agent, then $m=n-1$. Set $T$ is the set of agents who have the 
  top $k$ scores. Set $P$ is the set of agents who do not have 
  the top $k$ scores but will make it to top $k$ if they gave 
  zero score to all other agents. With probability $(k+|P|)/(k+m)$, 
  CredibleSubset selects a set of $k$ agents uniformly at 
  random from $T\cup P$ and with probability $1-(k+|T|)/(k+m)$, 
  it selects no one. This only makes sense if $|P|\leq m$. 
  """
  score_matrix = validate_matrix(score_matrix)
  if isinstance(score_matrix, int):
    return 0
  #Check that m is feasiable but we don't enforce 
  # that m entries be filled in (approval!)
  if m < 0 or m > score_matrix.shape[0]:
    print("infeasiable m!")
    return 0

  #Set T is the vanilla winners
  T = vanilla(score_matrix, k)

  N = list(range(score_matrix.shape[0]))
  P = []
  #For each element
  for c_agent in set(N).difference(set(T)):
    c_col = copy.copy(score_matrix[:, c_agent])
    score_matrix[:, c_agent] = 0
    manip = vanilla(score_matrix, k)
    if c_agent in manip:
      P.append(c_agent)
    score_matrix[:, c_agent] = c_col

  if _DEBUG: print("Set T: " + str(T))
  if _DEBUG: print("Set P: " + str(P))
  pr_winning_set = (float(k) + float(len(P))) / float(k+m)
  if _DEBUG: print("PR Winner: " + str(pr_winning_set))

  if random.random() < pr_winning_set:
    #shuffle the set and Returnsrn k of them.
    winning_set = T + P
    random.shuffle(winning_set)
    if _DEBUG: print("winning set:" + str(winning_set[:k]))
    winning_agents = [a for a,s in winning_set[:k]]
    return winning_agents

  #Otherwise no one wins...
  return []

def select_winners(score_matrix, elements, partition):
  """
  Selects elements[i] agents from partition[i] based on 
  their total scores in score_matrix.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  elements: array like
    A list of length equal to partition.keys() which is the 
    number of elements from each partition.  Must be less
    than the dimension of the matrix.

  partition: dict
    A mapping from an integer --> list(indicies) where the list of 
    inidicies are the rows of the score_matrix that contain 
    the scores. 

  Returns
  -----------
    winners: array like, tuples
      A list of length sum(elements) of winners
      each of a tuple (index, score) or 0 on error

  Notes
  -----------
  """
  #Sanity Checks....
  if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
    print("score_matrix is not square or has no values")
    return 0
  if sum(elements) > score_matrix.shape[0] or sum(elements) <= 0:
    print("must select more winners than shape or no winners")
    return 0

  #For each partition compute the scores of each of the elements
  winners = []
  score_tuples = {}
  for c_p in sorted(partition.keys()):
      score_tuples[c_p] = []
      #Add up the grades recieved...
      for i in partition[c_p]:
          score_tuples[c_p].append((i, score_matrix[i, :].sum()))
      #Sort by score, then by agent #.
      score_tuples[c_p] = sorted(score_tuples[c_p], key=lambda x: (-x[1],x[0]))
      # For each partition select the top s elements according to their score.
      # Note that at this point we implicitly deal with ties in lex order.
      winners += score_tuples[c_p][:elements[c_p]]
  return winners

def dollar_raffle_explicit(score_matrix, k, partition, normalize=True):
  """
  Selects *exactly* k agents from the generated partitions based on 
  their dollar shares in score_matrix not taking into account
  the scores within the partition.

  This implementation uses the raffle method where we, with replacement,
  draw partitions according to their dollar share probabilities until
  we are out.

  ** NOT IMPARTIAL IF NOT NORMALIZED! **

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    Number of total elements to select.

  partition: dict
    A mapping from an integer --> list(indicies) where the list of 
    inidicies are the rows of the score_matrix that contain 
    the scores. 

  normalize: Boolean
    A flag for normalizing or not normalizing the score matrix.
    Default is true.

  Returns
  -----------
    winners: array like, tuples
      A list of length k of winners or 0 on error.

  Notes
  -----------
  """
  ## Set N for convinence...
  n = score_matrix.shape[0]
  l = len(partition.keys())

  score_matrix = validate_matrix(score_matrix, partition)
  if isinstance(score_matrix, int):
    return 0
  if normalize:
    score_matrix = normalize_score_matrix(score_matrix, partition)

  #Create and Conglomerate the partition scores..
  partition_scores = np.zeros((l,l))

  #Compute the score that partition p[1] gives to p[0]..
  for p in itertools.permutations(list(range(l)), 2):
      t = 0.0
      for i in partition[p[0]]:
          for j in partition[p[1]]:
              t += score_matrix[i,j]
      partition_scores[p[0], p[1]] = t

  if _DEBUG: print("\nGroup Scores:\n" + str(partition_scores))

  #Compute a distribution from this ratio for each partition.
  #grades are a vec the same lenght as the column
  dist = np.zeros(partition_scores.shape[1])
  #Sum up each row (partition):
  dist += partition_scores.sum(axis=1)
  # Normalize this into a probability distribution.
  dist = dist / dist.sum()
  if _DEBUG: print("\nDistribution:\n" + str(dist))

  # Generate the RV according to the distribution above.
  # Need to check that the elements with positive support can accomidate
  # the selection....
  total = sum([len(partition[i]) for i,d in enumerate(dist) if d > 0.0])
  if total < k:
    print("Result does not have enough positive support, max selection ", str(total))
    return 0

  raffle_rvs = stats.rv_discrete(values=(list(range(len(partition))),dist))
  elements = [0]*len(partition)
  while sum(elements) != k:
    winner = raffle_rvs.rvs()
    # If there's someone to select, select them.
    if elements[winner] < len(partition[winner]):
      elements[winner] += 1

  if _DEBUG: print("\n Raffle results: ", str(elements))

  # Select the elements winners with highest score..
  winning_set = select_winners(score_matrix, elements, partition)

  if _DEBUG:
      print("\nWinners:")
      for agent,score in winning_set:
          print("Agent: " + str(agent) + " with Score: " + str(score))

  return [agent for agent,score in winning_set]


  pass

def dollar_raffle(score_matrix, k, l, randomize=True, normalize=True):
  """
  Selects *exactly* k agents from the generated partitions based on 
  their dollar shares in score_matrix not taking into account
  the scores within the partition.

  This implementation uses the raffle method where we, with replacement,
  draw partitions according to their dollar share probabilities until
  we are out.

  *** NOT IMPARTIAL IF NOT NORMALIZED ***

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    Number of total elements to select.

  l:integer
    Number of partitions to use.

  randomize: Boolean
    Whether or not to permute the order (make partitions random)

  normalize: Boolean
    A flag for normalizing or not normalizing the score matrix.
    Default is true.

  Returns
  -----------
    winners: array like, tuples
      A list of length k of winners or 0 on error.

  Notes
  -----------
  """
  index_set = list(range(score_matrix.shape[0]))
  if randomize:
      random.shuffle(index_set)
  partition = even_partition_order(index_set, l)
  score_matrix = validate_matrix(score_matrix, partition)
  if isinstance(score_matrix, int):
    return 0
  return dollar_raffle_explicit(score_matrix, k, partition, normalize)

def exact_dollar_partition_explicit(score_matrix, k, partition, normalize=True):
  """
  Selects exactly k agents from the partitions based on 
  their dollar shares in score_matrix not taking into account
  the scores within the partition.

  This implementation uses the lottery extension that we have 
  come up with.

  By default dollar partition normalizes the scores of the agents.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  k: integer
    Number of total elements to select.

  partition: dict
    A mapping from an integer --> list(indicies) where the list of 
    inidicies are the rows of the score_matrix that contain 
    the scores. 

  normalize: Boolean
    A flag for normalizing or not normalizing the score matrix.
    Default is true.

  Returns
  -----------
    winners: array like, tuples
      A list of length k?+len(partition)-1 of winners or 0 on error.

  Notes
  -----------
  """
  if _DEBUG: print("\n\tRunning Exact Dollar Raffle\n")
  ## Set N for convinence...
  n = score_matrix.shape[0]
  l = len(partition.keys())

  score_matrix = validate_matrix(score_matrix, partition)
  if isinstance(score_matrix, int):
    return 0
  if normalize:
    score_matrix = normalize_score_matrix(score_matrix, partition)

  #Create and Conglomerate the partition scores..
  partition_scores = np.zeros((l,l))

  #Compute the score that partition p[1] gives to p[0]..
  for p in itertools.permutations(list(range(l)), 2):
      t = 0.0
      for i in partition[p[0]]:
          for j in partition[p[1]]:
              t += score_matrix[i,j]
      partition_scores[p[0], p[1]] = t

  if _DEBUG: print("\nGroup Scores:\n" + str(partition_scores))

  #Compute a distribution from this ratio for each partition.
  #grades are a vec the same lenght as the column
  dist = np.zeros(partition_scores.shape[1])
  #Sum up each row (partition):
  dist += partition_scores.sum(axis=1)
  # Normalize this into a probability distribution.
  dist = dist / dist.sum()
  if _DEBUG: print("\nDistribution:\n" + str(dist))

  #Multiply the dist by k to get the quotas
  quotas = [x*k for x in dist]
  if _DEBUG: print("\nPartition Quotas:\n" + str(quotas))

  # You're A FOOL MATTEI!
  # Get the lottery. 
  lottery = randomized_allocation_from_quotas(quotas)

  if _DEBUG:
    print("\n Lottery: ")
    for k in sorted(lottery.keys(), reverse=True):
      print(str(k) + " :: " + str(lottery[k]))

  # Make sure we have the right view... 
  # this could likely be done in a more clever way...
  v = []
  xk = []
  pk = []
  for i,k in enumerate(lottery.keys()):
    v.append(k)
    xk.append(i)
    pk.append(lottery[k])

  # Create an RV and randomly draw an allocation
  model_rvs = stats.rv_discrete(values=(xk, pk))
  #print((list(lottery.keys()), list(lottery.values())))
  allocation = model_rvs.rvs()

  if _DEBUG: print("\nAllocation: " + str(v[allocation]))

  # Select the elements winners with highest score..
  winning_set = select_winners(score_matrix, v[allocation], partition)

  if _DEBUG:
      print("\nWinners:")
      for agent,score in winning_set:
          print("Agent: " + str(agent) + " with Score: " + str(score))

  return [agent for agent,score in winning_set]


"""
######## Helper Functions
"""

def randomized_allocation_from_quotas(quotas):
  """
  Given the quotas (s_1 ... s_l)
  we implement Omer's Algorithm which greedly builds 
  a probability distribution over vectors which yield
  a discrete probability distribution over integer allocations. 

  Parameters
  -----------
  quotas: array-like
    The list of s_i's for the clusters.  We will strip off any "almost"
    integer ones necessary before we start in order to make this consistant.
    Each of these should be a real value.

  Returns
  -----------
  distribution: dict
    A mapping from (tuple) --> p where each tuple is 
    the same length and is an allocation selected with probability
    p.  Note that the sum of p's should be 1.0.

  sort_keys: list like
    A list of the indicies of the origional distribution so
    selections can be mapped back onto the *CORRECT* partitions.

  Notes
  -----------
  """

  # Just in Case:
  split_shares = [math.modf(i) for i in quotas]
  frac_shares = [x[0] for x in split_shares]
  integer_shares = [int(x[1]) for x in split_shares]

  #Likely a more elegant way to do this, bit hacky.
  s = [q - math.floor(q) for q in quotas]
  sort_keys = sorted(range(len(s)), key=lambda k: s[k])
  s = sorted(copy.copy(quotas), key=lambda q: q - math.floor(q))

  if _DEBUG:
    print("Quotas: " + str(quotas))
    print("Sorted: " + str(s))
    print("Sort Key: " + str(sort_keys))
  

  alpha = sum(frac_shares)

  #Check to make sure that alpha is near what it should be
  if not np.isclose(alpha, int(round(alpha))):
    print("Alpha is " + str(alpha) + " too far from an integer.")
    exit(0)

  alpha = int(round(alpha))
  #print(quotas)
  #print(alpha)
  #print(s)
  allocated_probability = [0.0]*len(quotas)
  total_probability = 0.0
  distribution = {}

  # Indicies
  low = 0
  high = len(quotas)-1
  handled = 0

  while handled <= len(quotas):
  # Build Vector
    allocation = copy.copy(s)
    allocation = [math.floor(x) if i < low else x for i,x in enumerate(allocation)]
    allocation = [math.ceil(x) if i >= low and i < low+alpha else x for i,x in enumerate(allocation)]
    allocation = [math.floor(x) if i >= low + alpha and i <= high else x for i,x in enumerate(allocation)]
    allocation = [math.ceil(x) if i > high else x for i,x in enumerate(allocation)]

    #print(allocation)

    #Cases
    p = 0
    if s[low] - math.floor(s[low]) - allocated_probability[low] < \
        math.ceil(s[high]) - s[high] - total_probability + allocated_probability[high]:
      p = s[low] - math.floor(s[low]) - allocated_probability[low]
      # Update probability list.
      allocated_probability = [x + p if i >= low and i < low+alpha else x for i,x in enumerate(allocated_probability)]
      allocated_probability = [x + p if i > high else x for i,x in enumerate(allocated_probability)]
      low += 1
    else:
      p = math.ceil(s[high]) - s[high] - total_probability + allocated_probability[high]
      # Update probability list.
      allocated_probability = [x + p if i >= low and i < low+alpha else x for i,x in enumerate(allocated_probability)]
      allocated_probability = [x + p if i > high else x for i,x in enumerate(allocated_probability)]
      high -= 1
      alpha -= 1

    #print(str(p) + " total: " + str(total_probability))
    #print(allocated_probability)

    total_probability += p
    distribution[tuple(allocation)] = p
    handled += 1

  if _DEBUG:
    print("Total Probability is: " + str(total_probability) + "\n\n NON-SORTED Lottery")
    for k in sorted(distribution.keys(), reverse=True):
     print(str(k) + " :: " + str(distribution[k]))

  if not np.isclose(sum(distribution.values()), 1.0):
    print("Didn't get a distribution on allocation.  Allocated " + str(sum(distribution.values())) + ", should be near 1.0.")
    exit(0)

  # Convert Allocation vectors so that they draw from THE CORRECT CLUSTERS!!!
  # SORT THE ALLOCATION BY THE RESORT!
  sorted_distribution = {}
  for vec,prob in distribution.items():
    sorted_vec = [0]*len(sort_keys)
    for i,v in enumerate(vec):
      sorted_vec[sort_keys[i]] = v
    sorted_distribution[tuple(sorted_vec)] = prob

  if _DEBUG: 
    print("\nSORTED LOTTERY:")
    for k in sorted(sorted_distribution.keys(), reverse=True):
     print(str(k) + " :: " + str(sorted_distribution[k]))

  return sorted_distribution

def normalize_score_matrix(score_matrix, partition={}):
  """
  Normalize a score_matrix so that all the scores given 
  by each agent to other agents outside their clusters
  is equal to exactly 1.
  

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  partition: dict
    A mapping from an integer --> list(indicies) where the list of 
    inidicies are the rows of the score_matrix that contain 
    the scores. 

  Returns
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents
    normalized such that each column sums to 1.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  Notes
  -----------
  """
  # Normalize so that each Column sums to 1!
  col_sums = score_matrix.sum(axis=0)
  
  if partition != {}:
    #Sanity check: iterate over the agents by partition and ensure
    #that they gave someone a score... otherwise give 1/(n-|P|-1)
    #to each agent outside.
    n = score_matrix.shape[0]
    #Iterate over partitions
    for c_key in partition.keys():
     #For each agent in each partition...
     for c_agent in partition[c_key]:
       #Check that they gave scores...
       if col_sums[c_agent] == 0:
         #If they didn't, give everyone not themselves and not in their
         #partition a point.
         for c_other in range(n):
           if c_other != c_agent and c_other not in partition[c_key]:
             score_matrix[c_other][c_agent] = 1.
  else:
    #Give a 1 to everyone but themselves if their sum is 0.
    for j, total in enumerate(col_sums):
      if total == 0:
        score_matrix[:, j] = 1.
        score_matrix[j, j] = 0.

  #Resum and normalize..
  col_sums = score_matrix.sum(axis=0)
  norm_score_matrix = score_matrix / col_sums[np.newaxis , : ]
  # We may still have nan's because everyone's in one partition...
  norm_score_matrix = np.nan_to_num(norm_score_matrix)
  if _DEBUG: print("\nnormalized score matrix:\n" + str(norm_score_matrix))
  return norm_score_matrix

def even_partition_order(order, l):
  """
  Partition order into l parts as evenly as possible.
  The earlier partitions will take the remainders.

  Parameters
  -----------
  order: list like
    A list of the elements to be partitioned.

  l: integer
    The number of partitions to make.

  Returns
  -----------
    partition: dict
      A dict from int ---> order which gives
      the partitions of order into set 0...l-1.

  Notes
  -----------
  """

  #Create partition index sets..
  partition = {}
  #evenly distributed.
  index_set = copy.copy(order)
  for i in range(l):
    cut = int(math.ceil(float(len(index_set)/(l - i))))
    if len(index_set) >= cut:
      partition[i] = index_set[:cut]
      index_set = index_set[cut:]
    else:
        partition[i] = index_set
  if _DEBUG: print(order)
  if _DEBUG: print(partition)

  return partition

def validate_matrix(score_matrix, partition={}):
  """
  Validate and enforce constraints on a score matrix object:
  (1) must be square
  (2) must not score self
  Optional:
  (3) must not score others in my partition.
  (4) partition must not overlap.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  partition: dict
    non-overlapping partition of elements that are grouped together.

  Returns
  -----------
  score_matrix: array like
    score matrix obeying the above constraints.

  Notes
  -----------
  """
  if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
      print("score matrix is not square or has no values")
      return 0
  
  #Enforce the Diagonal is 0's
  for i in range(score_matrix.shape[0]):
      score_matrix[i,i] = 0.

  if partition != {}:
    if _DEBUG: print("\npartitions:\n" + str(partition))
    #Validate the partitions and check that 0's for the elements in my
    #partition.
    #Ensure that the partitions don't overlap.
    agent_set = list(itertools.chain(*partition.values()))
    if len(agent_set) != len(set(agent_set)):
     print("partitioning contains duplicates")
     return 0

    # Note we have to blank out the scores of the other agents in our partitions
    # before we do anything else.  We know we already have a 0. grade for ourselves.
    for c in partition.values():
      for pair in itertools.permutations(c, 2):
        score_matrix[pair[0],pair[1]] = 0.

  if _DEBUG: print("score matrix:\n"+str(score_matrix))

  return score_matrix
















