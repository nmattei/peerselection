'''
    File:   test_profile_generator.py
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
    This is a test file using py.test.  See how it works.
        
'''

import pytest

import numpy as np
from peerselect import profile_generator

def test_generate_approx_m_regular_assignment():
  ''' 
    Test to make sure the assignment generator is 
    creating assignemnts that are consistent...
  '''

  assert profile_generator.generate_approx_m_regular_assignment([0, 1, 2, 3], 3, randomize=False) \
    == {0:[1, 2, 3], 1:[2, 3, 0], 2:[3, 0, 1], 3:[0, 1, 2]}

  agents = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  m = 2
  clusters = {  0:[0, 1, 2],
                1:[3, 4, 5],
                2:[6, 7, 8]}
  assert profile_generator.generate_approx_m_regular_assignment(agents, m, clusters, randomize=False) \
    == { 0:[3,6],
         1:[4,7],
         2:[5,8],
         3:[6,0],
         4:[7,1],
         5:[8,2],
         6:[0,3],
         7:[1,4],
         8:[2,5]}

  print("Randomize and check that each person gets a review per other cluster.")
  result = profile_generator.generate_approx_m_regular_assignment(agents, m, clusters, randomize=True)
  for k,v in result.items():
    for c, a in clusters.items():
      if v[0] in a:
        assert v[1] not in a

  # Make sure that even if the ordering doesn't break if n/l is odd...



def test_generate_mallows_mixture_ranking():
  '''
    Test to make sure the mallows generator is still working...
  '''

  assert profile_generator.generate_mallows_mixture_ranking([0,1,2,3,4], [1,0,2,3,4], [1.0], [[4, 3, 2, 1, 0]], [0.0]) \
    == {0: [4, 3, 2, 1, 0],
        1: [4, 3, 2, 1, 0], 
        2: [4, 3, 2, 1, 0], 
        3: [4, 3, 2, 1, 0],
        4: [4, 3, 2, 1, 0],}



def test_profile_classes_to_score_matrix():
  '''
    Test the class generator...
  '''
  p =  {0: [4, 3, 2, 1, 0],
        1: [4, 3, 2, 1, 0], 
        2: [4, 3, 2, 1, 0], 
        3: [4, 3, 2, 1, 0],
        4: [4, 3, 2, 1, 0],}

  assert (profile_generator.profile_classes_to_score_matrix(p, [5, 0], [0.2, 0.8]) == \
    np.array(  [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5]]  )).all()

  assert (profile_generator.profile_classes_to_score_matrix(p, [5,4,0], [0.01,0.19,0.8]) == \
    np.array(  [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5]]  )).all()


def test_restrict_score_matrix():
  sm = np.array(  [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5]]  )

  assignment = {0:[4], 1:[], 2:[], 3:[2], 4:[2]}

  assert (profile_generator.restrict_score_matrix(sm, assignment) == \
      np.array(  [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 5, 5],
                  [0, 0, 0, 0, 0],
                  [5, 0, 0, 0, 0]]  )).all()

def test_strict_m_score_matrix():
  #  profile: dict
  #  mapping of agents to their orders as a list with list[0] being
  #  the most prefered.
  
  p =  {0: [4, 3, 2, 1, 0],
        1: [3, 2, 1, 0, 4], 
        2: [2, 1, 0, 4, 3], 
        3: [1, 0, 4, 3, 2],
        4: [0, 4, 3, 2, 1],}

  assignment = {0:[4, 2], 1:[0, 3], 2:[2, 3], 3:[1, 4], 4:[3, 4]}

  assert (profile_generator.strict_m_score_matrix(p, assignment, [9, 8]) == \
      np.array(  [[0, 8, 0, 0, 0],
                  [0, 0, 0, 9, 0],
                  [8, 0, 9, 0, 0],
                  [0, 9, 8, 0, 8],
                  [9, 0, 0, 8, 9]]  )).all()


