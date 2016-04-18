'''
    File:   test_impartial.py
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
import copy
import random

import numpy as np
from peerselect import impartial

def test_select_winners():
  score_matrix = np.array([[1, 1, 1],
                           [1, 0, 0],
                           [0, 0, 0]])
  partition = {0: [0, 1, 2]}
  assert impartial.select_winners(score_matrix, [1], partition) \
    == [(0,3)]
  assert impartial.select_winners(score_matrix, [2], partition) \
    == [(0,3), (1,1)]


  score_matrix = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [0, 0, 0]])
  assert impartial.select_winners(score_matrix, [1], partition) \
    == [(0,3)]
  assert impartial.select_winners(score_matrix, [2], partition) \
    == [(0,3), (1,3)]

  #Error cases..
  assert impartial.select_winners(score_matrix, [5], partition) \
      == 0
  assert impartial.select_winners(score_matrix, [0], partition) \
      == 0
  score_matrix = np.array([[1, 1],
                           [1, 1],
                           [0, 0]])
  assert impartial.select_winners(score_matrix, [1], partition) \
    == 0

def test_vanilla():
  '''
  Tests for the vanilla mechanism.
  The mechanism as implemented uses lexicographic tie breaking.
  '''

  a = np.array([[1, 1, 1],
                [1, 0, 0],
                [0, 0, 0]])
  assert impartial.vanilla(a, 1) == [(0,2)]
  assert impartial.vanilla(a, 2) == [(0,2), (1,1)]
  assert impartial.vanilla(a, 3) == [(0,2), (1,1), (2,0)]

  ''' check normalization code '''
  assert impartial.vanilla(a, 1, True) == [(0, 2.0)]
  assert impartial.vanilla(a, 2, True) == [(0, 2.0), (1,1.0)]
  assert impartial.vanilla(a, 3, True) == [(0, 2.0), (1, 1.0), (2, 0.0)]

  a = np.array([[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0]])
  assert impartial.vanilla(a, 1) == [(0,3)]
  assert impartial.vanilla(a, 2) == [(0,3), (1,3)]
  assert impartial.vanilla(a, 3) == [(0,3), (1,3), (2,3)]

def test_partition_explicit():
  ''' Test the explicit partition mechanism
  '''
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]])

  assert impartial.partition_explicit(a, [1, 0], {0: [0,1], 1:[2,3]}) \
    == [1]
  assert impartial.partition_explicit(a, [1, 1], {0: [0,1], 1:[2,3]}) \
    == [1,2]
  '''
    Agent 0 still wins if everyone is in the same partition.
  '''
  assert impartial.partition_explicit(a, [1], {0: [0,1,2,3]}, True) \
    == [0]

  assert impartial.partition_explicit(a, [1,1], {0: [0,1], 1: [2, 3]}, True) \
    == [0, 2]

  assert impartial.partition_explicit(a, [1,1], {0: [0,1,2], 1: [2, 3]}, True) \
    == 0


  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 0]])
  assert impartial.partition_explicit(a, [1,1], {0: [0,1], 1: [2, 3]}, True) \
    == [1, 3]

def test_partition():

  print("Check top element") 
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1]])
  assert impartial.partition(a, 1, 1) == [0]
  print("Ensure normalization has no affect")
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1]])
  assert impartial.partition(a, 1, 1, False, False) == [0]

  #Check that when we break into two it's in order.
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1]])
  assert impartial.partition(a, 2, 2, False, False) == [1, 3]

  #Normalizaton doesn't change result..
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1]])
  assert impartial.partition(a, 2, 2, False, True) == [1, 3]

  #Even with random we get the best element somewhere...
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1]])
  assert 1 in impartial.partition(a, 2, 2, True, False)

def test_dollar_partition_explicit():
  print("\nCHECKING NORMALIZATION!") 
  a = np.array([[0, 0, 0, 0],
                [1, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

  assert impartial.dollar_partition_explicit(a, 1, {0:[0, 1], 1:[2, 3]}, normalize=True) == [1, 2]

  print("Check top element") 
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1000, 0, 0, 0]])

  assert impartial.dollar_partition_explicit(a, 1, {0:[0, 1], 1:[2, 3]}) == [1,3]

def test_dollar_partition():
  print("Check top element") 
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

  assert impartial.dollar_partition(a, 1, 2,randomize=False, normalize=False) == [1]

def test_exact_dollar_partition_explicit():
  print("\nCHECKING NORMALIZATION!") 
  a = np.array([[0, 0, 0, 0],
                [1, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

  assert impartial.exact_dollar_partition_explicit(a, 1, {0:[0, 1], 1:[2, 3]}, normalize=True) == [1]

  print("Check top element") 
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1000, 0, 0, 0]])

  assert impartial.exact_dollar_partition_explicit(a, 1, {0:[0, 1], 1:[2, 3]}) == [1,3]



def test_randomized_allocation_from_quotas():
  print("Example 1")
  '''
  #s = [1.15, 1.15, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.8, 1.8, 1.9]
  s = [1.15, 1.15, 1.2, 1.2, 1.9, 1.2, 1.2, 1.2, 1.8, 1.8, 1.2]

  result = {(2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1) : 0.1,
            (2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2) : 0.05,
            (1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2) : 0.05,
            (1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2) : 0.05,
            (1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2) : 0.15,
            (1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2) : 0.2,
            (1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2) : 0.2,
            (1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2) : 0.2}

  assert impartial.randomized_allocation_from_quotas(s) == result
  '''
  #s = [1.1, 1.1, 1.3, 1.7, 1.8, 3.0]
  #assert impartial.randomized_allocation_from_quotas(s) == 0.0

  s = [0.9, 0.9, 1.2]
  assert impartial.randomized_allocation_from_quotas(s) == 0.0
  





def test_even_partition_order():
  ord = list(range(5))
  assert impartial.even_partition_order(ord, 2) == {0:[0, 1, 2], 1:[3, 4]}
  assert impartial.even_partition_order(ord, 3) == {0:[0, 1], 1:[2, 3], 2:[4]}

def test_normalize_score_matrix():
  a = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
  part = {0:[0, 1], 1:[2, 3]}

  assert (impartial.normalize_score_matrix(a, part) == \
    np.array([ [0, 0, .5, .5],
             [0, 0, .5, .5],
             [.5, .5, 0, 0],
             [.5, .5, 0, 0]])).all()

  a = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
  part = {0:[0, 1, 2, 3]}

  assert (impartial.normalize_score_matrix(a, part) == \
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])).all()

  a = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])

  assert (impartial.normalize_score_matrix(a) == \
    np.array([[0,    0.25, 0.25, 0.25, 0.25],
                [0.25, 0,    0.25, 0.25, 0.25],
                [0.25, 0.25, 0,    0.25, 0.25],
                [0.25, 0.25, 0.25, 0,    0.25],
                [0.25, 0.25, 0.25, 0.25, 0]])).all()



def test_validate_matrix():
  a = np.array([[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]])
  part = {0:[0, 1], 1:[2, 3]}

  assert (impartial.validate_matrix(a, part) == \
    np.array([[0, 0, 1, 1],
             [0, 0, 1, 1],
             [1, 1, 0, 0],
             [1, 1, 0, 0]])).all()

  a = np.array([[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]])
  part = {0:[0, 1, 2, 3]}

  assert (impartial.validate_matrix(a, part) == \
    np.array([[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]])).all()

  a = np.array([[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]])

  assert (impartial.validate_matrix(a) == \
    np.array([[0, 1, 1, 1],
             [1, 0, 1, 1],
             [1, 1, 0, 1],
             [1, 1, 1, 0]])).all()


def test_credible_subset():
  a = np.array([[1, 1, 0, 1],
                [0, 0, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 0]])

  assert impartial.credible_subset(a, 2, 3) == [2]

  ## TODO: This needs to be tested better...
  


# Testing for the dollar raffle.. basically 
# impossible...

def test_dollar_raffle_explicit():
  a = np.array([[0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

  assert impartial.dollar_raffle_explicit(a, 1, {0:[0, 1], 1:[2, 3]}, normalize=False) == [1]
  assert impartial.dollar_raffle_explicit(a, 2, {0:[0, 1], 1:[2, 3]}, normalize=False) == [1, 0]
  assert impartial.dollar_raffle_explicit(a, 3, {0:[0, 1], 1:[2, 3]}, normalize=False) == 0




