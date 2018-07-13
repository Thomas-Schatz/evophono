# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:55:27 2018

@author: Thomas Schatz

Instantiate interesting sets of parameters for our models
of phonology emergence.
"""

import numpy as np
import edit_distance as ED

def word_confusion(atoms_dissimilarity, max_word_len, beta=1, atoms=None):
    """
    Get word-level confusion probability matrix
    from atomic-signals-level dissimilarities
    for all words under max_word_len length.
    
    Dissimilarities are assumed to be numbers between 0 and 1.
    
    Based on computing edit-distances using the atoms_dissimilarity matrix
    as substitution costs and obtaining probabilities by
    taking exp(-beta*ED(w1, w2)) and renormalising each row to sum to 1.
    Use larger beta get more peaked confusion probabilities
    
    Justification: for same-length words we retrieve (and it's the only way to
    retrieve?) Fletcher's sequential law. Also used in one of the Harvard guy 
    working on evolutionary dynamics paper (1999, not PNAS).
    
    Returns a list of words (based on the atoms argument if it is a list of
    strings or an arbitrary alphabetic mapping if atoms is None) along
    with the confusion probability matrix between these words such that the 
    number on row i column j, corresponds to the probability of having word
    j being received when word i was intended.
    """
    assert np.all(atoms_dissimilarity <= 1)
    assert np.all(atoms_dissimilarity >= 0)
    
    alphabet_size = atoms_dissimilarity.shape[0]  # number of atomic signals
    if atoms is None:
        atoms = [chr(97+e) for e in range(alphabet_size)]
    
    insertion_costs = np.ones(alphabet_size)
    deletion_costs = np.ones(alphabet_size)
    substitution_costs = atoms_dissimilarity
    
    # get list of possible words by iterating on possible word lengths
    all_words = []
    # at the start of iteration l, word_list will contain all words of len l-1
    word_list = ['']
    for l in range(max_word_len):
        word_list = [word+atom for word in word_list for atom in atoms]
        all_words = all_words + word_list
    
    # for each possible pair of words compute edit distance
    word_dis = np.empty((len(all_words), len(all_words)))
    for ia, wa in enumerate(all_words):
        a = np.array([atoms.index(e) for e in wa])
        for ib, wb in enumerate(all_words):           
            # there is probably a smart way to pool computations instead of
            # considering each pair of words independently, let's look at it
            # if this part of the code ever becomes a bottleneck
            b = np.array([atoms.index(e) for e in wb])
            d = ED.edit_distance(a, b, s_cost=substitution_costs,
                                 d_cost=deletion_costs, i_cost=insertion_costs)
            word_dis[ia,ib] = d
    
    confusion_probas = np.exp(-beta*word_dis)
    S = np.sum(confusion_probas, axis=1)
    confusion_probas = confusion_probas / np.tile(S, (len(all_words), 1)).T
    return all_words, confusion_probas


atoms_dis = np.array([[0, .2, .6, .8],
                      [.2, 0, .8, .6],
                      [.6, .8, 0, .3],
                      [.8, .6, .3, 0]])
atoms = ['a', 'e', 'b', 'd']

a, c = word_confusion(atoms_dis, max_word_len=2, beta=1, atoms=atoms)

#TODO
#   find how to vary beta and atom_dis such that: distribution of confusions
#   get more or less peaked, but probability of confusing sound i with any
#   other sound stays constant


