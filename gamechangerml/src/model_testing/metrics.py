import numpy as np
from typing import List

## Order Unaware Metrics ##

def get_precision(true_positives: int, false_positives: int) -> float:
    '''To calculate precision@k, apply this formula to the top k results of a single query.'''
    return true_positives / (true_positives + false_positives)

def get_recall(true_positives: int, false_negatives: int) -> float:
    '''To calculate recall@k, apply this formula to the top k results of a single query.'''
    return true_positives / (true_positives + false_negatives)

def get_f1(precision: float, recall: float) -> float:
    '''To calculate f1@k, use precision@k and recall@k to the top k results of a single query.'''
    return (2 * ((precision * recall) / (precision + recall)))

def get_accuracy(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    return ((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives))

## Order-Aware Metrics ##

def reciprocal_rank(ranked_results: List[str], expected: List[str]) -> float:
    '''
    Calculates the reciprocal of the rank of the first correct relevant result (returns single value from 0 to 1).
    Note: This algorithm assumes any result NOT in a pre-defined list of expected results is not useful.
    '''
    first_relevant_rank = 0 # if no relevant results show up, the RR will be 0
    count = 1
    for i in ranked_results: # list in order of rank
        if i in expected:
            first_relevant_rank = count
            break
        else:
            count += 1
    
    return (1 / first_relevant_rank)

def get_MRR(reciprocal_ranks: List[float]) -> float:
    '''Takes list of reciprocal rank scores for each search and averages them.'''
    return reciprocal_ranks / len(reciprocal_ranks)

def average_precision(ranked_results: List[str], expected: List[str]) -> float:
    '''Averages the precision@k for each value of k in a sample of results (returns single value from 0 to 1)'''

    true_positives = 0
    false_positives = 0
    precision_scores = []
    for i in ranked_results:
        if i in expected:
            true_positives += 1
        else:
            false_positives += 1
        precision_scores.append(get_precision(true_positives, false_positives))
    
    return (np.mean(precision_scores) / true_positives)

def get_MAP(average_precision_scores: List[float]) -> float:
    '''Takes list of average precision scores and averages them.'''
    return average_precision_scores / len(average_precision_scores)