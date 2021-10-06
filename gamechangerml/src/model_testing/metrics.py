import numpy as np
import collections
from typing import List

from gamechangerml.src.utilities.text_utils import get_tokens

## Order Unaware Metrics ##
def get_precision(true_positives: int, false_positives: int) -> float:
    '''To calculate precision@k, apply this formula to the top k results of a single query.'''
    try:
        return np.round((true_positives / (true_positives + false_positives)), 3)
    except ZeroDivisionError:
        return 0
        
def get_recall(true_positives: int, false_negatives: int) -> float:
    '''To calculate recall@k, apply this formula to the top k results of a single query.'''
    try:
        return np.round((true_positives / (true_positives + false_negatives)), 3)
    except ZeroDivisionError:
        return 0

def get_f1(precision: float, recall: float) -> float:
    '''To calculate f1@k, use precision@k and recall@k to the top k results of a single query.'''
    try:
        result = 2 * (precision * recall) / (precision + recall)
        return np.round(result, 3)
    except ZeroDivisionError:
        return 0

def compute_QA_f1(a_gold, a_pred):
    '''
    Modified function from SQuAD evaluation to compute F1 on tokens of expected/predicted answers.
    Original function: https://rajpurkar.github.io/SQuAD-explorer/
    '''
    gold_tokens = get_tokens(a_gold)
    pred_tokens = get_tokens(a_pred)
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0
    precision = get_precision(true_positives=num_same, false_positives=(len(pred_tokens) - num_same))
    recall = get_recall(true_positives=num_same, false_negatives=(len(gold_tokens) - num_same))
    f1 = get_f1(precision=precision, recall=recall)
    return f1

def get_accuracy(true_positives: int, true_negatives: int, total: int) -> float:
    try:
        return np.round(((true_positives + true_negatives) / total), 3)
    except ZeroDivisionError:
        return 0

## Order-Aware Metrics ##

def reciprocal_rank(ranked_results: List[str], expected: List[str]) -> float:
    '''
    Calculates the reciprocal of the rank of the first correct relevant result (returns single value from 0 to 1).
    '''
    first_relevant_rank = 0 # if no relevant results show up, the RR will be 0
    count = 1
    for i in ranked_results: # list in order of rank
        if i in expected:
            first_relevant_rank = count
            break
        else:
            count += 1
    
    if first_relevant_rank > 0:
        return np.round((1 / first_relevant_rank), 3)
    else:
        return 0

def get_MRR(reciprocal_ranks: List[float]) -> float:
    '''Takes list of reciprocal rank scores for each search and averages them.'''
    return np.round(np.mean(reciprocal_ranks), 3)

def average_precision(ranked_results: List[str], expected: List[str]) -> float:
    '''
    Averages the precision@k for each value of k in a sample of results (returns single value from 0 to 1).
    '''

    true_positives = 0
    false_positives = 0
    precision_scores = []
    remainder = len(expected) # stop calculating precision after we find all our expected results
    for i in ranked_results:
        if remainder > 0:
            if i in expected:
                true_positives += 1
                remainder -= 1
            else:
                false_positives += 1
        else:
            break
        precision_scores.append(get_precision(true_positives, false_positives))
    
    if true_positives > 0:
        return np.round((np.sum(precision_scores) / true_positives), 3)
    else:
        return 0

def get_MAP(average_precision_scores: List[float]) -> float:
    '''Takes list of average precision scores and averages them.'''
    return np.round(np.mean(average_precision_scores), 3)