"""
Add Gaussian noise to sentence embeddings and evaluate the embeddings on downstream tasks
"""

PATH_SENTEVAL = '../../SentEval'
PATH_TO_DATA = '../../SentEval/data'

import sys
sys.path.insert(0, PATH_SENTEVAL)
import senteval

import numpy as np
import torch
import h5py
import time
import scann
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas
from tabulate import tabulate
import pprint 
import pickle
import logging
import datetime
import csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

class Eval(object):
    def __init__(self, knd, e, s_index_switch, noised_mat):
        self.knd = knd
        self.e = e
        self.s_index_switch = s_index_switch
        self.noised_mat = noised_mat

    def prepare(self, params, samples):
        return
    
    def batcher(self, params, batch): 
        s_index_switch = self.s_index_switch
        noised_mat = self.noised_mat
        sentences = [' '.join(s) for s in batch]
        try:
            nums = [int(s_index_switch[sent]) for sent in sentences]
        except KeyError:
            print(sentences)
        return noised_mat[nums]

    def evaluation(self, knd, e):
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
        
        se = senteval.engine.SE(params, self.batcher, self.prepare)
        corr_tasks = ['STSBenchmark', 'SICKEntailment']
        acc_tasks = ['SUBJ', 'SST2', 'TREC', 'MRPC', 'MR', 'CR']
        all_tasks = corr_tasks[:]
        all_tasks.extend(acc_tasks)
        res = se.eval(all_tasks)
        print(knd, e)
        s = "evaluation_result_{}_{:.2f}"
        with open("/home/otake/Sentence-DP/CMAG/experience/benchmark_experience/results/" + str(s.format(knd,e)) + ".pkl", "wb") as f:
            pickle.dump(res,f)
