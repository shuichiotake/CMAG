# CMAG 
 
# Introduction

This package makes experimental results in the paper "A Metric Differential Privacy Mechanism for Sentence Embeddings".
 
# Requirement
 
* Python 3.10
 
Only environment under Anaconda3 (VER:conda 23.1.0:linux-64) is tested.
 
# Experiments

1. To make the data set in data

   The initial given data are sentence_embedding.pkl, sentence_index.pkl, word_embedding.pkl and word_index.pkl. The data setsentence_embedding.pkl is the 122388 by 1024 sentence embedding matrix obtained by embedding sentences contained in benchmark experiences SST, SUBJ, TREC, CR, MR, MRPC, STS, NLI, using SimCSE(sup-simcse-roberta-large). The data word_embedding.pkl is the 51304 by 1024 word embedding matrix consists of all words(# = 51304) contained in the sentences in benchmark experiences. The data set sentence_index.pkl or word_index.pkl is just the index given for each sentence or word.   
   
   To make another data set necessary for the following experiences, please execute the python file preparation_sentence.py and preparation_word.py in the directory preparation.
   The pickle data, alpha_mod.pkl, convert_num.pkl, covering.pkl, num_Delta.pkl, sigma_mat_CMAG.pkl, sigma_mat_Mahalanobis.pkl, sd_CMAG.pkl, sd_CMAG(E).pkl, sd_Laplacian.pkl, sd_MAG.pkl, will be created in data directory.  
  
2. To make data for privacy tasks

   We can compare the privacy level of each mechansim by calculating Ns and Ss, which measure the predictability and the not-predictability of the original sentence from the perturbed one.

   Please execute the python file privacy_experience.py in the directory /experience/privacy_experience/.
   The pickle data Ns_.pkl and Ss_.pkl will be created in the directory /experience/privacy_experience/results/. Also the graphs Ns_percentile.png and Ss_percentile.png will be created in the directory /experience/privacy_experience/graphs/.  

3. To make data for downstream tasks
   
   We can compare the benchmark result SST, SUBJ, TREC, CR, MR, MRPC, STS, NLI for each mechanism.

   Please execute the python file benchmark_experience.py in the directory /experience/benchmark_experience/.
   The pickle data evaluation_result_.pkl will be crated in the directory /experience/benchmark_experience/results/ and the graphs SST2.png, SUBJ.png, TREC.png, CR.png, MR.png,
   MRPC.png, STSBenchmark.png, SICKEntailment.png will be created in the directory /experience/benchmark_experience/graphs/.

   Note that the benchmark_experience.py is on the test mode in default, and please download SentEval(https://github.com/facebookresearch/SentEval) and place it under the directory /experience/benchmark/. Also, please erase # on the lines 1, 2, 5, 6, 10, 115, 116, 139, 140, 165, 166, 189, 190, 214, 215, 248--252, 266--268 in benchmark_experience.py for the actual thing.
 
# Authors
 
* Danushka Bollegala, Shuichi Otake, Tomoya Machide, Ken-ichi Kawarabayashi
 
# License
 
Apache License 2.0