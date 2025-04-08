#PATH_SENTEVAL = '../SentEval'
#PATH_TO_DATA = '../SentEval/data'

import sys
#sys.path.insert(0, PATH_SENTEVAL)
#import senteval

sys.path.append("../../packages/")

#from evaluation import *
import functions as func
import itertools
import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager
from multiprocessing import Pool
import numpy as np
import pickle
import pylab
from scipy.stats import norm
import torch
from UT import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# if we have a GPU use it, otherwise CPU. GPU is recommended as it is x8 faster!
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(torch.cuda.is_available())

ut = UT()
    
with open("../../data/sentence_embedding.pkl", "rb") as f:
    s_embd = pickle.load(f)
    
with open("../../data/sentence_index.pkl", "rb") as f:
    s_index = pickle.load(f)
    
with open("../../data/sentence_embedding_mean.pkl", "rb") as f:
    s_embd_mean = pickle.load(f)

with open("../../data/covering.pkl", "rb") as f:
    covering = pickle.load(f)
    
with open("../../data/sigma_mat_CMAG.pkl", "rb") as f:
    sigma_mat_CMAG = pickle.load(f)
    
with open("../../data/sigma_mat_Mahalanobis.pkl", "rb") as f:
    sigma_mat_Mahalanobis = pickle.load(f)
    
with open("../../data/sd_CMAG.pkl", "rb") as f:
    sd_CMAG = pickle.load(f)
    
with open("../../data/sd_MAG.pkl", "rb") as f:
    sd_MAG = pickle.load(f)
    
with open("../../data/sd_CMAG(E).pkl", "rb") as f:
    sd_CMAG_E = pickle.load(f)
    
with open("../../data/sd_Laplacian.pkl", "rb") as f:
    sd_Laplacian = pickle.load(f)
    
with open("../../data/alpha.pkl", "rb") as f:
    alpha = pickle.load(f)
    
with open("../../data/convert_num.pkl", "rb") as f:
    convert_num = pickle.load(f)
    
with open("../../data/num_Delta.pkl", "rb") as f:
    num_Delta = pickle.load(f)

sigma_mat_dict = {"CMAG": sigma_mat_CMAG, "MAG": sigma_mat_Mahalanobis, "Mahalanobis": sigma_mat_Mahalanobis, "Laplacian": np.identity(1024)}
X = dict()

for i in covering.keys():
    D = {j : i for j in covering[i]}
    X = X | D   

sd_dict = {"CMAG":sd_CMAG, "MAG": sd_MAG, "CMAG(E)": sd_CMAG_E, "Laplacian": sd_Laplacian} 
parameters = [8*i/5 for i in range(1,26)]
residue = {e : round(5*e/8) for e in parameters}
noised_mat_dict = dict()

#Calculate noised matrix for each mechansim.

#for CMAG  

s1 = "noised_mat_CMAG_{:.2f}"

def noised_mat_CMAG(e):
    noi_vecs = []
    for i in range(0,122388):
        sd = sd_dict["CMAG"][e][i]
        sigma_mat = sigma_mat_dict["CMAG"][X[i]]
        np.random.seed(9*(50*i + residue[e]))
        noi_vec = np.dot(np.random.normal(0,sd,1024),sigma_mat)
        noi_vecs.append(noi_vec)
    noi_mat = np.vstack(noi_vecs)
    res_mat = s_embd + noi_mat
    print("CMAG", e)
    return (e,res_mat)
        
def main():
    with Pool(32) as pool:
        noised_mat_tuple = pool.map(noised_mat_CMAG, parameters)
    return dict(noised_mat_tuple)

#if __name__ == "__main__":
#    noised_mat_dict["CMAG"] = main()

#for CMAG(E)

s2 = "noised_mat_CMAG(E)_{:.2f}"

def noised_mat_CMAG_E(e):
    noi_vecs = []
    for i in range(0,122388):
        sd = sd_dict["CMAG(E)"][e][i] 
        np.random.seed(9*(50*i + residue[e]))
        noi_vec = np.random.normal(0,sd,1024)
        noi_vecs.append(noi_vec)
    noi_mat = np.vstack(noi_vecs)
    res_mat = s_embd + noi_mat
    print("CMAG(E)", e)
    return (e,res_mat)
        
def main():
    with Pool(32) as pool:
        noised_mat_tuple = pool.map(noised_mat_CMAG_E, parameters)
    return dict(noised_mat_tuple)

#if __name__ == "__main__":
#    noised_mat_dict["CMAG(E)"] = main()
    
#for MAG    

s3 = "noised_mat_MAG_{:.2f}"

def noised_mat_MAG(e):
    noi_vecs = []
    sd = sd_dict["MAG"][e]
    sd = max(sd,0.00000000000000000001)
    sigma_mat = sigma_mat_dict["MAG"]
    for i in range(0,122388):
        np.random.seed(9*(50*i + residue[e]))
        noi_vec = np.dot(np.random.normal(0,sd,1024),sigma_mat)
        noi_vecs.append(noi_vec)
    noi_mat = np.vstack(noi_vecs)
    res_mat = s_embd + noi_mat
    print("MAG", e)
    return(e,res_mat)
        
def main():
    with Pool(32) as pool:
        noised_mat_tuple = pool.map(noised_mat_MAG, parameters)
    return dict(noised_mat_tuple)

#if __name__ == "__main__":
#    noised_mat_dict["MAG"] = main()
    
#for Mahalanobis

s4 = "noised_mat_Mahalanobis_{:.2f}"

def noised_mat_Mahalanobis(e):
    noi_vecs = []
    sigma_mat = sigma_mat_dict["Mahalanobis"]
    for i in range(0,122388):
        np.random.seed(9*(50*i + residue[e]))
        noi_vec = func.Mahalanobis(e,sigma_mat)
        noi_vecs.append(noi_vec)
    noi_mat = np.vstack(noi_vecs)
    res_mat = s_embd + noi_mat
    print("Mahalanobis", e)
    return(e,res_mat)
        
def main():
    with Pool(32) as pool:
        noised_mat_tuple = pool.map(noised_mat_Mahalanobis, parameters)
    return dict(noised_mat_tuple)

#if __name__ == "__main__":
#    noised_mat_dict["Mahalanobis"] = main()

#for Laplacian    

s5 = "noised_mat_Laplacian_{:.2f}"

def noised_mat_Laplacian(e):
    noi_vecs = []
    sd = sd_dict["Laplacian"][e]
    sigma_mat = sigma_mat_dict["Laplacian"]
    for i in range(0,122388):
        np.random.seed(9*(50*i + residue[e]))
        noi_vec = func.Mahalanobis(sd,sigma_mat)
        noi_vecs.append(noi_vec)
    noi_mat = np.vstack(noi_vecs)
    res_mat = s_embd + noi_mat
    print("Laplacian", e)
    return(e,res_mat)
        
def main():
    with Pool(32) as pool:
        noised_mat_tuple = pool.map(noised_mat_Laplacian, parameters)
    return dict(noised_mat_tuple)

#if __name__ == "__main__":
#    noised_mat_dict["Laplacian"] = main()
    
#for NADP

s6 = "noised_mat_NADP_{:.2f}"

def noise_vec(para,i,j,m):
    print("noise_vec", "NADP", para, i, m)
    np.random.seed(m + 100* residue[para] + 5000*j + 32000*i)
    sd = alpha[para]*num_Delta[j]
    sd = max(sd,0.00000000000000000001)
    noi_vec = np.random.normal(0,sd,1024)
    return noi_vec

def noised_vec(i,e,m):
    if len(convert_num[i]) != 0:
        noi_vec_aveg = np.mean(np.array([noise_vec(e,i,j,m) for j in convert_num[i]]),axis=0)  
    else:
        noi_vec_aveg = np.random.normal(0,1000,1024)
    return (i, noi_vec_aveg)

def noised_mat_NADP(e):
    domain = itertools.product(range(0,122388), [e], [0])
    def perturb():
        with Pool(32) as pool:
            noi_vec_list = pool.starmap(noised_vec, domain) 
        return dict(noi_vec_list)
    if __name__ == "__main__":
        noi_vecs = perturb()
    noi_mat = np.array([noi_vecs[i] for i in range(0,122388)])     
    res_mat = s_embd_mean + noi_mat
    return res_mat

#matrix = dict()
#for e in parameters:
#    print("NADP",e)
#    matrix[e] = noised_mat_NADP(e)
#noised_mat_dict["NADP"] = matrix
    
#for no_noise

noised_mat_dict["no_noise"] = {e:s_embd for e in parameters}    
    
#Execute benchmark experience.    

knd = ["CMAG","MAG","Mahalanobis","Laplacian","CMAG(E)","NADP","no_noise"]

def evaluation(k,e):
    ev = Eval(k, e, s_index, noised_mat_dict[k][e])
    ev.evaluation(k,e)

#for k in knd:
#    for e in parameters:
#        evaluation(k,e)

#Generate graphs for benchmark experiences.

tasks = ['SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'STSBenchmark', "MR", "CR"]
task_method = {'SUBJ' : 'acc', 'SST2' : 'acc', 'TREC' : 'acc', 'MRPC' : 'acc', 'SICKEntailment' : 'acc', 'STSBenchmark': 'spearman', 'MR':'acc', 'CR':'acc'}
result = {'SUBJ':{}, 'SST2':{}, 'TREC':{}, 'MRPC':{}, 'SICKEntailment':{}, 'STSBenchmark':{}, "MR":{}, "CR":{}}
color = {"CMAG":"red","MAG":"blue","Mahalanobis":"orange","Laplacian":"grey","CMAG(E)":"green","NADP":"cyan","no_noise":"black"}
label = {"CMAG":"CMAG","MAG":"MAG","Mahalanobis":"Mahalanobis","Laplacian":"Laplacian","CMAG(E)":"CMAG(E)","NADP":"NADP","no_noise":"no_noise"}
tp = {'SUBJ' : 100, 'SST2' : 100, 'TREC' : 100, 'MRPC' : 100, 'MR' : 100, 'CR' : 100, 'SICKEntailment' : 100, 'STSBenchmark' : 1.0}
bm = {'SUBJ' : 0, 'SST2' : 0, 'TREC' : 0, 'MRPC' : 0, 'MR' : 0, 'CR' : 0, 'SICKEntailment' : 0, 'STSBenchmark' : -1.0}
y_label = {'SUBJ' : 'accuracy', 'SST2' : 'accuracy', 'TREC' : 'accuracy', 'MRPC' : 'accuracy', 'MR' : 'accuracy', 'CR' : 'accuracy',
           'SICKEntailment' : 'accuracy', 'STSBenchmark' : 'correlation'}
s_6 = "evaluation_result_{}_{:.2f}"
s_7 = "{}.png" 

parameters = [8*i/5 for i in range(1,26)]

for t in tasks:
    for k in knd:
        result_tuples = []
        for para in parameters:
            e = para
            with open("./results/" + str(s_6.format(k,e)) + ".pkl", "rb") as f:
                result_all = pickle.load(f)
            result_task = result_all[t][task_method[t]]
            result_tuples.append(((e,result_task)))
        result[t][k] = result_tuples

for t in tasks:
    DIR_root = "./graphs/"
    plt.rcParams["font.size"] = 24
    pylab.figure( num=None, figsize=(13, 13) )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.subplot(1,1,1)

    plt.xlim(left=min(parameters), right=max(parameters));
    plt.xlabel("ε",fontsize = 32)
    plt.ylim(bottom=bm[t], top=tp[t]);
    plt.ylabel(y_label[t],fontsize = 32)

    for k in knd:
        if k == "no_noise":
            result_dict = dict(result[t][k])
            result_values = list(result_dict.values())
            plt.plot(parameters,result_values,color=color[k],linewidth=2,linestyle="dotted")
        else:
            result_dict = dict(result[t][k])
            result_values = list(result_dict.values())
            plt.plot(parameters,result_values,color=color[k],linewidth=2,linestyle="solid",label=label[k])
    
    plt.legend(fontsize=16, loc='best')

    fle = DIR_root + str(s_7.format(t))
    print("Info in method(): save png; fle =", fle )
    plt.savefig( fle )
    plt.clf()
    plt.show()
