import sys
sys.path.append("../../packages/")
a
import functions as func
import itertools
import matplotlib.pyplot as plt
from matplotlib import font_manager
from multiprocessing import Pool
import numpy as np
import pickle
import pylab

# -- load

with open("../../data/sentence_embedding.pkl", "rb") as f:
    s_embd = pickle.load(f)

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

num_of_sent = s_embd.shape[0]
sigma_mat_dict = {"CMAG": sigma_mat_CMAG, "MAG": sigma_mat_Mahalanobis, "Mahalanobis": sigma_mat_Mahalanobis, "Laplacian": np.identity(1024)}
sample_sent = list(range(0,num_of_sent))

X = dict()

for i in covering.keys():
    D = {j : i for j in covering[i]}
    X = X | D

sd_dict = {"CMAG":sd_CMAG, "MAG": sd_MAG, "CMAG(E)": sd_CMAG_E, "Laplacian": sd_Laplacian} 
parameters = [8*i/5 for i in range(1,26)]
residue = {e : round(5*e/8) for e in parameters}
nums = {"CMAG" : 0, "MAG" : 0, "Mahalanobis" : 0, "Laplacian" : 0, "CMAG(E)": 0}
s1 = "Ns_{}"
s2 = "Ns_{}_{:.2f}"

#Add noise to all sentences 100 times.

#for CMAG

knd = ["CMAG"]

result_CMAG = dict()

for k in knd:
    for para in parameters:
        e = para
        def Nearest(i):
            L = list()
            sd = sd_dict[k][e][i] 
            print(str(s1.format(k)), e, i, sd)
            sigma_mat = sigma_mat_dict["CMAG"][X[i]]
            embds_100 = s_embd[covering[X[i]]]
            for m in range(0,100):
                np.random.seed(m + 100* residue[para] + 5000*i + nums[k])
                per_vec = s_embd[i] + np.dot(np.random.normal(0,sd,1024),sigma_mat)
                A = embds_100 - per_vec
                B = A*A
                C = B.sum(1)
                n = np.argsort(C)[0]
                nearest = covering[X[i]][n]
                L.append(nearest)
            return (i, L)
        def main():
            with Pool(32) as pool:
                X = pool.map(Nearest,range(0,num_of_sent))
            return X
        if __name__ == "__main__":
            Y = main()
        result_CMAG_para = dict(Y)
        result_CMAG[e] = result_CMAG_para      

#for CMAG(E)

knd = ["CMAG(E)"]

result_CMAG_E = dict()

for k in knd:
    for para in parameters:
        e = para
        def Nearest(i):
            L = list()
            print(str(s1.format(k)), e, i)
            sd = sd_dict[k][e][i] 
            embds_100 = s_embd[covering[X[i]]]
            for m in range(0,100):
                np.random.seed(m + 100* residue[para] + 5000*i + nums[k])
                per_vec = s_embd[i] + np.random.normal(0,sd,1024)
                A = embds_100 - per_vec
                B = A*A
                C = B.sum(1)
                n = np.argsort(C)[0]
                nearest = covering[X[i]][n]
                L.append(nearest)
            return (i, L)
        def main():
            with Pool(32) as pool:
                X = pool.map(Nearest,range(0,num_of_sent))
            return X
        if __name__ == "__main__":
            Y = main()
        result_CMAG_E_para = dict(Y)
        result_CMAG_E[e] = result_CMAG_E_para
                       
#for MAG

knd = ["MAG"]

result_MAG = dict()

for k in knd:
    sigma_mat = sigma_mat_dict[k]
    for para in parameters:
        e = para
        sd = sd_dict[k][e]
        def Nearest(i):
            L = list()
            print(str(s1.format(k)), e, i)
            embds_100 = s_embd[covering[X[i]]]
            for m in range(0,100):
                np.random.seed(m + 100* residue[para] + 5000*i + nums[k])
                per_vec = s_embd[i] + np.dot(np.random.normal(0,sd,1024),sigma_mat)
                A = embds_100 - per_vec
                B = A*A
                C = B.sum(1)
                n = np.argsort(C)[0]
                nearest = covering[X[i]][n]
                L.append(nearest)
            return (i, L)
        def main():
            with Pool(32) as pool:
                X = pool.map(Nearest,range(0,num_of_sent))
            return X
        if __name__ == "__main__":
            Y = main()
        result_MAG_para = dict(Y)
        result_MAG[e] = result_MAG_para

#for Mahalanobis

knd = ["Mahalanobis"]

result_Mahalanobis = dict()

for k in knd:
    sigma_mat = sigma_mat_dict[k]
    for para in parameters:
        e = para
        def Nearest(i):
            L = list()
            print(str(s1.format(k)), e, i)
            embds_100 = s_embd[covering[X[i]]]
            for m in range(0,100):
                np.random.seed(m + 100* residue[para] + 5000*i + nums[k])
                per_vec = s_embd[i] + func.Mahalanobis(e,sigma_mat)
                A = embds_100 - per_vec
                B = A*A
                C = B.sum(1)
                n = np.argsort(C)[0]
                nearest = covering[X[i]][n]
                L.append(nearest)
            return (i, L)
        def main():
            with Pool(32) as pool:
                X = pool.map(Nearest,range(0,num_of_sent))
            return X
        if __name__ == "__main__":
            Y = main()
        result_Mahalanobis_para = dict(Y)
        result_Mahalanobis[e] = result_Mahalanobis_para

#for Laplacian

knd = ["Laplacian"]

result_Laplacian = dict()

for k in knd:
    sigma_mat = sigma_mat_dict[k]
    for para in parameters:
        e = para
        sd = sd_dict[k][e]
        def Nearest(i):
            L = list()
            print(str(s1.format(k)), e, i)
            embds_100 = s_embd[covering[X[i]]]
            for m in range(0,100):
                np.random.seed(m + 100* residue[para] + 5000*i + nums[k])
                per_vec = embd_mat[i] + func.Mahalanobis(sd,sigma_mat)
                A = embds_100 - per_vec
                B = A*A
                C = B.sum(1)
                n = np.argsort(C)[0]
                nearest = covering[X[i]][n]
                L.append(nearest)
            return (i, L)
        def main():
            with Pool(32) as pool:
                X = pool.map(Nearest,range(0,num_of_sent))
            return X
        if __name__ == "__main__":
            Y = main()
        result_Laplacian_para = dict(Y)
        result_Laplacian[e] = result_Laplacian_para

#Calculate N_s and S_s.
        
result = {"CMAG":result_CMAG, "MAG":result_MAG, "Mahalanobis":result_Mahalanobis, "Laplacian":result_Laplacian, "CMAG(E)":result_CMAG_E}
knd = ["CMAG","MAG","Mahalanobis","Laplacian","CMAG(E)"]

#Step_1 Calculate N_s.

result_Ns = {"CMAG":dict(),"MAG":dict(),"Mahalanobis":dict(),"Laplacian":dict(),"CMAG(E)":dict()}

for k in knd:
    for para in parameters:
        e = para
        data = result[k][e]
        def Ns(i):
            M = data[i]
            pr = len([x for x in M if x == i])
            print("Ns",k,e,i)
            return pr
        def main():
            with Pool(32) as pool:
                X = pool.map(Ns,range(0,num_of_sent))
            return sorted(X)
        if __name__ == "__main__":
            Y = main()
        Ns_para = {5 : Y[round(num_of_sent*5/100)-1], 50 : Y[round(num_of_sent*50/100)-1], 95 : Y[round(num_of_sent*95/100)-1]}
        result_Ns[k][e] = Ns_para
        with open("./results/" + str(s2.format(k,e)) + ".pkl", "wb") as f:
            pickle.dump(Ns_para,f)
            
#Step_2 Calculate S_s.

result_Ss = {"CMAG":dict(),"MAG":dict(),"Mahalanobis":dict(),"Laplacian":dict(),"CMAG(E)":dict()}

for k in knd:
    for para in parameters:
        e = para
        data = result[k][e]
        def Ss(i):
            M = sorted(data[i])
            N = list(itertools.groupby(M))
            print("Ss",k,e,i)
            return len(N)
        def main():
            with Pool(32) as pool:
                X = pool.map(Ss,range(0,num_of_sent))
            return sorted(X)
        if __name__ == "__main__":
            Y = main()
        Ss_para = {5 : Y[round(num_of_sent*5/100)-1], 50 : Y[round(num_of_sent*50/100)-1], 95 : Y[round(num_of_sent*95/100)-1]}
        result_Ss[k][e] = Ss_para
        with open("./results/" + str(s4.format(k,e)) + ".pkl", "wb") as f:
            pickle.dump(Ss_para,f)
            
#Generate graphs for Ns and Ss.

color = {"CMAG":"red","MAG":"blue","Mahalanobis":"orange","Laplacian":"grey","CMAG(E)":"green"}
label = {"CMAG":"CMAG","MAG":"MAG","Mahalanobis":"Mahalanobis","Laplacian":"Laplacian","CMAG(E)":"CMAG(E)"}
per = [5, 50, 95]

#Step1 Generate graphs for Ns.

s = "Ns_{}th_percentile.png"
Ns_per_dict = {5: {}, 50: {}, 95: {}}

for p in per:
    for k in knd:
        Ns_list = list()
        for para in parameters:
            e = para
            Ns_dict = result_Ns[k][e]
            Ns = Ns_dict[p]
            Ns_list.append(Ns)
        Ns_per_dict[p][k] = Ns_list

for p in per:
    DIR_root = "./graphs/"

    plt.rcParams["font.size"] = 24
    pylab.figure( num=None, figsize=(13, 13) )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.subplot(1,1,1)

    plt.xlim(left=min(parameters), right=max(parameters));
    plt.xlabel("ε",fontsize = 32)
    plt.ylim(bottom = 0, top = 100);
    plt.ylabel("%",fontsize = 32)

    for k in knd:
        result_values = Ns_per_dict[p][k]
        plt.plot(parameters,result_values,color=color[k],linewidth=2,linestyle="solid",label=label[k])
    
    plt.legend(fontsize=16, loc='best')

    fle = DIR_root + str(s.format(p))
    print("Info in method(): save png; fle =", fle )
    plt.savefig( fle )
    plt.clf()
    plt.show()   
    
#Step_2 Generate the graph for Ss.

s = "Ss_{}th_percentile.png"
Ss_per_dict = {5: {}, 50: {}, 95: {}}

for p in per:
    for k in knd:
        Ss_list = list()
        for para in parameters:
            e = para
            Ss_dict = result_Ss[k][e]
            Ss = Ss_dict[p]
            Ss_list.append(Ss)
        Ss_per_dict[p][k] = Ss_list
        
for p in per:
    DIR_root = "./graphs/"

    plt.rcParams["font.size"] = 24
    pylab.figure( num=None, figsize=(13, 13) )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.subplot(1,1,1)

    plt.xlim(left=min(parameters), right=max(parameters));
    plt.xlabel("ε",fontsize = 32)
    plt.ylim(bottom = 0, top = 100);
    plt.ylabel("%",fontsize = 32)

    for k in knd:
        result_values = Ss_per_dict[p][k]
        plt.plot(parameters,result_values,color=color[k],linewidth=2,linestyle="solid",label=label[k])
    
    plt.legend(fontsize=16, loc='best')

    fle = DIR_root + str(s.format(p))
    print("Info in method(): save png; fle =", fle )
    plt.savefig( fle )
    plt.clf()
    plt.show()   
