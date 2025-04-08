import math
from mpmath import *
from multiprocessing import Pool
import numpy as np
import pickle
import scipy.linalg
from scipy.stats import norm
from sympy import *
a
# -- load

with open("../data/sentence_embedding.pkl", "rb") as f:
    s_embd = pickle.load(f)

num_of_sent = s_embd.shape[0]
q = num_of_sent // 1000

### --- Create covering ---

##Create top_100 for each sentence contained in the set of sentences S (|S| = 122388).
    
top_100 = dict()
    
def get_top_100(s):
    A = s_embd - s_embd[s]
    B = A*A
    C = B.sum(1)
    L = np.argsort(C)[0:100]
    return(L)    

def main(M):
    with Pool(32) as pool:
        X = pool.map(get_top_100,M)
    return X

for j in range(0,q):
    print(j)
    result = main(range(1000*j,1000*(j+1)))
    Y = {result[s][0] : result[s] for s in range(0,len(result))}
    top_100 = top_100 | Y
        
result = main(range(q*1000,num_of_sent))
Y = {result[s][0] : result[s] for s in range(0,len(result))}
top_100 = top_100 | Y

##Create a coveirng of S (Divide S (|S| = 122388) into two types of disjoint sets A_1 A_2, ..., A_1212 and B1, B2, ..., B_12 (|A_i| = 100, |B_j| = 99))

#step_1  Divide S into disjoint sets of sentences. Let R be an empty set. Then, choose any sentence s in S and put D = top_100[s] - R, R = R | D, S = S - R. Repeat this process while |D| = 100. Next, choose any sentence s in S and put D = top_100[s] - R, R = R | D, S = S -R. Repeat this process while |D| = 99 ...

S = set(range(0,num_of_sent))
R = set()
disjoint_sets = {i:dict() for i in range(1,101)}

for i in reversed(range(1,101)):
    print(i)
    S = S - R
    L = sorted(list(S))
    n = 0
    while n < len(L):
        s = L[n]
        if len(set(top_100[s]) & R) == 100-i:
            D = set(top_100[s]) - R
            disjoint_sets[i][s] = D
            R = R | D
            n = 0
            S = S - R
            L = sorted(list(S))
        else:
            n = n + 1

#step_2  Let rem be the union of all disjoint sets T_1, T_2, ..., T_k such that |T_i| <= 38. Then, choose any sentence s in rem and search the 'nearest' disjoint set T such that 39 <= |T| < 99. Then, put T = T | {s}. Repeat this process for all sentences s in rem.  

q = num_of_sent // 100
r = num_of_sent % 100
s = 99 - r
a = q - s
b = s + 1
TNC = a + b

k = 0
l = 101

while k < TNC:
    l = l - 1
    m = k
    k = k + len(disjoint_sets[l])   
    
L = list()
S = set()
Y = dict()

for i in reversed(range(l,101)):
    L = L + list(disjoint_sets[i].keys())
    for j in list(disjoint_sets[i].keys()):
        S = S | set(disjoint_sets[i][j])
        Y[j] = i

if k > TNC:
    i = l - 1
    n = TNC - m
    L = L + list(disjoint_sets[i].keys())[0:n]
    for j in list(disjoint_sets[i].keys())[0:n]:
        S = S | set(disjoint_sets[i][j])
        Y[j] = i
    
rem = sorted(list(set(range(0,num_of_sent)) - S))
L = sorted(L)

dict_1 = {i:list(disjoint_sets[i].keys()) for i in reversed(range(1,101))}
X = dict()

for i in reversed(range(1,101)):
    for n in dict_1[i]:
        X[n] = disjoint_sets[i][n]
               
for n, i in enumerate(rem):
    print(n)
    if len(disjoint_sets[100]) < a:
        L1 = sorted(list(disjoint_sets[100].keys()))
        L2 = sorted([x for x in L if x not in L1])
    else:
        L11 = sorted(list(disjoint_sets[100].keys()))
        L12 = sorted(list(disjoint_sets[99].keys()))
        L2 = sorted([x for x in L if x not in L11 + L12])
    cent = []
    for j in L2:
        A = sorted(list(X[j]))
        M = s_embd[A]
        c = (1/M.shape[0])*M.sum(0)
        cent.append(c)
    B = np.vstack(cent)
    C = B - s_embd[i]
    D = C*C
    E = D.sum(1)
    l = np.argsort(E)[0]
    m = L2[l]
    disjoint_sets[Y[m]+1][m] = disjoint_sets[Y[m]][m] | {i}
    del disjoint_sets[Y[m]][m]    
    Y[m] = Y[m] + 1

#Step 3  Create a coveirng of S and save it.

covering = dict()

for i in disjoint_sets[100].keys():
    covering[i] = sorted(list(disjoint_sets[100][i]))
    
for i in disjoint_sets[99].keys():
    covering[i] = sorted(list(disjoint_sets[99][i]))
            
with open("../data/covering.pkl", "wb") as f:
    pickle.dump(covering,f)

#Create distance matrix and sigma matrix for CMAG mechanism.
print("create distance matrix and sigma matrix for CMAG...")

dim = s_embd.shape[1] 
sigma_mat_CMAG = dict()
distance_mat_CMAG = dict()

for m,k in enumerate(covering.keys()):
    M = s_embd[covering[k]]
    W = np.cov(M.T)
    w,P = scipy.linalg.eigh(W)
    n = np.linalg.matrix_rank(np.diag(w))
    w1 = np.hstack([pow(10,-7)*np.ones(dim-n),np.zeros(n)])
    s_c_mat = P@np.diag(w+w1)@np.linalg.inv(P)
    s_s_c_mat = (dim/np.trace(s_c_mat))*s_c_mat
    v_s,B_s = np.linalg.eigh(s_s_c_mat)
    N = np.linalg.inv(s_s_c_mat)
    v_d,B_d = np.linalg.eigh(N)
    Y_s = B_s @ np.diag(np.sqrt(v_s)) @ B_s.T
    Y_d = B_d @ np.diag(np.sqrt(v_d)) @ B_d.T
    sigma_mat_CMAG[k] = Y_s
    distance_mat_CMAG[k] = Y_d
    print(m,k,"CMAG")

#Create sigma matrix for Mahalanobis mechanism.
print("create sigma matrix for Mahalanobis...")

W = s_embd.T
s_c_mat = np.cov(W)
s_s_c_mat = (dim/np.trace(s_c_mat))*s_c_mat
sigma_mat_Mahalanobis = scipy.linalg.sqrtm(s_s_c_mat)

with open("../data/sigma_mat_Mahalanobis.pkl", "wb") as f:
    pickle.dump(sigma_mat_Mahalanobis,f)

with open("../data/sigma_mat_CMAG.pkl", "wb") as f:
    pickle.dump(sigma_mat_CMAG,f)

#Caluculate the distance of the farthest pair of sentences for each disjoint set constituting the covering for CMAG mechanism and CMAG(E) mechanism.
print("calculate distances...")

X = dict()

for i in covering.keys():
    D = {j : i for j in covering[i]}
    X = X | D
    
def most_distant(i,k):
    if k == "CMAG":
        M = distance_mat_CMAG[X[i]]
        A = np.dot(s_embd[list(covering[X[i]])] - s_embd[i],M)
    else:
        A = s_embd[list(covering[X[i]])] - s_embd[i]
    B = A*A
    C = B.sum(1)
    n = np.argsort(C)[-1]
    d = pow(C[n],1/2)
    return d

dist = {"CMAG":dict(),"CMAG(E)":dict()}

for k in ["CMAG","CMAG(E)"]:
    for m,i in enumerate(covering.keys()):
        L = [most_distant(j,k) for j in covering[i]]
        d = max(L)
        dist[k][i] = d
        print(k,m,d)

#Calculate sd.

parameters = [8*i/5 for i in range(1,26)]

delta_CMAG = {1.6: 15.66542651977551, 3.2: 20.01747165287146, 4.8: 25.83659224665971, 6.4: 26.819801650604884, 8.0: 25.569318661857587, 
              9.6: 29.07037636138774, 11.2: 26.521861143800002, 12.8: 26.624481485433048, 14.4: 26.873302600851993, 16.0: 24.203023989205743, 
              17.6: 23.466776454100696,19.2: 23.732512747831656, 20.8: 21.863579029471428, 22.4: 21.80779667232261, 24.0: 22.149700122777986, 
              25.6: 21.012816216384095, 27.2: 19.811235092091326, 28.8: 19.247604271127205, 30.4: 19.614138653909876, 32.0: 19.894944859124895, 
              33.6: 19.13617119020461, 35.2: 19.133283900942832, 36.8: 18.51659119360555, 38.4: 18.490623868597428, 40.0: 18.57986472900284}

delta_CMAG_E = {1.6: 488.8494696814043095616081721394386007990866609798, 3.2: 635.25536100178704216769975235496127448177713374308, 
                 4.8: 801.72791818584496195558114635513449988523931741124, 6.4: 822.25769297738638905811423829702146693296162864152, 
                 8.0: 824.71365581593900260481054566079342352552111057397, 9.6: 845.72341362427170700701846600156385356759416699423, 
                 11.2: 815.93342962304885149505340581189505061721469196011, 12.8: 844.71015321540896228259266492705057749568283534474, 
                 14.4: 818.22787069944357568436845988092280520812421306676, 16.0: 767.40164608137948693677671158080602966756787449117, 
                 17.6: 787.03732125157287692856372697577444736619936023962, 19.2: 798.31282144555417708961622915266571319203077799936, 
                 20.8: 795.67368369844643192590889625840463519718898075538, 22.4: 780.25401723431991618936015646168850394268367499438, 
                 24.0: 799.62315229273992779779747982335892150321829129972, 25.6: 806.26898135410409389877039592993492266468211699392, 
                 27.2: 804.16888069582366512267773510243534189987276612657, 28.8: 831.05908130845175939805430741084320242473661310965, 
                 30.4: 843.04162877336883340091755933443866778606022081114, 32.0: 839.85322203020868380821077839612699158823529999028, 
                 33.6: 867.41788668367182802621586419517983299594607343759, 35.2: 868.12461078315733642613332325051979143017867735108, 
                 36.8: 824.3104056871498270745436525263120760220245095789, 38.4: 828.70174217605669824291868741170590130253994090458, 
                 40.0: 816.89541503776589255680281543561359009195680451243}

# above paraeters should be given by users. Please choose the most suitable ones for your experiment.

delta = {"CMAG":delta_CMAG,"CMAG(E)":delta_CMAG_E}

#Calculate sd for CMAG mechanism.
print("calculate sd for CMAG...")

def Bf(d,e,s):
    value = norm.cdf(d/(2*s)-e*s)-math.exp(e*d)*norm.cdf(-d/(2*s)-e*s)
    return value

def B(d,e,delta):
    s = 1
    while Bf(d,e,s) >= delta:
        s += 1
    s = s-1
    i = 1
    while Bf(d,e,s+pow(10,-1)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-1)*i
    i = 1
    while Bf(d,e,s+pow(10,-2)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-2)*i
    i = 1
    while Bf(d,e,s+pow(10,-3)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-3)*i
    i = 1
    while Bf(d,e,s+pow(10,-4)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-4)*i
    i = 1
    while Bf(d,e,s+pow(10,-5)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-5)*i
    i = 1
    while Bf(d,e,s+pow(10,-6)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-6)*i
    i = 1
    while Bf(d,e,s+pow(10,-7)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-7)*i
    i = 1
    while Bf(d,e,s+pow(10,-8)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-8)*i
    i = 1
    while Bf(d,e,s+pow(10,-9)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-9)*i
    i = 1
    while Bf(d,e,s+pow(10,-10)*i) >= delta:
        i += 1
    s = s + pow(10,-10)*i
    return s

def sigma_CMAG(e):
    k = delta["CMAG"][e]
    X = list()
    for i in covering.keys():
        d = dist["CMAG"][i]
        n = len(covering[i])
        if d == 0.0:
            s = 0
        else:
            s = B(d,e,1/n**k)
        Y = [(j,s) for j in covering[i]]
        X = X + Y
    return (e,X)

def main():
    with Pool(32) as pool:
        Z = pool.map(sigma_CMAG,parameters)
    return Z

if __name__ == "__main__":
    result = main()
    
W = dict()

for e in parameters:
    L = [ x[1] for x in result if x[0] == e]
    W[e] = L[0]
    W[e] = dict(W[e])
    
with open("../data/sd_CMAG.pkl", "wb") as f:
    pickle.dump(W,f)

#Calculate sd for NADP mechanism.
print("calculate sd for CMAG(E)...")

v = "{}"

def Bf(d,e,s):
    d = mp.mpf(v.format(d))
    e = mp.mpf(v.format(e))
    s = mp.mpf(v.format(s))
    value = ncdf(d/(2*s)-e*s)-exp(e*d)*ncdf(-d/(2*s)-e*s)
    return value

def B(d,e,delta):
    delta = mp.mpf(v.format(delta))
    s = 1
    while Bf(d,e,s) >= delta:
        s += 1
    s = s-1
    i = 1
    while Bf(d,e,s+pow(10,-1)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-1)*i
    i = 1
    while Bf(d,e,s+pow(10,-2)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-2)*i
    i = 1
    while Bf(d,e,s+pow(10,-3)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-3)*i
    i = 1
    while Bf(d,e,s+pow(10,-4)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-4)*i
    i = 1
    while Bf(d,e,s+pow(10,-5)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-5)*i
    i = 1
    while Bf(d,e,s+pow(10,-6)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-6)*i
    i = 1
    while Bf(d,e,s+pow(10,-7)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-7)*i
    i = 1
    while Bf(d,e,s+pow(10,-8)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-8)*i
    i = 1
    while Bf(d,e,s+pow(10,-9)*i) >= delta:
        i += 1
    i = i-1
    s = s + pow(10,-9)*i
    i = 1
    while Bf(d,e,s+pow(10,-10)*i) >= delta:
        i += 1
    s = s + pow(10,-10)*i
    return s

def sigma_CMAG_E(e):
    k = delta["CMAG(E)"][e]
    X = list()
    for i in covering.keys():
        d = dist["CMAG(E)"][i]
        n = len(covering[i])
        n = mp.mpf(v.format(n))
        if d == 0.0:
            s = 0
        else:
            s = B(d,e,1/n**k)
        Y = [(j,s) for j in covering[i]]
        X = X + Y
    return (e,X)

mp.dps = 50
mp.pretty = True

def main():
    with Pool(32) as pool:
        Z = pool.map(sigma_CMAG_E,parameters)
    return Z

if __name__ == "__main__":
    result = main()
    
W = dict()

for e in parameters:
    L = [ x[1] for x in result if x[0] == e]
    W[e] = L[0]
    W[e] = dict(W[e])
    
with open("../data/sd_CMAG(E).pkl", "wb") as f:
    pickle.dump(W,f)
    
#Calculate sd for MAG mechanism.
print("calculate sd for MAG...")

sd_MAG = {e:pow(dim+1,1/2)/e for e in parameters}

with open("../data/sd_MAG.pkl", "wb") as f:
    pickle.dump(sd_MAG,f)
    
#Calculate sd for Laplacian mechanism.
print("calculate sd for Laplacian...")
    
pre_sd_Laplacian = {1.6: 53.9416915156, 3.2: 27.0898970402, 4.8: 18.1359192888, 6.4: 13.657645375, 8.0: 10.9699722136, 9.6: 9.1777215071, 11.2: 7.2971989919, 
                    12.8: 6.55654, 14.4: 5.88914, 16.0: 5.30153, 17.6: 4.89346, 19.2: 4.54805, 20.8: 4.16, 22.4: 3.84, 24.0: 3.65, 25.6: 3.44, 27.2: 3.25, 28.8: 3.1,
                    30.4: 2.96, 32.0: 2.82, 33.6: 2.7, 35.2: 2.58, 36.8: 2.47, 38.4: 2.39, 40.0: 2.26419}

# above paraeters should be given by users. Please choose the most suitable ones for your experiment.

sd_Laplacian = {e:pow(dim+1,1/2)/pre_sd_Laplacian[e] for e in parameters}

with open("../data/sd_Laplacian.pkl", "wb") as f:
    pickle.dump(sd_Laplacian,f)
